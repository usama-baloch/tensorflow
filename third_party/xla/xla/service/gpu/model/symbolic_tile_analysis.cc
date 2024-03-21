/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/gpu/model/symbolic_tile_analysis.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <stack>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/gpu/model/indexing_context.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/service/gpu/model/tile_analysis.h"
#include "xla/service/instruction_fusion.h"
#include "xla/status.h"

namespace xla {
namespace gpu {

namespace {

using ::mlir::AffineExpr;
using ::mlir::AffineMap;
using ::mlir::SmallVector;

enum class VisitState { kNew = 0, kVisiting = 1, kVisited = 2 };

}  // namespace

/*static*/ SymbolicTileAnalysisOrError SymbolicTileAnalysis::AnalyzeComputation(
    const HloComputation& computation, IndexingContext* ctx) {
  std::vector<std::unique_ptr<TiledHloInstruction>> tiled_hlo_instructions;
  absl::flat_hash_map<std::pair<const HloInstruction*, IndexingMap>,
                      TiledHloInstruction*>
      tiled_hlo_instructions_map;

  std::stack<TiledHloInstruction*> dfs_stack;
  absl::flat_hash_map<TiledHloInstruction*, int64_t> topological_order;
  absl::flat_hash_map<TiledHloInstruction*, VisitState> visit_states;

  // Create a new tiled hlo instruction or return existing instruciton from
  // cache for the given hlo and indexing map.
  auto get_tiled_hlo_instruction = [&](const HloInstruction* hlo,
                                       IndexingMap indexing_map)
      -> std::variant<TiledHloInstruction*, FusionDecision> {
    auto key = std::make_pair(hlo, indexing_map);

    auto it = tiled_hlo_instructions_map.find(key);
    if (it != tiled_hlo_instructions_map.end()) {
      return it->second;
    }

    // Bail out on instructions that are known to cause problems down the
    // line. This is not an inherent limitation of the approach, but simply
    // issues to be resolved in the current implementation.
    if (hlo->opcode() == HloOpcode::kDot ||
        hlo->opcode() == HloOpcode::kReshape ||
        hlo->opcode() == HloOpcode::kBitcast ||
        hlo->opcode() == HloOpcode::kConcatenate) {
      return FusionDecision{} << "Bailing out on " << hlo->ToString();
    }

    auto symbolic_tile = SymbolicTile::FromIndexingMap(key.second);
    if (!symbolic_tile.has_value()) {
      return FusionDecision{} << "Failed to compute symbolic tile for "
                              << indexing_map.ToString() << " for HLO "
                              << hlo->ToString();
    }

    tiled_hlo_instructions.push_back(std::make_unique<TiledHloInstruction>(
        hlo, std::move(indexing_map), std::move(*symbolic_tile)));

    auto tiled_hlo_instruction = tiled_hlo_instructions.back().get();
    tiled_hlo_instructions_map.emplace(key, tiled_hlo_instruction);
    return tiled_hlo_instruction;
  };

  // Find and assign operands to the tiled hlo instruction.
  auto assign_operands = [&](TiledHloInstruction* tiled_hlo_instruction) {
    std::optional<HloInstructionIndexing> operands_indexing =
        ComputeOutputToInputIndexing(tiled_hlo_instruction->instruction,
                                     /*output_id=*/0, ctx);

    if (!operands_indexing.has_value()) {
      return FusionDecision{} << "Failed to compute operands indexing for "
                              << tiled_hlo_instruction->instruction->ToString();
    }

    for (auto [operand, operand_indexing_map_set] :
         llvm::zip(tiled_hlo_instruction->instruction->operands(),
                   operands_indexing->indexing_maps)) {
      CHECK_EQ(operand_indexing_map_set.size(), 1);

      IndexingMap operand_indexing_map =
          ComposeIndexingMaps(tiled_hlo_instruction->indexing_map,
                              *operand_indexing_map_set.begin());

      auto tiled_operand_or =
          get_tiled_hlo_instruction(operand, std::move(operand_indexing_map));

      if (auto fusion_decison =
              std::get_if<FusionDecision>(&tiled_operand_or)) {
        return *fusion_decison;
      }

      tiled_hlo_instruction->operands.push_back(
          std::get<TiledHloInstruction*>(tiled_operand_or));
    }

    return FusionDecision{};
  };

  auto get_visit_state = [&](TiledHloInstruction* tiled_hlo_instruction) {
    if (visit_states.find(tiled_hlo_instruction) == visit_states.end()) {
      visit_states[tiled_hlo_instruction] = VisitState::kNew;
    }
    return visit_states[tiled_hlo_instruction];
  };

  const HloInstruction* root = computation.root_instruction();
  auto tiled_root =
      get_tiled_hlo_instruction(root, CreateIdentityMap(root->shape(), ctx));
  if (auto* fusion_decision = std::get_if<FusionDecision>(&tiled_root)) {
    return *fusion_decision;
  }

  dfs_stack.push(std::get<TiledHloInstruction*>(tiled_root));
  while (!dfs_stack.empty()) {
    TiledHloInstruction* current = dfs_stack.top();
    auto visit_state = get_visit_state(current);

    if (visit_state == VisitState::kNew) {
      visit_states[current] = VisitState::kVisiting;

      // First time processing the instruction.
      // Assign tiled operands. Create new tiled instructions if necessary.
      if (auto fusion_decision = assign_operands(current); !fusion_decision) {
        return fusion_decision;
      }

      for (auto* operand : current->operands) {
        dfs_stack.push(operand);
      }
    } else {
      if (visit_state == VisitState::kVisiting) {
        // All operands had been processed on assigned topological order.
        visit_states[current] = VisitState::kVisited;
        topological_order[current] = topological_order.size();
      }
      dfs_stack.pop();
    }
  }

  // Order instructions in def-before-use order.
  absl::c_sort(tiled_hlo_instructions, [&](const auto& i1, const auto& i2) {
    return topological_order[i1.get()] < topological_order[i2.get()];
  });

  return SymbolicTileAnalysis(std::move(tiled_hlo_instructions), ctx);
}

namespace {

std::vector<int64_t> EvaluateTileMap(AffineMap affine_map,
                                     absl::Span<int64_t const> parameters) {
  CHECK_EQ(affine_map.getNumSymbols(), parameters.size());
  CHECK_EQ(affine_map.getNumDims(), 0);

  SmallVector<AffineExpr> symbol_replacements = llvm::to_vector(
      llvm::map_range(parameters, [affine_map](const int64_t v) -> AffineExpr {
        return mlir::getAffineConstantExpr(v, affine_map.getContext());
      }));

  mlir::AffineMap simplified_affine_map =
      mlir::simplifyAffineMap(affine_map.replaceDimsAndSymbols(
          /*dimReplacements=*/{}, symbol_replacements, /*numResultDims=*/0,
          /*numResultSyms=*/0));

  SmallVector<int64_t> results = llvm::to_vector(llvm::map_range(
      simplified_affine_map.getResults(), [](AffineExpr result) -> int64_t {
        return llvm::cast<mlir::AffineConstantExpr>(result).getValue();
      }));

  return std::vector<int64_t>(results.begin(), results.end());
}

}  // namespace

std::vector<int64_t> SymbolicTileAnalysis::TileOffsets(
    const TiledHloInstruction* tiled_hlo_instruciton) const {
  CHECK(tile_parameters_.has_value());
  return EvaluateTileMap(tiled_hlo_instruciton->tile.offset_map(),
                         *tile_parameters_);
}

// TODO(bchetioui): remove dependency on stride and offset parameters.
std::vector<int64_t> SymbolicTileAnalysis::TileSizes(
    const TiledHloInstruction* tiled_hlo_instruciton) const {
  CHECK(tile_parameters_.has_value());
  return EvaluateTileMap(tiled_hlo_instruciton->tile.size_map(),
                         *tile_parameters_);
}

std::vector<int64_t> SymbolicTileAnalysis::TileStrides(
    const TiledHloInstruction* tiled_hlo_instruciton) const {
  CHECK(tile_parameters_.has_value());
  return EvaluateTileMap(tiled_hlo_instruciton->tile.stride_map(),
                         *tile_parameters_);
}

void SymbolicTileAnalysis::SetTileParameters(
    absl::Span<int64_t const> parameters) {
  // TODO(bchetioui): CHECK num parameters somehow?
  tile_parameters_ = std::vector(parameters.begin(), parameters.end());
}

void SymbolicTileAnalysis::SetTileParametersWithDefaultOffsetsAndStrides(
    absl::Span<int64_t const> sizes) {
  std::vector<int64_t> parameters;
  parameters.reserve(3 * sizes.size());

  for (int64_t size : sizes) {
    // Untiled dims have offset = 0 and stride = 1.
    parameters.push_back(0);
    parameters.push_back(size);
    parameters.push_back(1);
  }
  SetTileParameters(parameters);
}

}  // namespace gpu
}  // namespace xla
