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
#include <cstddef>
#include <memory>
#include <string>

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/passes.h"

#define DEBUG_TYPE "composite-to-custom"

namespace mlir {
namespace odml {

#define GEN_PASS_DEF_LEGALIZECOMPOSITETOCUSTOMOPPASS
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/passes.h.inc"

namespace {
bool IsSupportedComposite(::mlir::stablehlo::CompositeOp op) {
  // List of supported composites to represent using CustomOp.
  return llvm::is_contained(
      {"odml.update_kv_cache", "odml.scaled_dot_product_attention"},
      op.getName());
}

TFL::ConstBytesAttr CustomOption(OpBuilder* builder,
                                 const std::string& content) {
  return TFL::ConstBytesAttr::get(builder->getContext(),
                                  StringRef(content.data(), content.size()));
}

LogicalResult BuildOption(flexbuffers::Builder* fbb, Operation* op,
                          NamedAttribute pair) {
  const char* key = pair.getName().data();
  const auto attr = pair.getValue();

  if (attr.isa<::mlir::IntegerAttr>()) {
    fbb->Int(key, attr.dyn_cast<mlir::IntegerAttr>().getInt());
    return success();
  }

  if (attr.isa<::mlir::FloatAttr>()) {
    fbb->Double(key, attr.dyn_cast<mlir::FloatAttr>().getValueAsDouble());
    return success();
  }

  // default
  return op->emitWarning("serialization not supported for : ") << key;
}

}  // namespace

struct LegalizeCompositeToCustomOpPass
    : public impl::LegalizeCompositeToCustomOpPassBase<
          LegalizeCompositeToCustomOpPass> {
  using LegalizeCompositeToCustomOpPassBase::
      LegalizeCompositeToCustomOpPassBase;

  void runOnOperation() override {
    func::FuncOp fn = getOperation();
    OpBuilder builder(fn.getContext());

    fn.walk([&](Operation* op) {
      // Process only StableHLO composite ops.
      auto composite = llvm::dyn_cast<stablehlo::CompositeOp>(op);
      if (!composite || !IsSupportedComposite(composite)) return;

      // stablehlo.composite "odml.some_op" <args> {composite_attrs = <attrs> }
      // ==> tfl.custom(<args>) { name = "odml.some_op", <attrs...> }
      StringRef custom_op_name = composite.getName();
      SmallVector<NamedAttribute> options =
          llvm::to_vector(composite.getCompositeAttributes());

      // Build flexbuffer options.
      std::string custom_option_buffer;
      auto fbb = std::make_unique<flexbuffers::Builder>();
      size_t map_start = fbb->StartMap();
      for (auto pair : options) {
        // Allows silently skipping unsupported attributes.
        (void)BuildOption(fbb.get(), op, pair);
      }
      fbb->EndMap(map_start);
      fbb->Finish();
      custom_option_buffer.assign(fbb->GetBuffer().begin(),
                                  fbb->GetBuffer().end());

      // Build TFL custom op, replace composite with custom op.
      builder.setInsertionPoint(op);
      auto tfl_custom_op = builder.create<TFL::CustomOp>(
          op->getLoc(), op->getResultTypes(), op->getOperands(), custom_op_name,
          CustomOption(&builder, custom_option_buffer));
      op->replaceAllUsesWith(tfl_custom_op);
      op->erase();
    });
  }
};

static PassRegistration<LegalizeCompositeToCustomOpPass> pass_v2s;

}  // namespace odml
}  // namespace mlir
