
#ifndef MLIR_HELLO_PASSES_H
#define MLIR_HELLO_PASSES_H

#include <memory>

#include "mlir/Pass/Pass.h"

namespace hello {
// std::unique_ptr<mlir::Pass> createLowerToAffinePass();
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

} // namespace hello

#endif // MLIR_HELLO_PASSES_H