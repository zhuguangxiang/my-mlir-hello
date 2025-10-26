
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
// 导入 Func Dialect
#include "mlir/Dialect/Func/IR/FuncOps.h"
// 导入 MLIR 自带 Pass
#include "mlir/Transforms/Passes.h"
// 导入我们新建的 Dialect
#include "Hello/HelloDialect.h"
#include "Hello/HelloOps.h"

#include "Hello/HelloOpsDialect.cpp.inc"
#define GET_OP_CLASSES
#include "Hello/HelloOps.cpp.inc"

using namespace mlir;
using namespace llvm;
using namespace hello;

void HelloDialect2::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Hello/HelloOps.cpp.inc"
      >();
}

int main(int argc, char **argv) {
  DialectRegistry registry;
  // 注册 Dialect
  registry.insert<hello::HelloDialect2, func::FuncDialect>();
  // 注册两个 Pass
  registerCSEPass();
  registerCanonicalizerPass();
  return asMainReturnCode(MlirOptMain(argc, argv, "hello-opt", registry));
}