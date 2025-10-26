
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "Hello/HelloDialect.h"
#include "Hello/HelloOps.h"

using namespace mlir;
using namespace hello;

// Hello dialect.

#include "Hello/HelloOpsDialect.cpp.inc"

void HelloDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Hello/HelloOps.cpp.inc"
      >();
}