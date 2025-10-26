
add_mlir_tool(hello-opt hello-opt.cpp DEPENDS MLIRHelloOpsIncGen)

target_link_libraries(hello-opt
    PRIVATE
    MLIRIR
    MLIRParser
    MLIRSupport
    MLIROptLib        # 对应 #include "mlir/Tools/mlir-opt/MlirOptMain.h"
    MLIRFuncDialect   # 对应 #include "mlir/Dialect/Func/IR/FuncOps.h"
    MLIRTransforms    # 对应 #include "mlir/Transforms/Passes.h"
)

llvm_update_compile_flags(hello-opt)
mlir_check_all_link_libraries(hello-opt)