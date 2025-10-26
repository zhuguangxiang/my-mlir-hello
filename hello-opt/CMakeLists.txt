
add_mlir_tool(hello-opt hello-opt.cpp DEPENDS MLIRHello)

target_link_libraries(hello-opt
    PRIVATE
    MLIRHello
    MLIRIR
    MLIRParser
    MLIRSupport
    MLIROptLib        # 对应 #include "mlir/Tools/mlir-opt/MlirOptMain.h"
    MLIRFuncDialect   # 对应 #include "mlir/Dialect/Func/IR/FuncOps.h"
    MLIRTransforms    # 对应 #include "mlir/Transforms/Passes.h"
    MLIRAnalysis
    MLIRCallInterfaces
    MLIRCastInterfaces
    MLIRExecutionEngine
    MLIRIR
    MLIRLLVMCommonConversion
    MLIRLLVMToLLVMIRTranslation
    MLIRMemRefDialect
    MLIRLLVMDialect
    MLIRParser
    MLIRPass
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRTargetLLVMIRExport
    MLIRTransforms
    MLIROptLib
    MLIRSCFDialect
)

llvm_update_compile_flags(hello-opt)
mlir_check_all_link_libraries(hello-opt)