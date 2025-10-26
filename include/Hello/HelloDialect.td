
#ifndef HELLO_DIALECT
#define HELLO_DIALECT

include "mlir/IR/OpBase.td"

// Hello dialect definition.

def Hello_Dialect2 : Dialect {
    let name = "hello";
    let summary = "A hello out-of-tree MLIR dialect.";
    let description = [{
        This dialect is minimal example to implement hello-world for MLIR.
    }];

    let cppNamespace = "::hello";
    // let hasConstantMaterializer = 1;
}

// Base hello operation definition.

// mnemonic 指名字
class Hello_Op<string mnemonic, list<Trait> traits = []> :
    Op<Hello_Dialect2, mnemonic, traits>;

#endif // HELLO_DIALECT
