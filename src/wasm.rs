pub mod syntax {
    // Reference: https://webassembly.github.io/spec/core/syntax/index.html

    pub use index::*;
    pub use instructions::*;
    pub use module::*;
    pub use types::*;
    pub use values::*;

    pub mod values {
        pub type Byte = u8;
        pub type Name = String;
    }

    pub mod types {
        #[derive(Copy, Clone, Debug, PartialEq, Eq)]
        pub enum ValType {
            I32,
            I64,
            F32,
            F64,
        }

        impl std::fmt::Display for ValType {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                use ValType::*;
                match self {
                    I32 => write!(f, "i32"),
                    I64 => write!(f, "i64"),
                    F32 => write!(f, "f32"),
                    F64 => write!(f, "f64"),
                }
            }
        }

        #[derive(Debug, Clone, PartialEq, Eq)]
        pub struct ResultType(pub Vec<ValType>);

        #[derive(Debug, Clone, PartialEq, Eq)]
        pub struct FuncType {
            pub from: ResultType,
            pub to: ResultType,
        }

        #[derive(Debug)]
        pub struct Limits {
            pub min: u32,
            pub max: Option<u32>,
        }

        #[derive(Debug)]
        pub struct MemType(pub Limits);

        #[derive(Debug)]
        pub struct TableType(pub Limits, pub ElemType);
        #[derive(Debug)]
        pub enum ElemType {
            // XXX
            FuncRef,
        }

        #[derive(Debug)]
        pub struct GlobalType(pub Mut, pub ValType);
        #[derive(Copy, Clone, Debug, PartialEq, Eq)]
        pub enum Mut {
            Const,
            Var,
        }

        // We don't really use `ExternType`, although import and
        // exports can be typed via this, since we didn't require that
        // kind of type management.
        #[allow(dead_code)]
        pub enum ExternType {
            Func(FuncType),
            Table(TableType),
            Mem(MemType),
            Global(GlobalType),
        }
    }

    pub mod index {
        #[derive(Copy, Clone, Debug, PartialEq, Eq)]
        pub struct TypeIdx(pub u32);
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
        pub struct FuncIdx(pub u32);
        #[derive(Copy, Clone, Debug)]
        pub struct TableIdx(pub u32);
        #[derive(Copy, Clone, Debug)]
        pub struct MemIdx(pub u32);
        #[derive(Copy, Clone, Debug)]
        pub struct GlobalIdx(pub u32);
        #[derive(Copy, Clone, Debug)]
        pub struct LocalIdx(pub u32);
        #[derive(Copy, Clone, Debug)]
        pub struct LabelIdx(pub u32);
    }

    pub mod instructions {
        use super::index::*;
        use super::types::*;

        #[derive(Clone, Copy, Debug)]
        pub enum PackSize {
            Pack8,
            Pack16,
            Pack32,
        }

        impl std::fmt::Display for PackSize {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                use PackSize::*;
                match self {
                    Pack8 => write!(f, "8"),
                    Pack16 => write!(f, "16"),
                    Pack32 => write!(f, "32"),
                }
            }
        }

        #[rustfmt::skip]
        pub mod intop {
            use super::PackSize;

            #[derive(Clone, Copy, Debug)]
            pub enum UnOp { Clz, Ctz, Popcnt, ExtendS(PackSize), }
            #[derive(Clone, Copy, Debug)]
            pub enum BinOp { Add, Sub, Mul, DivS, DivU, RemS, RemU, And, Or,
                             Xor, Shl, ShrS, ShrU, Rotl, Rotr, }
            #[derive(Clone, Copy, Debug)]
            pub enum TestOp { Eqz, }
            #[derive(Clone, Copy, Debug)]
            pub enum RelOp { Eq, Ne, LtS, LtU, GtS, GtU, LeS, LeU, GeS, GeU, }
            #[derive(Clone, Copy, Debug)]
            pub enum CvtOp { ExtendSI32, ExtendUI32, WrapI64, TruncSF32,
                             TruncUF32, TruncSF64, TruncUF64, TruncSatSF32,
                             TruncSatUF32, TruncSatSF64, TruncSatUF64, ReinterpretFloat, }
        }

        #[rustfmt::skip]
        pub mod floatop {
            #[derive(Clone, Copy, Debug)]
            pub enum UnOp { Neg, Abs, Ceil, Floor, Trunc, Nearest, Sqrt, }
            #[derive(Clone, Copy, Debug)]
            pub enum BinOp { Add, Sub, Mul, Div, Min, Max, CopySign, }
            #[derive(Clone, Copy, Debug)]
            pub enum RelOp { Eq, Ne, Lt, Gt, Le, Ge, }
            #[derive(Clone, Copy, Debug)]
            pub enum CvtOp { ConvertSI32, ConvertUI32, ConvertSI64, ConvertUI64,
                             PromoteF32, DemoteF64, ReinterpretInt, }
        }

        #[derive(Clone, Copy, Debug)]
        pub enum Const {
            I32(i32),
            I64(i64),
            F32(f32),
            F64(f64),
        }

        impl std::fmt::Display for Const {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                use std::num::FpCategory::*;
                use Const::*;
                match self {
                    I32(v) => write!(f, "{}i32", v),
                    I64(v) => write!(f, "{}i64", v),
                    F32(v) => match v.classify() {
                        Normal | Zero => write!(f, "{}f32", v),
                        Nan => write!(f, "f32::NAN"),
                        Infinite => {
                            if v.is_sign_positive() {
                                write!(f, "f32::INFINITY")
                            } else {
                                write!(f, "f32::NEG_INFINITY")
                            }
                        }
                        Subnormal => write!(f, "subnormal-f32-{:#x}", v.to_bits()),
                    },
                    F64(v) => match v.classify() {
                        Normal | Zero => write!(f, "{}f64", v),
                        Nan => write!(f, "f64::NAN"),
                        Infinite => {
                            if v.is_sign_positive() {
                                write!(f, "f64::INFINITY")
                            } else {
                                write!(f, "f64::NEG_INFINITY")
                            }
                        }
                        Subnormal => write!(f, "subnormal-f64-{:#x}", v.to_bits()),
                    },
                }
            }
        }

        #[derive(Clone, Copy, Debug)]
        pub enum BitSize {
            B32,
            B64,
        }

        impl std::fmt::Display for BitSize {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                use BitSize::*;
                match self {
                    B32 => write!(f, "32"),
                    B64 => write!(f, "64"),
                }
            }
        }

        #[derive(Clone, Copy, Debug)]
        pub enum BlockType {
            TypeIdx(TypeIdx),
            ValType(Option<ValType>),
        }

        #[derive(Clone, Copy, Debug)]
        pub struct MemArg {
            pub offset: u32,
            pub align: u32,
        }

        #[derive(Clone, Copy, Debug)]
        pub enum SX {
            U,
            S,
        }

        impl std::fmt::Display for SX {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                match self {
                    SX::U => write!(f, "u"),
                    SX::S => write!(f, "i"),
                }
            }
        }

        #[derive(Clone, Copy, Debug)]
        pub struct MemLoad {
            pub typ: ValType,
            pub extend: Option<(usize, SX)>, // bit width, sign extension
            pub memarg: MemArg,
        }

        #[derive(Clone, Copy, Debug)]
        pub struct MemStore {
            pub typ: ValType,
            pub memarg: MemArg,
            pub bitwidth: Option<usize>,
        }

        #[derive(Clone, Debug)]
        pub enum Instr {
            // Numeric instructions
            Const(Const),
            IUnOp(BitSize, intop::UnOp),
            FUnOp(BitSize, floatop::UnOp),
            IBinOp(BitSize, intop::BinOp),
            FBinOp(BitSize, floatop::BinOp),
            ITestOp(BitSize, intop::TestOp),
            IRelOp(BitSize, intop::RelOp),
            FRelOp(BitSize, floatop::RelOp),
            ICvtOp(BitSize, intop::CvtOp),
            FCvtOp(BitSize, floatop::CvtOp),

            // Parametric instructions
            Drop,
            Select,

            // Variable instructions
            LocalGet(LocalIdx),
            LocalSet(LocalIdx),
            LocalTee(LocalIdx),
            GlobalGet(GlobalIdx),
            GlobalSet(GlobalIdx),

            // Memory instructions
            MemLoad(MemLoad),
            MemStore(MemStore),
            MemSize,
            MemGrow,

            // Control instructions
            Nop,
            Unreachable,
            Block(BlockType, Vec<Instr>),
            Loop(BlockType, Vec<Instr>),
            If(BlockType, Vec<Instr>, Vec<Instr>),
            Br(LabelIdx),
            BrIf(LabelIdx),
            BrTable(Vec<LabelIdx>, LabelIdx),
            Return,
            Call(FuncIdx),
            CallIndirect(TypeIdx),
        }

        #[derive(Debug)]
        pub struct Expr(pub Vec<Instr>);
    }

    pub mod module {
        use super::index::*;
        use super::instructions::*;
        use super::types::*;
        use super::values::*;

        #[derive(Debug)]
        pub struct Module {
            pub types: Vec<FuncType>,
            pub funcs: Vec<Func>,
            pub tables: Vec<Table>,
            pub mems: Vec<Mem>,
            pub globals: Vec<Global>,
            pub elem: Vec<Elem>,
            pub data: Vec<Data>,
            pub start: Option<Start>,
            pub imports: Vec<Import>,
            pub exports: Vec<Export>,
            pub names: Names, // non-standard custom section that appears in WASI
        }

        #[derive(Debug)]
        pub struct Names {
            pub module: Option<Name>,
            pub functions: std::collections::HashMap<FuncIdx, Name>,
            pub locals: std::collections::HashMap<FuncIdx, Vec<Name>>,
        }

        #[derive(Debug)]
        pub enum FuncInternals {
            LocalFunc { locals: Vec<ValType>, body: Expr },
            ImportedFunc { module: Name, name: Name },
        }

        #[derive(Debug)]
        pub struct Func {
            pub typ: TypeIdx,
            pub internals: FuncInternals,
        }

        #[derive(Debug)]
        pub struct Table {
            pub typ: TableType,
        }

        #[derive(Debug)]
        pub struct Mem {
            pub typ: MemType,
        }

        #[derive(Debug)]
        pub struct Global {
            pub typ: GlobalType,
            pub init: Expr,
        }

        #[derive(Debug)]
        pub struct Elem {
            pub table: TableIdx,
            pub offset: Expr,
            pub init: Vec<FuncIdx>,
        }

        #[derive(Debug)]
        pub struct Data {
            pub data: MemIdx,
            pub offset: Expr,
            pub init: Vec<Byte>,
        }

        #[derive(Debug)]
        pub struct Start {
            pub func: FuncIdx,
        }

        #[derive(Debug)]
        pub struct Export {
            pub name: Name,
            pub desc: ExportDesc,
        }
        #[derive(Debug)]
        pub enum ExportDesc {
            Func(FuncIdx),
            Table(TableIdx),
            Mem(MemIdx),
            Global(GlobalIdx),
        }

        #[derive(Debug)]
        pub struct Import {
            pub module: Name,
            pub name: Name,
            pub desc: ImportDesc,
        }
        #[derive(Debug)]
        pub enum ImportDesc {
            Func(TypeIdx),
            Table(TableType),
            Mem(MemType),
            Global(GlobalType),
        }
    }
}
