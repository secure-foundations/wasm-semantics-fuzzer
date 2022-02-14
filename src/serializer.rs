use crate::wasm::syntax::{self, *};

struct Stream {
    base: Vec<u8>,
}

macro_rules! generate {
    ($name:ident : $ty:ty = $body:expr) => {
        impl Stream {
            fn $name(&mut self, v: $ty) -> &mut Self {
                #[allow(unused_macros)]
                macro_rules! s {
                    () => {
                        self
                    };
                }
                #[allow(unused_macros)]
                macro_rules! v {
                    () => {
                        v
                    };
                }
                $body
            }
        }
    };
}

impl Stream {
    fn new() -> Self {
        Self { base: vec![] }
    }

    fn extract(&self) -> &[u8] {
        self.base.as_slice()
    }
}

generate! {bytes : Vec<u8> = {
    let mut v = v!();
    s!().base.append(&mut v);
    s!()
}}

generate! {leb128_u : u64 = {
    let mut value = v!();
    loop {
        let mut byte: u8 = value as u8 & 0b111_1111; // lower order 7 bits
        value >>= 7;
        if value != 0 {
            // more bits to come. set high order bit of byte
            byte |= 0b1000_0000;
        }
        s!().byte(byte);
        if value == 0 {
            break;
        }
    }
    s!()
}}
generate! {leb128_s : i64 = {
    let mut value = v!();
    let mut more = true;
    while more {
        let mut byte: u8 = value as u8 & 0b111_1111; // lower order 7 bits
        value >>= 7; // arithmetic shift right, since `value` is signed

        // sign bit of byte is second high order bit (0x40)
        if (value == 0 && (byte & 0x40) == 0) || (value == -1 && (byte & 0x40) != 0) {
            more = false;
        } else {
            byte |= 0b1000_0000;
        }
        s!().byte(byte);
    }
    s!()
}}

#[cfg(test)]
mod test_leb128 {
    #[test]
    fn leb128_u_spot_tests() {
        use super::Stream;
        assert_eq!(Stream::new().leb128_u(0).extract(), [0x00]);
        assert_eq!(Stream::new().leb128_u(1).extract(), [0x01]);
        assert_eq!(Stream::new().leb128_u(123456).extract(), [0xc0, 0xc4, 0x07]);
    }

    #[test]
    fn leb128_s_spot_tests() {
        use super::Stream;
        assert_eq!(Stream::new().leb128_s(0).extract(), [0x00]);
        assert_eq!(Stream::new().leb128_s(127).extract(), [0xff, 0x00]);
        assert_eq!(
            Stream::new().leb128_s(-123456).extract(),
            [0xc0, 0xbb, 0x78]
        );
    }
}

generate! {u32 : u32 = s!().leb128_u(v!() as _)}
generate! {u64 : u64 = s!().leb128_u(v!())}
generate! {i32 : i32 = s!().leb128_s(v!() as _)}
generate! {i64 : i64 = s!().leb128_s(v!())}

generate! {f32 : f32 = s!().bytes(v!().to_le_bytes().into())}
generate! {f64 : f64 = s!().bytes(v!().to_le_bytes().into())}

generate! {s33 : i64 = s!().leb128_s(v!())}

impl Stream {
    fn vec_no_lenprefix<T, F>(&mut self, v: Vec<T>, elem: F) -> &mut Self
    where
        F: Fn(&mut Self, T) -> &mut Self,
    {
        v.into_iter().for_each(|x| {
            elem(self, x);
        });
        self
    }

    fn vec<T, F>(&mut self, v: Vec<T>, elem: F) -> &mut Self
    where
        F: Fn(&mut Self, T) -> &mut Self,
    {
        self.u32(v.len() as u32);
        self.vec_no_lenprefix(v, elem)
    }
}

generate! {byte : u8 = {s!().base.push(v!()); s!()}}

generate! {name: String = s!().vec(v!().bytes().collect(), Stream::byte)}

generate! {valtype: ValType = s!().byte(match v!() {
    ValType::I32 => 0x7f,
    ValType::I64 => 0x7e,
    ValType::F32 => 0x7d,
    ValType::F64 => 0x7c,
})}

generate! {resulttype: ResultType = s!().vec(v!().0, Stream::valtype)}

generate! {functype: FuncType = s!().byte(0x60).resulttype(v!().from).resulttype(v!().to)}

generate! {limits: Limits = match v!().max {
    None => s!().byte(0).u32(v!().min),
    Some(max) => s!().byte(1).u32(v!().min).u32(max),
}}

generate! {memtype: MemType = s!().limits(v!().0)}

generate! {elemtype: ElemType = {
    // There is only one possible ElemType, so this is
    // irrefutable. However, it is inserted to make sure to
    // complain if wasm::syntax ever changes.
    let ElemType::FuncRef = v!();
    s!().byte(0x70)
}}

generate! {tabletype: TableType = {
    // Yes, it is elemtype first here
    s!().elemtype(v!().1).limits(v!().0)}
}

generate! {globaltype: GlobalType = s!().valtype(v!().1).byte(match v!().0 {
    Mut::Const => 0,
    Mut::Var => 1,
})}

generate! {blocktype: BlockType = match v!() {
    BlockType::ValType(None) => s!().byte(0x40),
    BlockType::ValType(Some(v)) => s!().valtype(v),
    BlockType::TypeIdx(TypeIdx(v)) => s!().s33(v as i64),
}}

generate! {instr : Instr = {
    use Instr::*;
    match v!() {
        // Control instructions
        Unreachable => s!().byte(0x00),
        Nop => s!().byte(0x01),
        Block(bt, ins) => s!().byte(0x02)
            .blocktype(bt).vec_no_lenprefix(ins, Stream::instr).byte(0x0b),
        Loop(bt, ins) => s!().byte(0x03)
            .blocktype(bt).vec_no_lenprefix(ins, Stream::instr).byte(0x0b),
        If(bt, ins1, ins2) => {
            s!().byte(0x04).blocktype(bt).vec_no_lenprefix(ins1, Stream::instr);
            if ins2.len() > 0 {
            s!().byte(0x05).vec_no_lenprefix(ins2, Stream::instr);
            }
            s!().byte(0x0b)
        },
        Br(lbl) => s!().byte(0x0c).labelidx(lbl),
        BrIf(lbl) => s!().byte(0x0d).labelidx(lbl),
        BrTable(ls, ln) => s!().byte(0x0e).vec(ls, Stream::labelidx).labelidx(ln),
        Return => s!().byte(0x0f),
        Call(f) => s!().byte(0x10).funcidx(f),
        CallIndirect(x) => s!().byte(0x11).typeidx(x).byte(0x00),

        // Parametric instructions
        Drop => s!().byte(0x1a),
        Select => s!().byte(0x1b),

        // Variable instructions
        LocalGet(x) => s!().byte(0x20).localidx(x),
        LocalSet(x) => s!().byte(0x21).localidx(x),
        LocalTee(x) => s!().byte(0x22).localidx(x),
        GlobalGet(x) => s!().byte(0x23).globalidx(x),
        GlobalSet(x) => s!().byte(0x24).globalidx(x),

        // Memory instructions
        MemLoad(syntax::MemLoad { typ, extend: None, memarg }) =>
            s!().byte(match typ {
                ValType::I32 => 0x28,
                ValType::I64 => 0x29,
                ValType::F32 => 0x2a,
                ValType::F64 => 0x2b,
            }).memarg(memarg),
        MemLoad(syntax::MemLoad { typ, extend: Some((bw, sx)), memarg }) =>
            s!().byte(match (typ, bw, sx) {
                (ValType::I32, 8, SX::S) => 0x2c,
                (ValType::I32, 8, SX::U) => 0x2d,
                (ValType::I32, 16, SX::S) => 0x2e,
                (ValType::I32, 16, SX::U) => 0x2f,
                (ValType::I64, 8, SX::S) => 0x30,
                (ValType::I64, 8, SX::U) => 0x31,
                (ValType::I64, 16, SX::S) => 0x32,
                (ValType::I64, 16, SX::U) => 0x33,
                (ValType::I64, 32, SX::S) => 0x34,
                (ValType::I64, 32, SX::U) => 0x35,
                _ => panic!("Invalid MemLoad"),
            }).memarg(memarg),
        MemStore(syntax::MemStore { typ, bitwidth: None, memarg }) =>
            s!().byte(match typ {
                ValType::I32 => 0x36,
                ValType::I64 => 0x37,
                ValType::F32 => 0x38,
                ValType::F64 => 0x39,
            }).memarg(memarg),
        MemStore(syntax::MemStore { typ, bitwidth: Some(bw), memarg }) =>
            s!().byte(match (typ, bw) {
                (ValType::I32, 8) => 0x3a,
                (ValType::I32, 16) => 0x3b,
                (ValType::I64, 8) => 0x3c,
                (ValType::I64, 16) => 0x3d,
                (ValType::I64, 32) => 0x3e,
                _ => panic!("Invalid MemStore"),
            }).memarg(memarg),
        MemSize => s!().byte(0x3f).byte(0x00),
        MemGrow => s!().byte(0x40).byte(0x00),

        // Numeric instructions
        Const(syntax::Const::I32(x)) => s!().byte(0x41).i32(x),
        Const(syntax::Const::I64(x)) => s!().byte(0x42).i64(x),
        Const(syntax::Const::F32(x)) => s!().byte(0x43).f32(x),
        Const(syntax::Const::F64(x)) => s!().byte(0x44).f64(x),

        ITestOp(BitSize::B32, intop::TestOp::Eqz) => s!().byte(0x45),
        IRelOp(BitSize::B32, intop::RelOp::Eq) => s!().byte(0x46),
        IRelOp(BitSize::B32, intop::RelOp::Ne) => s!().byte(0x47),
        IRelOp(BitSize::B32, intop::RelOp::LtS) => s!().byte(0x48),
        IRelOp(BitSize::B32, intop::RelOp::LtU) => s!().byte(0x49),
        IRelOp(BitSize::B32, intop::RelOp::GtS) => s!().byte(0x4a),
        IRelOp(BitSize::B32, intop::RelOp::GtU) => s!().byte(0x4b),
        IRelOp(BitSize::B32, intop::RelOp::LeS) => s!().byte(0x4c),
        IRelOp(BitSize::B32, intop::RelOp::LeU) => s!().byte(0x4d),
        IRelOp(BitSize::B32, intop::RelOp::GeS) => s!().byte(0x4e),
        IRelOp(BitSize::B32, intop::RelOp::GeU) => s!().byte(0x4f),

        ITestOp(BitSize::B64, intop::TestOp::Eqz) => s!().byte(0x50),
        IRelOp(BitSize::B64, intop::RelOp::Eq) => s!().byte(0x51),
        IRelOp(BitSize::B64, intop::RelOp::Ne) => s!().byte(0x52),
        IRelOp(BitSize::B64, intop::RelOp::LtS) => s!().byte(0x53),
        IRelOp(BitSize::B64, intop::RelOp::LtU) => s!().byte(0x54),
        IRelOp(BitSize::B64, intop::RelOp::GtS) => s!().byte(0x55),
        IRelOp(BitSize::B64, intop::RelOp::GtU) => s!().byte(0x56),
        IRelOp(BitSize::B64, intop::RelOp::LeS) => s!().byte(0x57),
        IRelOp(BitSize::B64, intop::RelOp::LeU) => s!().byte(0x58),
        IRelOp(BitSize::B64, intop::RelOp::GeS) => s!().byte(0x59),
        IRelOp(BitSize::B64, intop::RelOp::GeU) => s!().byte(0x5a),

        FRelOp(BitSize::B32, floatop::RelOp::Eq) => s!().byte(0x5b),
        FRelOp(BitSize::B32, floatop::RelOp::Ne) => s!().byte(0x5c),
        FRelOp(BitSize::B32, floatop::RelOp::Lt) => s!().byte(0x5d),
        FRelOp(BitSize::B32, floatop::RelOp::Gt) => s!().byte(0x5e),
        FRelOp(BitSize::B32, floatop::RelOp::Le) => s!().byte(0x5f),
        FRelOp(BitSize::B32, floatop::RelOp::Ge) => s!().byte(0x60),

        FRelOp(BitSize::B64, floatop::RelOp::Eq) => s!().byte(0x61),
        FRelOp(BitSize::B64, floatop::RelOp::Ne) => s!().byte(0x62),
        FRelOp(BitSize::B64, floatop::RelOp::Lt) => s!().byte(0x63),
        FRelOp(BitSize::B64, floatop::RelOp::Gt) => s!().byte(0x64),
        FRelOp(BitSize::B64, floatop::RelOp::Le) => s!().byte(0x65),
        FRelOp(BitSize::B64, floatop::RelOp::Ge) => s!().byte(0x66),

        IUnOp(BitSize::B32, intop::UnOp::Clz) => s!().byte(0x67),
        IUnOp(BitSize::B32, intop::UnOp::Ctz) => s!().byte(0x68),
        IUnOp(BitSize::B32, intop::UnOp::Popcnt) => s!().byte(0x69),
        IBinOp(BitSize::B32, intop::BinOp::Add) => s!().byte(0x6a),
        IBinOp(BitSize::B32, intop::BinOp::Sub) => s!().byte(0x6b),
        IBinOp(BitSize::B32, intop::BinOp::Mul) => s!().byte(0x6c),
        IBinOp(BitSize::B32, intop::BinOp::DivS) => s!().byte(0x6d),
        IBinOp(BitSize::B32, intop::BinOp::DivU) => s!().byte(0x6e),
        IBinOp(BitSize::B32, intop::BinOp::RemS) => s!().byte(0x6f),
        IBinOp(BitSize::B32, intop::BinOp::RemU) => s!().byte(0x70),
        IBinOp(BitSize::B32, intop::BinOp::And) => s!().byte(0x71),
        IBinOp(BitSize::B32, intop::BinOp::Or) => s!().byte(0x72),
        IBinOp(BitSize::B32, intop::BinOp::Xor) => s!().byte(0x73),
        IBinOp(BitSize::B32, intop::BinOp::Shl) => s!().byte(0x74),
        IBinOp(BitSize::B32, intop::BinOp::ShrS) => s!().byte(0x75),
        IBinOp(BitSize::B32, intop::BinOp::ShrU) => s!().byte(0x76),
        IBinOp(BitSize::B32, intop::BinOp::Rotl) => s!().byte(0x77),
        IBinOp(BitSize::B32, intop::BinOp::Rotr) => s!().byte(0x78),

        IUnOp(BitSize::B64, intop::UnOp::Clz) => s!().byte(0x79),
        IUnOp(BitSize::B64, intop::UnOp::Ctz) => s!().byte(0x7a),
        IUnOp(BitSize::B64, intop::UnOp::Popcnt) => s!().byte(0x7b),
        IBinOp(BitSize::B64, intop::BinOp::Add) => s!().byte(0x7c),
        IBinOp(BitSize::B64, intop::BinOp::Sub) => s!().byte(0x7d),
        IBinOp(BitSize::B64, intop::BinOp::Mul) => s!().byte(0x7e),
        IBinOp(BitSize::B64, intop::BinOp::DivS) => s!().byte(0x7f),
        IBinOp(BitSize::B64, intop::BinOp::DivU) => s!().byte(0x80),
        IBinOp(BitSize::B64, intop::BinOp::RemS) => s!().byte(0x81),
        IBinOp(BitSize::B64, intop::BinOp::RemU) => s!().byte(0x82),
        IBinOp(BitSize::B64, intop::BinOp::And) => s!().byte(0x83),
        IBinOp(BitSize::B64, intop::BinOp::Or) => s!().byte(0x84),
        IBinOp(BitSize::B64, intop::BinOp::Xor) => s!().byte(0x85),
        IBinOp(BitSize::B64, intop::BinOp::Shl) => s!().byte(0x86),
        IBinOp(BitSize::B64, intop::BinOp::ShrS) => s!().byte(0x87),
        IBinOp(BitSize::B64, intop::BinOp::ShrU) => s!().byte(0x88),
        IBinOp(BitSize::B64, intop::BinOp::Rotl) => s!().byte(0x89),
        IBinOp(BitSize::B64, intop::BinOp::Rotr) => s!().byte(0x8a),

        FUnOp(BitSize::B32, floatop::UnOp::Abs) => s!().byte(0x8b),
        FUnOp(BitSize::B32, floatop::UnOp::Neg) => s!().byte(0x8c),
        FUnOp(BitSize::B32, floatop::UnOp::Ceil) => s!().byte(0x8d),
        FUnOp(BitSize::B32, floatop::UnOp::Floor) => s!().byte(0x8e),
        FUnOp(BitSize::B32, floatop::UnOp::Trunc) => s!().byte(0x8f),
        FUnOp(BitSize::B32, floatop::UnOp::Nearest) => s!().byte(0x90),
        FUnOp(BitSize::B32, floatop::UnOp::Sqrt) => s!().byte(0x91),
        FBinOp(BitSize::B32, floatop::BinOp::Add) => s!().byte(0x92),
        FBinOp(BitSize::B32, floatop::BinOp::Sub) => s!().byte(0x93),
        FBinOp(BitSize::B32, floatop::BinOp::Mul) => s!().byte(0x94),
        FBinOp(BitSize::B32, floatop::BinOp::Div) => s!().byte(0x95),
        FBinOp(BitSize::B32, floatop::BinOp::Min) => s!().byte(0x96),
        FBinOp(BitSize::B32, floatop::BinOp::Max) => s!().byte(0x97),
        FBinOp(BitSize::B32, floatop::BinOp::CopySign) => s!().byte(0x98),

        FUnOp(BitSize::B64, floatop::UnOp::Abs) => s!().byte(0x99),
        FUnOp(BitSize::B64, floatop::UnOp::Neg) => s!().byte(0x9a),
        FUnOp(BitSize::B64, floatop::UnOp::Ceil) => s!().byte(0x9b),
        FUnOp(BitSize::B64, floatop::UnOp::Floor) => s!().byte(0x9c),
        FUnOp(BitSize::B64, floatop::UnOp::Trunc) => s!().byte(0x9d),
        FUnOp(BitSize::B64, floatop::UnOp::Nearest) => s!().byte(0x9e),
        FUnOp(BitSize::B64, floatop::UnOp::Sqrt) => s!().byte(0x9f),
        FBinOp(BitSize::B64, floatop::BinOp::Add) => s!().byte(0xa0),
        FBinOp(BitSize::B64, floatop::BinOp::Sub) => s!().byte(0xa1),
        FBinOp(BitSize::B64, floatop::BinOp::Mul) => s!().byte(0xa2),
        FBinOp(BitSize::B64, floatop::BinOp::Div) => s!().byte(0xa3),
        FBinOp(BitSize::B64, floatop::BinOp::Min) => s!().byte(0xa4),
        FBinOp(BitSize::B64, floatop::BinOp::Max) => s!().byte(0xa5),
        FBinOp(BitSize::B64, floatop::BinOp::CopySign) => s!().byte(0xa6),

        ICvtOp(BitSize::B32, intop::CvtOp::WrapI64) => s!().byte(0xa7),
        ICvtOp(BitSize::B32, intop::CvtOp::TruncSF32) => s!().byte(0xa8),
        ICvtOp(BitSize::B32, intop::CvtOp::TruncUF32) => s!().byte(0xa9),
        ICvtOp(BitSize::B32, intop::CvtOp::TruncSF64) => s!().byte(0xaa),
        ICvtOp(BitSize::B32, intop::CvtOp::TruncUF64) => s!().byte(0xab),
        ICvtOp(BitSize::B64, intop::CvtOp::ExtendSI32) => s!().byte(0xac),
        ICvtOp(BitSize::B64, intop::CvtOp::ExtendUI32) => s!().byte(0xad),
        ICvtOp(BitSize::B64, intop::CvtOp::TruncSF32) => s!().byte(0xae),
        ICvtOp(BitSize::B64, intop::CvtOp::TruncUF32) => s!().byte(0xaf),
        ICvtOp(BitSize::B64, intop::CvtOp::TruncSF64) => s!().byte(0xb0),
        ICvtOp(BitSize::B64, intop::CvtOp::TruncUF64) => s!().byte(0xb1),
        FCvtOp(BitSize::B32, floatop::CvtOp::ConvertSI32) => s!().byte(0xb2),
        FCvtOp(BitSize::B32, floatop::CvtOp::ConvertUI32) => s!().byte(0xb3),
        FCvtOp(BitSize::B32, floatop::CvtOp::ConvertSI64) => s!().byte(0xb4),
        FCvtOp(BitSize::B32, floatop::CvtOp::ConvertUI64) => s!().byte(0xb5),
        FCvtOp(BitSize::B32, floatop::CvtOp::DemoteF64) => s!().byte(0xb6),
        FCvtOp(BitSize::B64, floatop::CvtOp::ConvertSI32) => s!().byte(0xb7),
        FCvtOp(BitSize::B64, floatop::CvtOp::ConvertUI32) => s!().byte(0xb8),
        FCvtOp(BitSize::B64, floatop::CvtOp::ConvertSI64) => s!().byte(0xb9),
        FCvtOp(BitSize::B64, floatop::CvtOp::ConvertUI64) => s!().byte(0xba),
        FCvtOp(BitSize::B64, floatop::CvtOp::PromoteF32) => s!().byte(0xbb),
        ICvtOp(BitSize::B32, intop::CvtOp::ReinterpretFloat) => s!().byte(0xbc),
        ICvtOp(BitSize::B64, intop::CvtOp::ReinterpretFloat) => s!().byte(0xbd),
        FCvtOp(BitSize::B32, floatop::CvtOp::ReinterpretInt) => s!().byte(0xbe),
        FCvtOp(BitSize::B64, floatop::CvtOp::ReinterpretInt) => s!().byte(0xbf),

        IUnOp(BitSize::B32, intop::UnOp::ExtendS(PackSize::Pack8)) => s!().byte(0xc0),
        IUnOp(BitSize::B32, intop::UnOp::ExtendS(PackSize::Pack16)) => s!().byte(0xc1),
        IUnOp(BitSize::B64, intop::UnOp::ExtendS(PackSize::Pack8)) => s!().byte(0xc2),
        IUnOp(BitSize::B64, intop::UnOp::ExtendS(PackSize::Pack16)) => s!().byte(0xc3),
        IUnOp(BitSize::B64, intop::UnOp::ExtendS(PackSize::Pack32)) => s!().byte(0xc4),

        ICvtOp(BitSize::B32, intop::CvtOp::TruncSatSF32) => s!().byte(0xfc).byte(0x00),
        ICvtOp(BitSize::B32, intop::CvtOp::TruncSatUF32) => s!().byte(0xfc).byte(0x01),
        ICvtOp(BitSize::B32, intop::CvtOp::TruncSatSF64) => s!().byte(0xfc).byte(0x02),
        ICvtOp(BitSize::B32, intop::CvtOp::TruncSatUF64) => s!().byte(0xfc).byte(0x03),
        ICvtOp(BitSize::B64, intop::CvtOp::TruncSatSF32) => s!().byte(0xfc).byte(0x04),
        ICvtOp(BitSize::B64, intop::CvtOp::TruncSatUF32) => s!().byte(0xfc).byte(0x05),
        ICvtOp(BitSize::B64, intop::CvtOp::TruncSatSF64) => s!().byte(0xfc).byte(0x06),
        ICvtOp(BitSize::B64, intop::CvtOp::TruncSatUF64) => s!().byte(0xfc).byte(0x07),

        // Explicitly panic on bad instructions
        FCvtOp(BitSize::B32, floatop::CvtOp::PromoteF32) |
        FCvtOp(BitSize::B64, floatop::CvtOp::DemoteF64) |
        ICvtOp(BitSize::B32, intop::CvtOp::ExtendSI32) |
        ICvtOp(BitSize::B32, intop::CvtOp::ExtendUI32) |
        ICvtOp(BitSize::B64, intop::CvtOp::WrapI64) |
        IUnOp(BitSize::B32, intop::UnOp::ExtendS(PackSize::Pack32)) =>
            panic!(),
    }
}}

generate! { memarg : MemArg = s!().u32(v!().align).u32(v!().offset) }

generate! { expr : Expr = s!().vec_no_lenprefix(v!().0, Stream::instr).byte(0x0b) }

generate! { typeidx: TypeIdx = s!().u32(v!().0) }
generate! { funcidx : FuncIdx = s!().u32(v!().0) }
generate! { tableidx : TableIdx = s!().u32(v!().0) }
generate! { memidx : MemIdx = s!().u32(v!().0) }
generate! { globalidx : GlobalIdx = s!().u32(v!().0) }
generate! { localidx : LocalIdx = s!().u32(v!().0) }
generate! { labelidx : LabelIdx = s!().u32(v!().0) }

macro_rules! section {
    ($n:literal, $name:ident : Option<$ty:ty> = $body:expr) => {
        section! {$n @ Option::is_none, $name : Option<$ty> = $body}
    };
    ($n:literal, $name:ident : Option<$ty:ty> via $elem:expr) => {
        section! {$n, $name : Option<$ty> = match v!() {
            None => s!(),
            Some(s) => $elem(s!(), s),
        }}
    };
    ($n:literal, $name:ident : Vec<$ty:ty> = $body:expr) => {
        section! {$n @ Vec::is_empty, $name : Vec<$ty> = $body}
    };
    ($n:literal, $name:ident : Vec<$ty:ty> via $elem:expr) => {
        section! {$n, $name : Vec<$ty> = s!().vec(v!(), $elem)}
    };
    ($n:literal @ $default:expr, $name:ident : $ty:ty = $body:expr) => {
        impl Stream {
            fn $name(&mut self, v: $ty) -> &mut Self {
                use std::convert::TryInto;

                if $default(&v) {
                    // We see the default value, skip this section
                    return self;
                }

                fn aux(aux_self: &mut Stream, aux_v: $ty) -> &mut Stream {
                    #[allow(unused_macros)]
                    macro_rules! s {
                        () => {
                            aux_self
                        };
                    }
                    #[allow(unused_macros)]
                    macro_rules! v {
                        () => {
                            aux_v
                        };
                    }
                    $body
                }

                let mut inner = Self::new();
                aux(&mut inner, v);

                self.byte($n)
                    .u32(inner.base.len().try_into().unwrap())
                    .bytes(inner.base)
            }
        }
    };
}

// Note: Custom section are unsupported for now for serialization;
// this is fine since they are optional anyways.

section! {1, typesec : Vec<FuncType> via Stream::functype}

section! {2, importsec : Vec<Import> via Stream::import}
generate! { import : Import = s!().name(v!().module).name(v!().name).importdesc(v!().desc) }
generate! { importdesc : ImportDesc = match v!() {
    ImportDesc::Func(x) => s!().byte(0).typeidx(x),
    ImportDesc::Table(x) => s!().byte(1).tabletype(x),
    ImportDesc::Mem(x) => s!().byte(2).memtype(x),
    ImportDesc::Global(x) => s!().byte(3).globaltype(x),
}}

section! {3, funcsec : Vec<TypeIdx> via Stream::typeidx}

section! {4, tablesec : Vec<Table> via Stream::table}
generate! {table: Table = s!().tabletype(v!().typ)}

section! {5, memsec : Vec<Mem> via Stream::mem}
generate! {mem : Mem = s!().memtype(v!().typ)}

section! {6, globalsec : Vec<Global> via Stream::global}
generate! {global: Global = s!().globaltype(v!().typ).expr(v!().init)}

section! {7, exportsec: Vec<Export> via Stream::export}
generate! {export: Export = s!().name(v!().name).exportdesc(v!().desc)}
generate! {exportdesc: ExportDesc = match v!() {
    ExportDesc::Func(x) => s!().byte(0).funcidx(x),
    ExportDesc::Table(x) => s!().byte(1).tableidx(x),
    ExportDesc::Mem(x) => s!().byte(2).memidx(x),
    ExportDesc::Global(x) => s!().byte(3).globalidx(x),
}}

section! {8, startsec: Option<Start> via Stream::start}
generate! {start: Start = s!().funcidx(v!().func)}

section! {9, elemsec: Vec<Elem> via Stream::elem}
generate! {elem: Elem = {
    s!().tableidx(v!().table).expr(v!().offset).vec(v!().init, Stream::funcidx)
}}

fn run_length_encode<T>(v: Vec<T>) -> Vec<(T, u32)>
where
    T: PartialEq,
{
    if v.is_empty() {
        return vec![];
    }

    let mut v = v.into_iter();

    let mut result = vec![];
    let mut prev = v.next().unwrap(); // safe because empty case is already handled
    let mut prev_len = 1;

    while let Some(cur) = v.next() {
        if cur == prev {
            prev_len += 1;
        } else {
            result.push((prev, prev_len));
            prev = cur;
            prev_len = 1;
        }
    }

    result.push((prev, prev_len));

    result
}

section! {10, codesec: Vec<(Vec<ValType>, Expr)> via Stream::code}
generate! {code: (Vec<ValType>, Expr) = {
    use std::convert::TryInto;
    let mut s = Stream::new();
    s.func(v!());
    s!().u32(s.base.len().try_into().unwrap()).bytes(s.base)
}}
generate! {func: (Vec<ValType>, Expr) = {
    s!().vec(run_length_encode(v!().0), Stream::rle_local).expr(v!().1)}
}
generate! {rle_local: (ValType, u32) = s!().u32(v!().1).valtype(v!().0)}

section! {11, datasec: Vec<Data> via Stream::data}
generate! {data: Data = s!().memidx(v!().data).expr(v!().offset).vec(v!().init, Stream::byte)}

generate! {module: Module = {
    // magic
    s!().byte(0x00).byte(0x61).byte(0x73).byte(0x6d);
    // version
    s!().byte(0x01).byte(0x00).byte(0x00).byte(0x00);

    let Module { types, funcs, tables, mems, globals,
                 elem, data, start, imports, exports,
                 names: _ } = v!();

    let (funcsec, codesec) = {
        let (mut fs, mut cs) = (vec![], vec![]);
        for f in funcs.into_iter() {
            match f.internals {
                FuncInternals::ImportedFunc {..} => {
                    // XXX: Here we assume we already have the
                    // function marked in the imports table. It is
                    // possible we do actually need to check that
                    // it exists in the imports.
                },
                FuncInternals::LocalFunc { locals, body } => {
                    fs.push(f.typ);
                    cs.push((locals, body));
                }
            }
        }
        (fs, cs)
    };

    s!().typesec(types);
    s!().importsec(imports);
    s!().funcsec(funcsec);
    s!().tablesec(tables);
    s!().memsec(mems);
    s!().globalsec(globals);
    s!().exportsec(exports);
    s!().startsec(start);
    s!().elemsec(elem);
    s!().codesec(codesec);
    s!().datasec(data);

    s!()
}}

pub fn serialize_module(module: Module) -> Vec<u8> {
    let mut s = Stream::new();
    s.module(module);
    s.base
}
