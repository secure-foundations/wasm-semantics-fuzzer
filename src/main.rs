#![allow(dead_code)] // TEMP

mod parser;
mod serializer;
mod wasm;

mod rng;

type Maybe<T> = color_eyre::eyre::Result<T>;

struct WasmWriter {
    written: Vec<String>,
}

const BYTES_PER_PAGE: usize = 65536;

impl WasmWriter {
    /// The maximum number of 64k pages that WasmWriter is allowed to
    /// use. Can be bumped up safely. Other parts of the code will
    /// shift things respectively to compensate.
    ///
    /// If the data from the writer cannot fit into this size, it will
    /// cause a runtime panic when trying to generate the Wasm module,
    /// and thus it is always safe. If such a panic occurs, this value
    /// should just be bumped up by 1.
    const USABLE_PAGES: usize = 1;

    fn new() -> Self {
        WasmWriter { written: vec![] }
    }

    fn write<S: Into<String>>(&mut self, s: S) -> wasm::syntax::Instr {
        use wasm::syntax::*;

        let s = s.into();

        let write_position = match self.written.iter().position(|x| x == &s) {
            Some(p) => p,
            None => {
                self.written.push(s);
                self.written.len() - 1
            }
        } as i32;

        Instr::Block(
            BlockType::ValType(None),
            vec![
                // fd
                Instr::Const(Const::I32(1)),
                // iovs_ptr
                Instr::Const(Const::I32(8 * (1 + write_position))),
                // iovs_len
                Instr::Const(Const::I32(1)),
                // nwritten_ptr
                Instr::Const(Const::I32(0)),
                // Call to fd_write
                Instr::Call(FuncIdx(1)),
                // Check if result is zero
                Instr::ITestOp(BitSize::B32, intop::TestOp::Eqz),
                // If yes, life is good, otherwise, we must immediately exit
                Instr::If(BlockType::ValType(None), vec![], wasm_exit_with(1)),
            ],
        )
    }

    fn generate_memory_data(self, rng: &mut rng::Rng) -> Vec<wasm::syntax::Data> {
        use wasm::syntax::*;

        let mut init = vec![];

        let space_to_actual_strings = (rng.next() % 50) as i32;

        let start_offset_due_to_nwritten = if rng.next() % 2 == 0 {
            // space for nwritten
            for _ in 0..8 {
                init.push(0);
            }
            0
        } else {
            8
        };

        // iovecs
        let mut next_posn = space_to_actual_strings + 8 * (1 + self.written.len() as i32);
        for s in self.written.iter() {
            let len = s.bytes().count() as i32;
            init.extend_from_slice(&next_posn.to_le_bytes());
            init.extend_from_slice(&len.to_le_bytes());
            next_posn += len;
        }

        /// Perform a checked creation of a Data structure. Should
        /// only use this within this function to create the Data
        /// structure.
        fn data_at(offset: i32, data: Vec<u8>) -> Data {
            if offset as usize + data.len() >= BYTES_PER_PAGE * WasmWriter::USABLE_PAGES {
                panic!(
                    "Unable to fit WasmWriter values in {} pages. Please bump it up.",
                    WasmWriter::USABLE_PAGES
                );
            }
            Data {
                data: MemIdx(0),
                offset: Expr(vec![Instr::Const(Const::I32(offset))]),
                init: data,
            }
        }

        if rng.next() % 2 == 0 {
            // Contiguous Initializer

            // space to actual strings
            for _ in 0..space_to_actual_strings {
                init.push(rng.next() as u8);
            }

            // the actual strings
            for s in self.written.iter() {
                init.extend(s.bytes());
            }

            vec![data_at(start_offset_due_to_nwritten, init)]
        } else {
            // Split Initializer

            let string_start =
                start_offset_due_to_nwritten + init.len() as i32 + space_to_actual_strings;

            let mut res = vec![data_at(start_offset_due_to_nwritten, init)];

            if rng.next() % 2 == 0 {
                // Strings in one blob
                let mut init = vec![];
                for s in self.written {
                    init.extend(s.bytes());
                }
                res.push(data_at(string_start, init));
                res
            } else {
                // Split strings
                let mut string_start = string_start;
                for s in self.written {
                    res.push(data_at(string_start, s.bytes().collect()));
                    string_start += s.bytes().count() as i32;
                }
                res
            }
        }
    }
}

fn valtype_of_const(c: wasm::syntax::Const) -> wasm::syntax::ValType {
    use wasm::syntax::*;
    match c {
        Const::I32(_) => ValType::I32,
        Const::I64(_) => ValType::I64,
        Const::F32(_) => ValType::F32,
        Const::F64(_) => ValType::F64,
    }
}

fn build_module_from(
    funcs: Vec<(
        Vec<wasm::syntax::Const>,
        Vec<wasm::syntax::ValType>,
        Vec<wasm::syntax::Instr>,
    )>,
    writer: WasmWriter,
    rng: &mut rng::Rng,
) -> wasm::syntax::Module {
    use wasm::syntax::*;
    let main_func = Expr(
        funcs
            .iter()
            .enumerate()
            .map(|(i, (args, _locals, _body))| {
                args.iter()
                    .map(|a| Instr::Const(*a))
                    .chain(std::iter::once(Instr::Call(FuncIdx(3 + i as u32))))
            })
            .flatten()
            .chain(std::iter::once(instr_wasm_exit_with(0)))
            .collect(),
    );

    fn functype_of_args(args: &[Const]) -> FuncType {
        FuncType {
            from: ResultType(args.iter().cloned().map(valtype_of_const).collect()),
            to: ResultType(vec![]),
        }
    }

    let types = {
        let mut t = vec![
            FuncType {
                from: ResultType(vec![ValType::I32]),
                to: ResultType(vec![]),
            },
            FuncType {
                from: ResultType(vec![ValType::I32; 4]),
                to: ResultType(vec![ValType::I32]),
            },
            FuncType {
                from: ResultType(vec![]),
                to: ResultType(vec![]),
            },
            // i32->i32 used for `check_loops`
            FuncType {
                from: ResultType(vec![ValType::I32]),
                to: ResultType(vec![ValType::I32]),
            },
        ];
        for (args, _locals, _body) in &funcs {
            let expected = functype_of_args(args);
            if !t.contains(&expected) {
                t.push(expected);
            }
        }
        t
    };

    let funcs = vec![
        Func {
            typ: TypeIdx(0),
            internals: FuncInternals::ImportedFunc {
                module: "wasi_snapshot_preview1".into(),
                name: "proc_exit".into(),
            },
        },
        Func {
            typ: TypeIdx(1),
            internals: FuncInternals::ImportedFunc {
                module: "wasi_snapshot_preview1".into(),
                name: "fd_write".into(),
            },
        },
        Func {
            typ: TypeIdx(2),
            internals: FuncInternals::LocalFunc {
                locals: vec![],
                body: main_func,
            },
        },
    ]
    .into_iter()
    .chain(funcs.into_iter().map(|(args, locals, body)| {
        Func {
            typ: TypeIdx(
                types
                    .iter()
                    .position(|x| x == &functype_of_args(&args))
                    .unwrap() as u32,
            ),
            internals: FuncInternals::LocalFunc {
                locals,
                body: Expr(body),
            },
        }
    }))
    .collect();

    Module {
        types,
        funcs,
        tables: vec![],
        mems: vec![Mem {
            typ: MemType(Limits {
                min: WasmWriter::USABLE_PAGES as u32 + 1,
                max: None,
            }),
        }],
        globals: vec![
            // NOTE: See `check_global_ops` for required values
            // here. Must stay in sync with that function.
            Global {
                typ: GlobalType(Mut::Var, ValType::I32),
                init: Expr(vec![Instr::Const(Const::I32(0x32323232))]),
            },
            Global {
                typ: GlobalType(Mut::Var, ValType::I64),
                init: Expr(vec![Instr::Const(Const::I64(0x64646464))]),
            },
        ],
        elem: vec![],
        data: writer.generate_memory_data(rng),
        start: None,
        imports: vec![
            Import {
                module: "wasi_snapshot_preview1".into(),
                name: "proc_exit".into(),
                desc: ImportDesc::Func(TypeIdx(0)),
            },
            Import {
                module: "wasi_snapshot_preview1".into(),
                name: "fd_write".into(),
                desc: ImportDesc::Func(TypeIdx(1)),
            },
        ],
        exports: vec![
            Export {
                name: "memory".into(),
                desc: ExportDesc::Mem(MemIdx(0)),
            },
            Export {
                name: "_start".into(),
                desc: ExportDesc::Func(FuncIdx(2)),
            },
        ],
        names: Names {
            module: None,
            functions: Default::default(),
            locals: Default::default(),
        },
    }
}

fn wasm_exit_with(code: i32) -> Vec<wasm::syntax::Instr> {
    use wasm::syntax::*;
    vec![
        Instr::Const(Const::I32(code)),
        Instr::Call(FuncIdx(0)),
        Instr::Unreachable,
    ]
}

fn instr_wasm_exit_with(code: i32) -> wasm::syntax::Instr {
    use wasm::syntax::*;
    Instr::Block(BlockType::ValType(None), wasm_exit_with(code))
}

const INTERESTING_VALUES_I32: [i32; 17] = [
    0,
    -1,
    1,
    2,
    3,
    8,
    16,
    32,
    64,
    7,
    15,
    31,
    63,
    i32::MIN,
    i32::MIN + 1,
    i32::MAX,
    i32::MAX - 1,
];

const INTERESTING_VALUES_I64: [i64; 23] = [
    0,
    -1,
    1,
    2,
    3,
    8,
    16,
    32,
    64,
    7,
    15,
    31,
    63,
    i32::MIN as i64,
    i32::MIN as i64 - 1,
    i32::MIN as i64 + 1,
    i32::MAX as i64,
    i32::MAX as i64 - 1,
    i32::MAX as i64 + 1,
    i64::MIN,
    i64::MIN + 1,
    i64::MAX,
    i64::MAX - 1,
];

const INTERESTING_VALUES_F32: [f32; 4] = [0.0, 1.0, f32::INFINITY, f32::EPSILON];

const INTERESTING_VALUES_F64: [f64; 4] = [0.0, 1.0, f64::INFINITY, f64::EPSILON];

fn gen_test(
    writer: &mut WasmWriter,
    test_as_str: &str,
    test_instructions: Vec<wasm::syntax::Instr>,
) -> wasm::syntax::Instr {
    // Note: test_instrs should be of type []->[i32] where the value
    // they leave on stack should be 1 if success, 0 if fail.
    use wasm::syntax::*;
    Instr::Block(
        BlockType::ValType(None),
        vec![
            writer.write(format!("[ ] Testing {}\n", test_as_str)),
            Instr::Block(BlockType::ValType(Some(ValType::I32)), test_instructions),
            Instr::If(
                BlockType::ValType(None),
                vec![writer.write("[+] Success\n")],
                vec![
                    writer.write("[!] Failure. Exiting.\n"),
                    instr_wasm_exit_with(2),
                ],
            ),
        ],
    )
}

fn gen_rand_i32(rng: &mut rng::Rng, related: Option<i32>) -> i32 {
    match related {
        None => {
            if rng.next() % 100 < 50 {
                *rng.choice(&INTERESTING_VALUES_I32)
            } else {
                rng.next() as i32
            }
        }
        Some(a) => {
            if rng.next() % 100 < 45 {
                *rng.choice(&INTERESTING_VALUES_I32)
            } else if rng.next() % 100 < 10 {
                *rng.choice(&[a.wrapping_add(1), a.wrapping_sub(1), a])
            } else {
                rng.next() as i32
            }
        }
    }
}
fn gen_rand_i64(rng: &mut rng::Rng, related: Option<i64>) -> i64 {
    match related {
        None => {
            if rng.next() % 100 < 50 {
                *rng.choice(&INTERESTING_VALUES_I64)
            } else {
                rng.next() as i64
            }
        }
        Some(a) => {
            if rng.next() % 100 < 45 {
                *rng.choice(&INTERESTING_VALUES_I64)
            } else if rng.next() % 100 < 10 {
                *rng.choice(&[a.wrapping_add(1), a.wrapping_sub(1), a])
            } else {
                rng.next() as i64
            }
        }
    }
}
fn gen_rand_f32(rng: &mut rng::Rng, related: Option<f32>) -> f32 {
    rng.choice(&[-1., 1.])
        * match related {
            None => {
                if rng.next() % 100 < 50 {
                    *rng.choice(&INTERESTING_VALUES_F32)
                } else {
                    rng.next() as f32 / rng.next() as f32
                }
            }
            Some(a) => {
                if rng.next() % 100 < 45 {
                    *rng.choice(&INTERESTING_VALUES_F32)
                } else if rng.next() % 100 < 10 {
                    *rng.choice(&[a + 1., a - 1., a, a + f32::EPSILON, a - f32::EPSILON])
                } else {
                    rng.next() as f32 / rng.next() as f32
                }
            }
        }
}
fn gen_rand_f64(rng: &mut rng::Rng, related: Option<f64>) -> f64 {
    rng.choice(&[-1., 1.])
        * match related {
            None => {
                if rng.next() % 100 < 50 {
                    *rng.choice(&INTERESTING_VALUES_F64)
                } else {
                    rng.next() as f64 / rng.next() as f64
                }
            }
            Some(a) => {
                if rng.next() % 100 < 45 {
                    *rng.choice(&INTERESTING_VALUES_F64)
                } else if rng.next() % 100 < 10 {
                    *rng.choice(&[a + 1., a - 1., a, a + f64::EPSILON, a - f64::EPSILON])
                } else {
                    rng.next() as f64 / rng.next() as f64
                }
            }
        }
}

macro_rules! gen_ibinop {
    ($name:ident, $ty:ty, $uty:ty, $sz:literal, $genrand:ident, $bs:expr, $const:expr) => {
        fn $name(rng: &mut rng::Rng, writer: &mut WasmWriter) -> Option<wasm::syntax::Instr> {
            use intop::BinOp::*;
            use wasm::syntax::*;

            let a: $ty = $genrand(rng, None);
            let b: $ty = $genrand(rng, Some(a));
            let op = *rng.choice(&[
                Add, Sub, Mul, DivS, DivU, RemS, RemU, And, Or, Xor, Shl, ShrS, ShrU, Rotl, Rotr,
            ]);
            let res = match op {
                Add => a.wrapping_add(b),
                Sub => a.wrapping_sub(b),
                Mul => a.wrapping_mul(b),
                DivS => (b != 0).then(|| a.wrapping_div(b))?,
                DivU => (b != 0).then(|| (a as $uty).wrapping_div(b as $uty) as $ty)?,
                RemS => (b != 0).then(|| a.wrapping_rem(b))?,
                RemU => (b != 0).then(|| (a as $uty).wrapping_rem(b as $uty) as $ty)?,
                And => a & b,
                Or => a | b,
                Xor => a ^ b,
                Shl => a.wrapping_shl(b as u32 % $sz),
                ShrS => a.wrapping_shr(b as u32 % $sz),
                ShrU => ((a as $uty).wrapping_shr(b as u32 % $sz)) as $ty,
                Rotl => a.rotate_left(b as u32),
                Rotr => a.rotate_right(b as u32),
            };

            Some(gen_test(
                writer,
                &format!("({} bit ibinop) {} {:?} {} =? {}", $sz, a, op, b, res),
                vec![
                    Instr::Const($const(a)),
                    Instr::Const($const(b)),
                    Instr::IBinOp($bs, op),
                    Instr::Const($const(res)),
                    Instr::IRelOp($bs, intop::RelOp::Eq),
                ],
            ))
        }
    };
}

gen_ibinop! {
    gen_ibinop_64, i64, u64, 64, gen_rand_i64,
    wasm::syntax::BitSize::B64, wasm::syntax::Const::I64
}
gen_ibinop! {
    gen_ibinop_32, i32, u32, 32, gen_rand_i32,
    wasm::syntax::BitSize::B32, wasm::syntax::Const::I32
}

fn gen_iunop(rng: &mut rng::Rng, writer: &mut WasmWriter) -> Option<wasm::syntax::Instr> {
    use intop::UnOp::*;
    use wasm::syntax::*;

    let bitsize = *rng.choice(&[BitSize::B32, BitSize::B64]);

    let a: Const = match bitsize {
        BitSize::B32 => Const::I32(gen_rand_i32(rng, None)),
        BitSize::B64 => Const::I64(gen_rand_i64(rng, None)),
    };
    // Note: We explicitly _disable_ ExtendS since it didn't exist on
    // a (slightly) older version of Wasm, which is what vWasm and
    // wat2vasm is built on. Compilers don't seem to emit this
    // instruction (yet) so we can punt the implementation of this in
    // vWasm and wat2vasm for later.
    let op = *rng.choice(&[
        Clz, Ctz,
        Popcnt,
        // ExtendS(PackSize::Pack8),
        // ExtendS(PackSize::Pack16),
        // ExtendS(PackSize::Pack32),
    ]);
    let res = match a {
        Const::I32(a) => Const::I32(match op {
            Clz => a.leading_zeros() as i32,
            Ctz => a.trailing_zeros() as i32,
            Popcnt => a.count_ones() as i32,
            ExtendS(PackSize::Pack8) => (a as i8) as i32,
            ExtendS(PackSize::Pack16) => (a as i16) as i32,
            ExtendS(PackSize::Pack32) => return None,
        }),
        Const::I64(a) => Const::I64(match op {
            Clz => a.leading_zeros() as i64,
            Ctz => a.trailing_zeros() as i64,
            Popcnt => a.count_ones() as i64,
            ExtendS(PackSize::Pack8) => (a as i8) as i64,
            ExtendS(PackSize::Pack16) => (a as i16) as i64,
            ExtendS(PackSize::Pack32) => (a as i32) as i64,
        }),
        Const::F32(_) | Const::F64(_) => unreachable!(),
    };

    Some(gen_test(
        writer,
        &format!("({} bit iunop) {:?} {} =? {}", bitsize, op, a, res),
        vec![
            Instr::Const(a),
            Instr::IUnOp(bitsize, op),
            Instr::Const(res),
            Instr::IRelOp(bitsize, intop::RelOp::Eq),
        ],
    ))
}

fn gen_itestop_or_irelop(
    rng: &mut rng::Rng,
    writer: &mut WasmWriter,
) -> Option<wasm::syntax::Instr> {
    use intop::RelOp::*;
    use intop::TestOp::*;
    use wasm::syntax::*;

    let bitsize = *rng.choice(&[BitSize::B32, BitSize::B64]);

    let a: Const = match bitsize {
        BitSize::B32 => Const::I32(gen_rand_i32(rng, None)),
        BitSize::B64 => Const::I64(gen_rand_i64(rng, None)),
    };
    let b: Const = match a {
        Const::I32(a) => Const::I32(gen_rand_i32(rng, Some(a))),
        Const::I64(a) => Const::I64(gen_rand_i64(rng, Some(a))),
        _ => unreachable!(),
    };
    let op = rng
        .choice(&[
            Instr::ITestOp(bitsize, Eqz),
            Instr::IRelOp(bitsize, Eq),
            Instr::IRelOp(bitsize, Ne),
            Instr::IRelOp(bitsize, LtS),
            Instr::IRelOp(bitsize, LtU),
            Instr::IRelOp(bitsize, GtS),
            Instr::IRelOp(bitsize, GtU),
            Instr::IRelOp(bitsize, LeS),
            Instr::IRelOp(bitsize, LeU),
            Instr::IRelOp(bitsize, GeS),
            Instr::IRelOp(bitsize, GeU),
        ])
        .clone();
    let res = Const::I32(match (a, b) {
        (Const::I32(a), Const::I32(b)) => match op {
            Instr::ITestOp(_, Eqz) => (a == 0),
            Instr::IRelOp(_, Eq) => (a == b),
            Instr::IRelOp(_, Ne) => (a != b),
            Instr::IRelOp(_, LtS) => (a < b),
            Instr::IRelOp(_, LtU) => ((a as u32) < (b as u32)),
            Instr::IRelOp(_, GtS) => (a > b),
            Instr::IRelOp(_, GtU) => ((a as u32) > (b as u32)),
            Instr::IRelOp(_, LeS) => (a <= b),
            Instr::IRelOp(_, LeU) => ((a as u32) <= (b as u32)),
            Instr::IRelOp(_, GeS) => (a >= b),
            Instr::IRelOp(_, GeU) => ((a as u32) >= (b as u32)),
            _ => unreachable!(),
        },
        (Const::I64(a), Const::I64(b)) => match op {
            Instr::ITestOp(_, Eqz) => (a == 0),
            Instr::IRelOp(_, Eq) => (a == b),
            Instr::IRelOp(_, Ne) => (a != b),
            Instr::IRelOp(_, LtS) => (a < b),
            Instr::IRelOp(_, LtU) => ((a as u64) < (b as u64)),
            Instr::IRelOp(_, GtS) => (a > b),
            Instr::IRelOp(_, GtU) => ((a as u64) > (b as u64)),
            Instr::IRelOp(_, LeS) => (a <= b),
            Instr::IRelOp(_, LeU) => ((a as u64) <= (b as u64)),
            Instr::IRelOp(_, GeS) => (a >= b),
            Instr::IRelOp(_, GeU) => ((a as u64) >= (b as u64)),
            _ => unreachable!(),
        },
        _ => unreachable!(),
    } as i32);

    match op {
        Instr::ITestOp(..) => Some(gen_test(
            writer,
            &format!("({} bit itestop) {:?} {} =? {}", bitsize, op, a, res),
            vec![
                Instr::Const(a),
                op,
                Instr::Const(res),
                Instr::IRelOp(BitSize::B32, intop::RelOp::Eq),
            ],
        )),
        Instr::IRelOp(..) => Some(gen_test(
            writer,
            &format!("({} bit irelop) {} {:?} {} =? {}", bitsize, a, op, b, res),
            vec![
                Instr::Const(a),
                Instr::Const(b),
                op,
                Instr::Const(res),
                Instr::IRelOp(BitSize::B32, intop::RelOp::Eq),
            ],
        )),
        _ => unreachable!(),
    }
}

trait FloatConv {
    fn try_to_i32(self) -> Option<i32>;
    fn try_to_i64(self) -> Option<i64>;
    fn try_to_u32(self) -> Option<u32>;
    fn try_to_u64(self) -> Option<u64>;
}
macro_rules! float_conv {
    ($from:ty) => {
        impl FloatConv for $from {
            float_conv! {@@internal-i try_to_i32 i32}
            float_conv! {@@internal-i try_to_i64 i64}
            float_conv! {@@internal-u try_to_u32 u32}
            float_conv! {@@internal-u try_to_u64 u64}
        }
    };
    (@@internal-i $name:ident $to:ty) => {
        fn $name(self) -> Option<$to> {
            if self.is_finite() {
                Some(self as $to)
            } else {
                None
            }
        }
    };
    (@@internal-u $name:ident $to:ty) => {
        fn $name(self) -> Option<$to> {
            if self.is_finite() && self >= 0. {
                Some(self as $to)
            } else {
                None
            }
        }
    };
}
float_conv! {f32}
float_conv! {f64}

fn gen_icvtop(rng: &mut rng::Rng, writer: &mut WasmWriter) -> Option<wasm::syntax::Instr> {
    use intop::CvtOp::*;
    use wasm::syntax::*;
    use BitSize::*;

    // Note: We explicitly _disable_ TruncSat since it didn't exist on
    // a (slightly) older version of Wasm, which is what vWasm and
    // wat2vasm is built on. Compilers don't seem to emit this
    // instruction (yet) so we can punt the implementation of this in
    // vWasm and wat2vasm for later.
    let op = *rng.choice(&[
        ExtendSI32,
        ExtendUI32,
        WrapI64,
        TruncSF32,
        TruncUF32,
        TruncSF64,
        TruncUF64,
        // TruncSatSF32,
        // TruncSatUF32,
        // TruncSatSF64,
        // TruncSatUF64,
        ReinterpretFloat,
    ]);
    let bitsize = *rng.choice(&[B32, B64]);

    let r_i32: i32 = gen_rand_i32(rng, None);
    let r_i64: i64 = gen_rand_i64(rng, None);
    let r_f32: f32 = gen_rand_f32(rng, None);
    let r_f64: f64 = gen_rand_f64(rng, None);

    let c_i32: Const = Const::I32(r_i32);
    let c_i64: Const = Const::I64(r_i64);
    let c_f32: Const = Const::F32(r_f32);
    let c_f64: Const = Const::F64(r_f64);

    let input = match (op, bitsize) {
        (WrapI64, B32) => c_i64,
        (WrapI64, B64) => return None,
        (TruncSF32, _) => c_f32,
        (TruncUF32, _) => c_f32,
        (TruncSF64, _) => c_f64,
        (TruncUF64, _) => c_f64,
        (ExtendSI32, B32) => return None,
        (ExtendSI32, B64) => c_i32,
        (ExtendUI32, B32) => return None,
        (ExtendUI32, B64) => c_i32,
        (TruncSatSF32, _) => c_f32,
        (TruncSatUF32, _) => c_f32,
        (TruncSatSF64, _) => c_f64,
        (TruncSatUF64, _) => c_f64,
        (ReinterpretFloat, B32) => c_f32,
        (ReinterpretFloat, B64) => c_f64,
    };

    let output = match bitsize {
        B32 => Const::I32(match op {
            WrapI64 => r_i64 as i32,
            TruncSF32 => r_f32.trunc().try_to_i32()?,
            TruncUF32 => r_f32.trunc().try_to_u32()? as i32,
            TruncSF64 => r_f64.trunc().try_to_i32()?,
            TruncUF64 => r_f64.trunc().try_to_u32()? as i32,
            ExtendSI32 => unreachable!(),
            ExtendUI32 => unreachable!(),
            TruncSatSF32 => r_f32.trunc() as i32,
            TruncSatUF32 => r_f32.trunc() as u32 as i32,
            TruncSatSF64 => r_f64.trunc() as i32,
            TruncSatUF64 => r_f64.trunc() as u32 as i32,
            ReinterpretFloat => r_f32.to_bits() as i32,
        }),
        B64 => Const::I64(match op {
            WrapI64 => unreachable!(),
            TruncSF32 => r_f32.trunc().try_to_i64()?,
            TruncUF32 => r_f32.trunc().try_to_u64()? as i64,
            TruncSF64 => r_f64.trunc().try_to_i64()?,
            TruncUF64 => r_f64.trunc().try_to_u64()? as i64,
            ExtendSI32 => r_i32 as i64,
            ExtendUI32 => r_i32 as u32 as u64 as i64,
            TruncSatSF32 => r_f32.trunc() as i64,
            TruncSatUF32 => r_f32.trunc() as u64 as i64,
            TruncSatSF64 => r_f64.trunc() as i64,
            TruncSatUF64 => r_f64.trunc() as u64 as i64,
            ReinterpretFloat => r_f64.to_bits() as i64,
        }),
    };

    Some(gen_test(
        writer,
        &format!("({} bit icvtop) {:?} {} =? {}", bitsize, op, input, output),
        vec![
            Instr::Const(input),
            Instr::ICvtOp(bitsize, op),
            Instr::Const(output),
            Instr::IRelOp(bitsize, intop::RelOp::Eq),
        ],
    ))
}

fn gen_iops(num: usize, rng: &mut rng::Rng, writer: &mut WasmWriter) -> Vec<wasm::syntax::Instr> {
    (0..)
        .filter_map(|_i| {
            rng.choice::<fn(&mut _, &mut _) -> _>(&[
                gen_ibinop_32,
                gen_ibinop_64,
                gen_iunop,
                gen_itestop_or_irelop,
                gen_icvtop,
            ])(rng, writer)
        })
        .take(num)
        .collect()
}

fn check_memory_size(rng: &mut rng::Rng, writer: &mut WasmWriter) -> Vec<wasm::syntax::Instr> {
    use wasm::syntax::*;
    let mem_resize_by = ((rng.next() % 5) + 1) as i32;
    vec![
        gen_test(
            writer,
            "memory size: signed greater than 0",
            vec![
                Instr::MemSize,
                Instr::Const(Const::I32(0)),
                Instr::IRelOp(BitSize::B32, intop::RelOp::GtU),
            ],
        ),
        gen_test(
            writer,
            "memory size: unsigned greater than 0",
            vec![
                Instr::MemSize,
                Instr::Const(Const::I32(0)),
                Instr::IRelOp(BitSize::B32, intop::RelOp::GtU),
            ],
        ),
        gen_test(
            writer,
            "memory resize by 0 pages",
            vec![
                Instr::Const(Const::I32(0)),
                Instr::MemGrow,
                Instr::MemSize,
                Instr::IRelOp(BitSize::B32, intop::RelOp::Eq),
            ],
        ),
        gen_test(
            writer,
            &format!("memory resize by {} pages", mem_resize_by),
            vec![
                Instr::Const(Const::I32(mem_resize_by)),
                Instr::MemGrow,
                Instr::MemSize,
                Instr::Const(Const::I32(mem_resize_by)),
                Instr::IBinOp(BitSize::B32, intop::BinOp::Sub),
                // the result must be the "old" size, which is
                // `mem_resize_by` less than the new size
                Instr::IRelOp(BitSize::B32, intop::RelOp::Eq),
            ],
        ),
    ]
}

fn check_memory_ops(rng: &mut rng::Rng, writer: &mut WasmWriter) -> Vec<wasm::syntax::Instr> {
    use wasm::syntax::*;
    let mem_base_32 = (WasmWriter::USABLE_PAGES * BYTES_PER_PAGE) as u32;
    let mem_base_64 = (WasmWriter::USABLE_PAGES * BYTES_PER_PAGE) as u32 + 8;

    let mem_val_i32 = rng.next() as i32;
    let mem_val_i64 = rng.next() as i64;

    let align_32 = (rng.next() % 3) as u32; // 0, 1, 2 are fine
    let align_64 = (rng.next() % 4) as u32; // all are fine

    vec![
        gen_test(
            writer,
            &format!("initial memory load - 32 bit [align={}]", align_32),
            vec![
                Instr::Const(Const::I32(0)),
                // The const-0 before the mem-load is an additional
                // offset on top of the memarg offset
                Instr::Const(Const::I32(0)),
                Instr::MemLoad(MemLoad {
                    typ: ValType::I32,
                    extend: None,
                    memarg: MemArg {
                        offset: mem_base_32,
                        align: align_32,
                    },
                }),
                Instr::IRelOp(BitSize::B32, intop::RelOp::Eq),
            ],
        ),
        gen_test(
            writer,
            &format!(
                "initial memory load - 32 bit w/ flipped offset [align={}]",
                align_32
            ),
            vec![
                Instr::Const(Const::I32(0)),
                Instr::Const(Const::I32(mem_base_32 as i32)),
                Instr::MemLoad(MemLoad {
                    typ: ValType::I32,
                    extend: None,
                    memarg: MemArg {
                        offset: 0,
                        align: align_32,
                    },
                }),
                Instr::IRelOp(BitSize::B32, intop::RelOp::Eq),
            ],
        ),
        gen_test(
            writer,
            &format!("initial memory load - 64 bit [align={}]", align_64),
            vec![
                Instr::Const(Const::I64(0)),
                Instr::Const(Const::I32(0)),
                Instr::MemLoad(MemLoad {
                    typ: ValType::I64,
                    extend: None,
                    memarg: MemArg {
                        offset: mem_base_64,
                        align: align_64,
                    },
                }),
                Instr::IRelOp(BitSize::B64, intop::RelOp::Eq),
            ],
        ),
        gen_test(
            writer,
            &format!(
                "memory store - 32 bit [v={}, align={}]",
                mem_val_i32, align_32
            ),
            vec![
                Instr::Const(Const::I32(0)),
                Instr::Const(Const::I32(mem_val_i32)),
                Instr::MemStore(MemStore {
                    typ: ValType::I32,
                    memarg: MemArg {
                        offset: mem_base_32,
                        align: align_32,
                    },
                    bitwidth: None,
                }),
                Instr::Const(Const::I32(mem_val_i32)),
                Instr::Const(Const::I32(0)),
                Instr::MemLoad(MemLoad {
                    typ: ValType::I32,
                    extend: None,
                    memarg: MemArg {
                        offset: mem_base_32,
                        align: (rng.next() % 3) as u32, // keep it potentially different :)
                    },
                }),
                Instr::IRelOp(BitSize::B32, intop::RelOp::Eq),
            ],
        ),
        gen_test(
            writer,
            &format!("unchanged memory load - 64 bit [align={}]", align_64),
            vec![
                Instr::Const(Const::I64(0)),
                Instr::Const(Const::I32(0)),
                Instr::MemLoad(MemLoad {
                    typ: ValType::I64,
                    extend: None,
                    memarg: MemArg {
                        offset: mem_base_64,
                        align: align_64,
                    },
                }),
                Instr::IRelOp(BitSize::B64, intop::RelOp::Eq),
            ],
        ),
        gen_test(
            writer,
            &format!(
                "memory store - 64 bit [v={}, align={}]",
                mem_val_i64, align_64
            ),
            vec![
                Instr::Const(Const::I32(0)),
                Instr::Const(Const::I64(mem_val_i64)),
                Instr::MemStore(MemStore {
                    typ: ValType::I64,
                    memarg: MemArg {
                        offset: mem_base_64,
                        align: align_64,
                    },
                    bitwidth: None,
                }),
                Instr::Const(Const::I64(mem_val_i64)),
                Instr::Const(Const::I32(0)),
                Instr::MemLoad(MemLoad {
                    typ: ValType::I64,
                    extend: None,
                    memarg: MemArg {
                        offset: mem_base_64,
                        align: (rng.next() % 4) as u32, // keep it potentially different :)
                    },
                }),
                Instr::IRelOp(BitSize::B64, intop::RelOp::Eq),
            ],
        ),
        gen_test(
            writer,
            &format!("unchanged memory load - 32 bit"),
            vec![
                Instr::Const(Const::I32(mem_val_i32)),
                Instr::Const(Const::I32(0)),
                Instr::MemLoad(MemLoad {
                    typ: ValType::I32,
                    extend: None,
                    memarg: MemArg {
                        offset: mem_base_32,
                        align: 1,
                    },
                }),
                Instr::IRelOp(BitSize::B32, intop::RelOp::Eq),
            ],
        ),
        gen_test(writer, &format!("memory endianness check - 32 bit"), {
            let load_8_bits = Instr::MemLoad(MemLoad {
                typ: ValType::I32,
                extend: Some((8, SX::U)),
                memarg: MemArg {
                    offset: mem_base_32,
                    align: 0, // force alignment at 0, because byte
                },
            });
            vec![
                Instr::Const(Const::I32(mem_val_i32.to_le_bytes()[0] as i32)),
                Instr::Const(Const::I32(0)),
                load_8_bits.clone(),
                Instr::IRelOp(BitSize::B32, intop::RelOp::Eq),
                //
                Instr::Const(Const::I32(mem_val_i32.to_le_bytes()[1] as i32)),
                Instr::Const(Const::I32(1)),
                load_8_bits.clone(),
                Instr::IRelOp(BitSize::B32, intop::RelOp::Eq),
                Instr::IBinOp(BitSize::B32, intop::BinOp::And),
                //
                Instr::Const(Const::I32(mem_val_i32.to_le_bytes()[2] as i32)),
                Instr::Const(Const::I32(2)),
                load_8_bits.clone(),
                Instr::IRelOp(BitSize::B32, intop::RelOp::Eq),
                Instr::IBinOp(BitSize::B32, intop::BinOp::And),
                //
                Instr::Const(Const::I32(mem_val_i32.to_le_bytes()[3] as i32)),
                Instr::Const(Const::I32(3)),
                load_8_bits.clone(),
                Instr::IRelOp(BitSize::B32, intop::RelOp::Eq),
                Instr::IBinOp(BitSize::B32, intop::BinOp::And),
            ]
        }),
        gen_test(
            writer,
            &format!("memory endianness check - 64-bit split as 32-bit"),
            {
                let load_32_bits = Instr::MemLoad(MemLoad {
                    typ: ValType::I32,
                    extend: None,
                    memarg: MemArg {
                        offset: mem_base_64,
                        align: 2, // align at 2 because 32-bit == 4 byes == 2^2 bytes
                    },
                });
                vec![
                    Instr::Const(Const::I32((mem_val_i64 >> (0 * 32)) as i32)),
                    Instr::Const(Const::I32(0)),
                    load_32_bits.clone(),
                    Instr::IRelOp(BitSize::B32, intop::RelOp::Eq),
                    //
                    Instr::Const(Const::I32((mem_val_i64 >> (1 * 32)) as i32)),
                    Instr::Const(Const::I32(4)),
                    load_32_bits.clone(),
                    Instr::IRelOp(BitSize::B32, intop::RelOp::Eq),
                    Instr::IBinOp(BitSize::B32, intop::BinOp::And),
                ]
            },
        ),
        gen_test(writer, &format!("memory signed-extend-load"), {
            let load_8_bits = Instr::MemLoad(MemLoad {
                typ: ValType::I32,
                extend: Some((8, SX::S)),
                memarg: MemArg {
                    offset: mem_base_32,
                    align: 0, // force alignment at 0, because byte
                },
            });
            let load_16_bits = Instr::MemLoad(MemLoad {
                typ: ValType::I32,
                extend: Some((16, SX::S)),
                memarg: MemArg {
                    offset: mem_base_32,
                    align: 1, // align at either 0 or 1
                },
            });
            vec![
                Instr::Const(Const::I32(mem_val_i32.to_le_bytes()[0] as i8 as i32)),
                Instr::Const(Const::I32(0)),
                load_8_bits.clone(),
                Instr::IRelOp(BitSize::B32, intop::RelOp::Eq),
                //
                Instr::Const(Const::I32(mem_val_i32.to_le_bytes()[1] as i8 as i32)),
                Instr::Const(Const::I32(1)),
                load_8_bits.clone(),
                Instr::IRelOp(BitSize::B32, intop::RelOp::Eq),
                Instr::IBinOp(BitSize::B32, intop::BinOp::And),
                //
                Instr::Const(Const::I32(mem_val_i32 as u16 as i16 as i32)),
                Instr::Const(Const::I32(0)),
                load_16_bits.clone(),
                Instr::IRelOp(BitSize::B32, intop::RelOp::Eq),
                Instr::IBinOp(BitSize::B32, intop::BinOp::And),
                //
                Instr::Const(Const::I32((mem_val_i32 >> 16) as u16 as i16 as i32)),
                Instr::Const(Const::I32(2)),
                load_16_bits.clone(),
                Instr::IRelOp(BitSize::B32, intop::RelOp::Eq),
                Instr::IBinOp(BitSize::B32, intop::BinOp::And),
            ]
        }),
    ]
}

fn check_global_ops(rng: &mut rng::Rng, writer: &mut WasmWriter) -> Vec<wasm::syntax::Instr> {
    use wasm::syntax::*;
    let global_val_i32 = rng.next() as i32;
    let global_val_i64 = rng.next() as i64;
    vec![
        gen_test(
            writer,
            &format!("global-get 32-bit original value"),
            vec![
                Instr::GlobalGet(GlobalIdx(0)),
                Instr::Const(Const::I32(0x32323232)),
                Instr::IRelOp(BitSize::B32, intop::RelOp::Eq),
            ],
        ),
        gen_test(
            writer,
            &format!("global-get 64-bit original value"),
            vec![
                Instr::GlobalGet(GlobalIdx(1)),
                Instr::Const(Const::I64(0x64646464)),
                Instr::IRelOp(BitSize::B64, intop::RelOp::Eq),
            ],
        ),
        gen_test(
            writer,
            &format!("global-set (32-bit) to {}", global_val_i32),
            vec![
                Instr::Const(Const::I32(global_val_i32)),
                Instr::GlobalSet(GlobalIdx(0)),
                Instr::Const(Const::I32(global_val_i32)),
                Instr::GlobalGet(GlobalIdx(0)),
                Instr::IRelOp(BitSize::B32, intop::RelOp::Eq),
            ],
        ),
        gen_test(
            writer,
            &format!("global-get (unchanged) 64-bit value"),
            vec![
                Instr::GlobalGet(GlobalIdx(1)),
                Instr::Const(Const::I64(0x64646464)),
                Instr::IRelOp(BitSize::B64, intop::RelOp::Eq),
            ],
        ),
        gen_test(
            writer,
            &format!("global-set (64-bit) to {}", global_val_i64),
            vec![
                Instr::Const(Const::I64(global_val_i64)),
                Instr::GlobalSet(GlobalIdx(1)),
                Instr::Const(Const::I64(global_val_i64)),
                Instr::GlobalGet(GlobalIdx(1)),
                Instr::IRelOp(BitSize::B64, intop::RelOp::Eq),
            ],
        ),
        gen_test(
            writer,
            &format!("global-get (unchanged) 32-bit value"),
            vec![
                Instr::GlobalGet(GlobalIdx(0)),
                Instr::Const(Const::I32(global_val_i32)),
                Instr::IRelOp(BitSize::B32, intop::RelOp::Eq),
            ],
        ),
    ]
}

fn check_local_ops(
    rng: &mut rng::Rng,
    writer: &mut WasmWriter,
) -> (
    Vec<wasm::syntax::Const>,
    Vec<wasm::syntax::ValType>,
    Vec<wasm::syntax::Instr>,
) {
    use wasm::syntax::*;

    fn get_loc(typ: ValType) -> LocalIdx {
        LocalIdx(match typ {
            ValType::I32 => 0,
            ValType::I64 => 1,
            ValType::F32 => 2,
            ValType::F64 => 3,
        })
    }

    fn gen_initial(writer: &mut WasmWriter, typ: ValType) -> Instr {
        gen_test(
            writer,
            &format!("initial local get {}", typ),
            vec![
                Instr::LocalGet(get_loc(typ)),
                Instr::Const(const_zero(typ)),
                cmpeq(typ),
            ],
        )
    }

    fn gen_local_write(rng: &mut rng::Rng, writer: &mut WasmWriter, typ: ValType) -> Instr {
        let c = rand_const(rng, typ);
        let loc = get_loc(typ);
        gen_test(
            writer,
            &format!("local write {} [v={}]", typ, c),
            vec![
                Instr::Const(c),
                Instr::LocalSet(loc),
                //
                Instr::Const(c),
                Instr::LocalGet(loc),
                cmpeq(typ),
            ],
        )
    }

    fn gen_local_tee(rng: &mut rng::Rng, writer: &mut WasmWriter, typ: ValType) -> Instr {
        let c = rand_const(rng, typ);
        let loc = get_loc(typ);
        gen_test(
            writer,
            &format!("local tee {} [v={}]", typ, c),
            vec![
                Instr::Const(c),
                Instr::LocalTee(loc),
                Instr::Const(c),
                cmpeq(typ),
                Instr::Const(c),
                Instr::LocalGet(loc),
                cmpeq(typ),
                Instr::IBinOp(BitSize::B32, intop::BinOp::And),
            ],
        )
    }

    (
        vec![],
        vec![ValType::I32, ValType::I64, ValType::F32, ValType::F64],
        vec![
            gen_initial(writer, ValType::I32),
            gen_initial(writer, ValType::I64),
            gen_initial(writer, ValType::F32),
            gen_initial(writer, ValType::F64),
            gen_local_write(rng, writer, ValType::I32),
            gen_local_write(rng, writer, ValType::I64),
            gen_local_write(rng, writer, ValType::F32),
            gen_local_write(rng, writer, ValType::F64),
            gen_local_tee(rng, writer, ValType::I32),
            gen_local_tee(rng, writer, ValType::I64),
            gen_local_tee(rng, writer, ValType::F32),
            gen_local_tee(rng, writer, ValType::F64),
        ],
    )
}

fn check_loops(
    _rng: &mut rng::Rng,
    writer: &mut WasmWriter,
) -> (
    Vec<wasm::syntax::Const>,
    Vec<wasm::syntax::ValType>,
    Vec<wasm::syntax::Instr>,
) {
    use wasm::syntax::*;

    // A "fuel" counter to prevent infinite loops
    let locals = vec![ValType::I32];

    let loop_info = writer.write(
        "[i] NOTE: Some automated checking of loops occurs, \
         but it is important to manually check whether loop counts \
         match what is expected.\n",
    );
    let loop_iteration = writer.write("    > loop iteration\n");
    let failure = writer.write("[!] Failure. Exiting.\n");

    let body = vec![
        loop_info,
        gen_test(
            writer,
            "loop - instant quit (stuck ==> bad)",
            vec![
                Instr::Block(
                    BlockType::ValType(None),
                    vec![Instr::Loop(
                        BlockType::ValType(None),
                        vec![
                            // This should quit
                            Instr::Br(LabelIdx(1)),
                            // But if it doesn't, then we have instantly failed
                            failure.clone(),
                            instr_wasm_exit_with(3),
                        ],
                    )],
                ),
                Instr::Const(Const::I32(1)),
            ],
        ),
        gen_test(
            writer,
            &format!("loop - quit after 3 iterations (anything else ==> bad)"),
            vec![
                Instr::Const(Const::I32(3)),
                Instr::LocalSet(LocalIdx(0)),
                Instr::Block(
                    BlockType::ValType(None),
                    vec![Instr::Loop(
                        BlockType::ValType(None),
                        vec![
                            loop_iteration.clone(),
                            Instr::LocalGet(LocalIdx(0)),
                            Instr::Const(Const::I32(1)),
                            Instr::IBinOp(BitSize::B32, intop::BinOp::Sub),
                            Instr::LocalTee(LocalIdx(0)),
                            Instr::ITestOp(BitSize::B32, intop::TestOp::Eqz),
                            Instr::BrIf(LabelIdx(1)),
                            Instr::Br(LabelIdx(0)),
                        ],
                    )],
                ),
                Instr::Const(Const::I32(1)),
            ],
        ),
        gen_test(
            writer,
            &format!("loop - fallthrough instant quit (stuck ==> bad)"),
            vec![Instr::Loop(
                BlockType::ValType(Some(ValType::I32)),
                vec![Instr::Const(Const::I32(1))],
            )],
        ),
        // Unsupported in vWasm
        //    multi-value merged into the WebAssembly standard on 2020/04/09
        //    See vWasm-internal#129
        /*
        gen_test(
            writer,
            &format!("loop - carry-through value for 3 iterations (anything else ==> bad)"),
            vec![
                Instr::Const(Const::I32(3)),
                Instr::Block(
                    BlockType::TypeIdx(TypeIdx(3)),
                    vec![Instr::Loop(
                        BlockType::TypeIdx(TypeIdx(3)),
                        vec![
                            loop_iteration.clone(),
                            Instr::Const(Const::I32(1)),
                            Instr::IBinOp(BitSize::B32, intop::BinOp::Sub),
                            Instr::LocalTee(LocalIdx(0)),
                            Instr::LocalGet(LocalIdx(0)),
                            Instr::ITestOp(BitSize::B32, intop::TestOp::Eqz),
                            Instr::BrIf(LabelIdx(1)),
                            Instr::LocalGet(LocalIdx(0)),
                            Instr::Br(LabelIdx(0)),
                        ],
                    )],
                ),
                Instr::ITestOp(BitSize::B32, intop::TestOp::Eqz),
            ],
        ),
        */
        gen_test(
            writer,
            &format!("loop - quit after 3 iterations using branch table (anything else ==> bad)"),
            vec![
                Instr::Const(Const::I32(3)),
                Instr::LocalSet(LocalIdx(0)),
                Instr::Block(
                    BlockType::ValType(None),
                    vec![
                        Instr::Loop(
                            BlockType::ValType(None),
                            vec![
                                loop_iteration.clone(),
                                Instr::LocalGet(LocalIdx(0)),
                                Instr::Const(Const::I32(1)),
                                Instr::IBinOp(BitSize::B32, intop::BinOp::Sub),
                                Instr::LocalTee(LocalIdx(0)),
                                Instr::BrTable(
                                    vec![LabelIdx(1), LabelIdx(0), LabelIdx(0), LabelIdx(0)],
                                    LabelIdx(1),
                                ),
                                // Should never reach here
                                failure.clone(),
                                instr_wasm_exit_with(5),
                            ],
                        ),
                        // Should never reach here
                        failure.clone(),
                        instr_wasm_exit_with(4),
                    ],
                ),
                Instr::Const(Const::I32(1)),
            ],
        ),
    ];

    (vec![], locals, body)
}

fn cmpeq(typ: wasm::syntax::ValType) -> wasm::syntax::Instr {
    use wasm::syntax::*;
    match typ {
        ValType::I32 => Instr::IRelOp(BitSize::B32, intop::RelOp::Eq),
        ValType::I64 => Instr::IRelOp(BitSize::B64, intop::RelOp::Eq),
        ValType::F32 => Instr::FRelOp(BitSize::B32, floatop::RelOp::Eq),
        ValType::F64 => Instr::FRelOp(BitSize::B64, floatop::RelOp::Eq),
    }
}

fn rand_const(rng: &mut rng::Rng, typ: wasm::syntax::ValType) -> wasm::syntax::Const {
    use wasm::syntax::*;
    match typ {
        ValType::I32 => Const::I32(gen_rand_i32(rng, None)),
        ValType::I64 => Const::I64(gen_rand_i64(rng, None)),
        ValType::F32 => Const::F32(gen_rand_f32(rng, None)),
        ValType::F64 => Const::F64(gen_rand_f64(rng, None)),
    }
}

fn const_zero(typ: wasm::syntax::ValType) -> wasm::syntax::Const {
    use wasm::syntax::*;
    match typ {
        ValType::I32 => Const::I32(0),
        ValType::I64 => Const::I64(0),
        ValType::F32 => Const::F32(0.),
        ValType::F64 => Const::F64(0.),
    }
}

fn check_parametrics(rng: &mut rng::Rng, writer: &mut WasmWriter) -> Vec<wasm::syntax::Instr> {
    use wasm::syntax::*;

    let test_i32_1 = Const::I32(gen_rand_i32(rng, None));
    let test_i32_2 = Const::I32(gen_rand_i32(rng, None));
    let test_i64_1 = Const::I64(gen_rand_i64(rng, None));
    let test_i64_2 = Const::I64(gen_rand_i64(rng, None));
    let test_f32_1 = Const::F32(gen_rand_f32(rng, None));
    let test_f32_2 = Const::F32(gen_rand_f32(rng, None));
    let test_f64_1 = Const::F64(gen_rand_f64(rng, None));
    let test_f64_2 = Const::F64(gen_rand_f64(rng, None));

    fn gen_drop(writer: &mut WasmWriter, typ: ValType, c1: Const, c2: Const) -> Instr {
        gen_test(
            writer,
            &format!("drop {}", typ),
            vec![
                Instr::Const(c1),
                Instr::Const(c2),
                Instr::Drop,
                Instr::Const(c1),
                cmpeq(typ),
            ],
        )
    }

    fn gen_select(writer: &mut WasmWriter, typ: ValType, c1: Const, c2: Const, s: i32) -> Instr {
        gen_test(
            writer,
            &format!(
                "select {}: {}",
                typ,
                match s {
                    0 => "zero",
                    1 => "one",
                    2 => "two",
                    _ => unreachable!(),
                }
            ),
            vec![
                Instr::Const(c1),
                Instr::Const(c2),
                Instr::Const(Const::I32(s)),
                Instr::Select,
                Instr::Const(if s == 0 { c2 } else { c1 }),
                cmpeq(typ),
            ],
        )
    }

    vec![
        gen_drop(writer, ValType::I32, test_i32_1, test_i32_2),
        gen_select(writer, ValType::I32, test_i32_1, test_i32_2, 0),
        gen_select(writer, ValType::I32, test_i32_1, test_i32_2, 1),
        gen_select(writer, ValType::I32, test_i32_1, test_i32_2, 2),
        gen_drop(writer, ValType::I64, test_i64_1, test_i64_2),
        gen_select(writer, ValType::I64, test_i64_1, test_i64_2, 0),
        gen_select(writer, ValType::I64, test_i64_1, test_i64_2, 1),
        gen_select(writer, ValType::I64, test_i64_1, test_i64_2, 2),
        gen_drop(writer, ValType::F32, test_f32_1, test_f32_2),
        gen_select(writer, ValType::F32, test_f32_1, test_f32_2, 0),
        gen_select(writer, ValType::F32, test_f32_1, test_f32_2, 1),
        gen_select(writer, ValType::F32, test_f32_1, test_f32_2, 2),
        gen_drop(writer, ValType::F64, test_f64_1, test_f64_2),
        gen_select(writer, ValType::F64, test_f64_1, test_f64_2, 0),
        gen_select(writer, ValType::F64, test_f64_1, test_f64_2, 1),
        gen_select(writer, ValType::F64, test_f64_1, test_f64_2, 2),
    ]
}

fn with_safe_float_cmp(
    mut test_instructions: Vec<wasm::syntax::Instr>,
    bitsize: wasm::syntax::BitSize,
    op: wasm::syntax::floatop::RelOp,
) -> Vec<wasm::syntax::Instr> {
    // Note: test_instrs should be of type []->[f{bitsize},
    // f{bitsize}] and the returned value from this function will be
    // []->[i32] (i.e., directly usable for `gen_test`).
    use floatop::RelOp::*;
    use wasm::syntax::*;

    let epsilon = match bitsize {
        BitSize::B32 => Const::F32(f32::EPSILON),
        BitSize::B64 => Const::F64(f64::EPSILON),
    };

    match op {
        Eq => test_instructions.extend(vec![
            Instr::FBinOp(bitsize, floatop::BinOp::Sub),
            Instr::FUnOp(bitsize, floatop::UnOp::Abs),
            Instr::Const(epsilon),
            Instr::FRelOp(bitsize, floatop::RelOp::Le),
        ]),
        Ne => test_instructions.extend(vec![
            Instr::FBinOp(bitsize, floatop::BinOp::Sub),
            Instr::FUnOp(bitsize, floatop::UnOp::Abs),
            Instr::Const(epsilon),
            Instr::FRelOp(bitsize, floatop::RelOp::Gt),
        ]),
        Lt | Gt | Le | Ge => test_instructions.push(Instr::FRelOp(bitsize, op)),
    }

    test_instructions
}

fn check_basic_floating_epsilon(
    rng: &mut rng::Rng,
    writer: &mut WasmWriter,
) -> Vec<wasm::syntax::Instr> {
    use wasm::syntax::*;

    let c_f32: f32 = rng.next() as f32 / rng.next() as f32;
    let c_f64: f64 = rng.next() as f64 / rng.next() as f64;

    vec![
        gen_test(
            writer,
            &format!("32-bit floating epsilon equality for {}", c_f32),
            with_safe_float_cmp(
                vec![
                    Instr::Const(Const::F32(c_f32)),
                    Instr::Const(Const::F32(c_f32)),
                ],
                BitSize::B32,
                floatop::RelOp::Eq,
            ),
        ),
        gen_test(
            writer,
            &format!("64-bit floating epsilon equality for {}", c_f64),
            with_safe_float_cmp(
                vec![
                    Instr::Const(Const::F64(c_f64)),
                    Instr::Const(Const::F64(c_f64)),
                ],
                BitSize::B64,
                floatop::RelOp::Eq,
            ),
        ),
        gen_test(
            writer,
            &format!("32-bit floating epsilon equality for {} + 0", c_f32),
            with_safe_float_cmp(
                vec![
                    Instr::Const(Const::F32(c_f32)),
                    Instr::Const(Const::F32(0.)),
                    Instr::FBinOp(BitSize::B32, floatop::BinOp::Add),
                    Instr::Const(Const::F32(c_f32)),
                ],
                BitSize::B32,
                floatop::RelOp::Eq,
            ),
        ),
        gen_test(
            writer,
            &format!("64-bit floating epsilon equality for {} + 0", c_f64),
            with_safe_float_cmp(
                vec![
                    Instr::Const(Const::F64(c_f64)),
                    Instr::Const(Const::F64(0.)),
                    Instr::FBinOp(BitSize::B64, floatop::BinOp::Add),
                    Instr::Const(Const::F64(c_f64)),
                ],
                BitSize::B64,
                floatop::RelOp::Eq,
            ),
        ),
        gen_test(
            writer,
            &format!("32-bit {0} + 1. != {0}", c_f64),
            with_safe_float_cmp(
                vec![
                    Instr::Const(Const::F32(c_f32)),
                    Instr::Const(Const::F32(1.)),
                    Instr::FBinOp(BitSize::B32, floatop::BinOp::Add),
                    Instr::Const(Const::F32(c_f32)),
                ],
                BitSize::B32,
                floatop::RelOp::Ne,
            ),
        ),
    ]
}

fn gen_funop(rng: &mut rng::Rng, writer: &mut WasmWriter) -> Option<wasm::syntax::Instr> {
    use floatop::UnOp::*;
    use wasm::syntax::*;

    let bitsize = *rng.choice(&[BitSize::B32, BitSize::B64]);

    let a: Const = match bitsize {
        BitSize::B32 => Const::F32(gen_rand_f32(rng, None)),
        BitSize::B64 => Const::F64(gen_rand_f64(rng, None)),
    };
    let op = *rng.choice(&[Neg, Abs, Ceil, Floor, Trunc, Nearest, Sqrt]);
    let res = match a {
        Const::I32(_) => unreachable!(),
        Const::I64(_) => unreachable!(),
        Const::F32(a) => Const::F32(match op {
            Neg => -a,
            Abs => a.abs(),
            Ceil => a.ceil(),
            Floor => a.floor(),
            Trunc => a.trunc(),
            Nearest => a.round(),
            Sqrt => a.sqrt(),
        }),
        Const::F64(a) => Const::F64(match op {
            Neg => -a,
            Abs => a.abs(),
            Ceil => a.ceil(),
            Floor => a.floor(),
            Trunc => a.trunc(),
            Nearest => a.round(),
            Sqrt => a.sqrt(),
        }),
    };

    match res {
        Const::F32(r) => {
            if r.is_nan() || (r - r).abs().is_nan() {
                // our floating comparison routine would falsely show
                // a failure here, so we just don't generate this.
                return None;
            }
        }
        Const::F64(r) => {
            if r.is_nan() || (r - r).abs().is_nan() {
                // our floating comparison routine would falsely show
                // a failure here, so we just don't generate this.
                return None;
            }
        }
        Const::I32(_) | Const::I64(_) => unreachable!(),
    }

    Some(gen_test(
        writer,
        &format!("({} bit funop) {:?} {} =? {}", bitsize, op, a, res),
        with_safe_float_cmp(
            vec![
                Instr::Const(a),
                Instr::FUnOp(bitsize, op),
                Instr::Const(res),
            ],
            bitsize,
            floatop::RelOp::Eq,
        ),
    ))
}

fn gen_fbinop(rng: &mut rng::Rng, writer: &mut WasmWriter) -> Option<wasm::syntax::Instr> {
    use floatop::BinOp::*;
    use wasm::syntax::*;

    let bitsize: BitSize = *rng.choice(&[BitSize::B32, BitSize::B64]);

    let a: Const = match bitsize {
        BitSize::B32 => Const::F32(gen_rand_f32(rng, None)),
        BitSize::B64 => Const::F64(gen_rand_f64(rng, None)),
    };
    let b: Const = match a {
        Const::I32(_) | Const::I64(_) => unreachable!(),
        Const::F32(a) => Const::F32(gen_rand_f32(rng, Some(a))),
        Const::F64(a) => Const::F64(gen_rand_f64(rng, Some(a))),
    };
    let op = *rng.choice(&[Add, Sub, Mul, Div, Min, Max, CopySign]);
    let res = match (a, b) {
        (Const::F32(a), Const::F32(b)) => Const::F32(match op {
            Add => a + b,
            Sub => a - b,
            Mul => a * b,
            Div => a / b,
            Min => a.min(b),
            Max => a.max(b),
            CopySign => a.copysign(b),
        }),
        (Const::F64(a), Const::F64(b)) => Const::F64(match op {
            Add => a + b,
            Sub => a - b,
            Mul => a * b,
            Div => a / b,
            Min => a.min(b),
            Max => a.max(b),
            CopySign => a.copysign(b),
        }),
        _ => unreachable!(),
    };

    match res {
        Const::F32(r) => {
            if r.is_nan() || (r - r).abs().is_nan() {
                // our floating comparison routine would falsely show
                // a failure here, so we just don't generate this.
                return None;
            }
        }
        Const::F64(r) => {
            if r.is_nan() || (r - r).abs().is_nan() {
                // our floating comparison routine would falsely show
                // a failure here, so we just don't generate this.
                return None;
            }
        }
        Const::I32(_) | Const::I64(_) => unreachable!(),
    }

    Some(gen_test(
        writer,
        &format!("({} bit fbinop) {} {:?} {} =? {}", bitsize, a, op, b, res),
        with_safe_float_cmp(
            vec![
                Instr::Const(a),
                Instr::Const(b),
                Instr::FBinOp(bitsize, op),
                Instr::Const(res),
            ],
            bitsize,
            floatop::RelOp::Eq,
        ),
    ))
}

fn gen_frelop(rng: &mut rng::Rng, writer: &mut WasmWriter) -> Option<wasm::syntax::Instr> {
    use floatop::RelOp::*;
    use wasm::syntax::*;

    let bitsize = *rng.choice(&[BitSize::B32, BitSize::B64]);

    let a: Const = match bitsize {
        BitSize::B32 => Const::F32(gen_rand_f32(rng, None)),
        BitSize::B64 => Const::F64(gen_rand_f64(rng, None)),
    };
    let b: Const = match a {
        Const::I32(_) | Const::I64(_) => unreachable!(),
        Const::F32(a) => Const::F32(gen_rand_f32(rng, Some(a))),
        Const::F64(a) => Const::F64(gen_rand_f64(rng, Some(a))),
    };
    let relop = *rng.choice(&[Eq, Ne, Lt, Gt, Le, Ge]);

    let res = Const::I32(match (a, b) {
        (Const::F32(a), Const::F32(b)) => match relop {
            Eq => a == b,
            Ne => a != b,
            Lt => a < b,
            Gt => a > b,
            Le => a <= b,
            Ge => a >= b,
        },
        (Const::F64(a), Const::F64(b)) => match relop {
            Eq => a == b,
            Ne => a != b,
            Lt => a < b,
            Gt => a > b,
            Le => a <= b,
            Ge => a >= b,
        },
        _ => unreachable!(),
    } as i32);

    Some(gen_test(
        writer,
        &format!(
            "({} bit frelop) {} {:?} {} =? {}",
            bitsize, a, relop, b, res
        ),
        vec![
            Instr::Const(a),
            Instr::Const(b),
            Instr::FRelOp(bitsize, relop),
            Instr::Const(res),
            Instr::IRelOp(BitSize::B32, intop::RelOp::Eq),
        ],
    ))
}

fn gen_fcvtop(rng: &mut rng::Rng, writer: &mut WasmWriter) -> Option<wasm::syntax::Instr> {
    use floatop::CvtOp::*;
    use wasm::syntax::*;
    use BitSize::*;

    // Note: We explicitly _disable_ TruncSat since it didn't exist on
    // a (slightly) older version of Wasm, which is what vWasm and
    // wat2vasm is built on. Compilers don't seem to emit this
    // instruction (yet) so we can punt the implementation of this in
    // vWasm and wat2vasm for later.
    let op = *rng.choice(&[
        ConvertSI32,
        ConvertUI32,
        ConvertSI64,
        ConvertUI64,
        PromoteF32,
        DemoteF64,
        ReinterpretInt,
    ]);
    let bitsize = *rng.choice(&[B32, B64]);

    let r_i32: i32 = gen_rand_i32(rng, None);
    let r_i64: i64 = gen_rand_i64(rng, None);
    let r_f32: f32 = gen_rand_f32(rng, None);
    let r_f64: f64 = gen_rand_f64(rng, None);

    let c_i32: Const = Const::I32(r_i32);
    let c_i64: Const = Const::I64(r_i64);
    let c_f32: Const = Const::F32(r_f32);
    let c_f64: Const = Const::F64(r_f64);

    let input = match (op, bitsize) {
        (ConvertSI32, _) => c_i32,
        (ConvertUI32, _) => c_i32,
        (ConvertSI64, _) => c_i64,
        (ConvertUI64, _) => c_i64,
        (PromoteF32, B32) => return None,
        (PromoteF32, B64) => c_f32,
        (DemoteF64, B32) => c_f64,
        (DemoteF64, B64) => return None,
        (ReinterpretInt, B32) => c_i32,
        (ReinterpretInt, B64) => c_i64,
    };

    let output = match bitsize {
        B32 => Const::F32(match op {
            ConvertSI32 => r_i32 as f32,
            ConvertUI32 => r_i32 as u32 as f32,
            ConvertSI64 => r_i64 as f32,
            ConvertUI64 => r_i64 as u64 as f32,
            PromoteF32 => unreachable!(),
            DemoteF64 => r_f64 as f32,
            ReinterpretInt => f32::from_bits(r_i32 as u32),
        }),
        B64 => Const::F64(match op {
            ConvertSI32 => r_i32 as f64,
            ConvertUI32 => r_i32 as u32 as f64,
            ConvertSI64 => r_i64 as f64,
            ConvertUI64 => r_i64 as u64 as f64,
            PromoteF32 => r_f32 as f64,
            DemoteF64 => unreachable!(),
            ReinterpretInt => f64::from_bits(r_i64 as u64),
        }),
    };

    match output {
        Const::F32(r) => {
            if r.is_nan() || (r - r).abs().is_nan() {
                // our floating comparison routine would falsely show
                // a failure here, so we just don't generate this.
                return None;
            }
        }
        Const::F64(r) => {
            if r.is_nan() || (r - r).abs().is_nan() {
                // our floating comparison routine would falsely show
                // a failure here, so we just don't generate this.
                return None;
            }
        }
        Const::I32(_) | Const::I64(_) => unreachable!(),
    };

    Some(gen_test(
        writer,
        &format!("({} bit fcvtop) {:?} {} =? {}", bitsize, op, input, output),
        with_safe_float_cmp(
            vec![
                Instr::Const(input),
                Instr::FCvtOp(bitsize, op),
                Instr::Const(output),
            ],
            bitsize,
            floatop::RelOp::Eq,
        ),
    ))
}

#[rustfmt::skip] // temporarily skip auto-reformatting this function
fn gen_fops(num: usize, rng: &mut rng::Rng, writer: &mut WasmWriter) -> Vec<wasm::syntax::Instr> {
    (0..)
        .filter_map(|_i| {
            rng.choice::<fn(&mut _, &mut _) -> _>(&[
                gen_fbinop,
                gen_funop,
                gen_frelop,
                gen_fcvtop,
            ])(rng, writer)
        })
        .take(num)
        .collect()
}

fn check_args(
    rng: &mut rng::Rng,
    writer: &mut WasmWriter,
) -> (
    Vec<wasm::syntax::Const>,
    Vec<wasm::syntax::ValType>,
    Vec<wasm::syntax::Instr>,
) {
    use wasm::syntax::*;

    let args: Vec<Const> = (0..rng.next() % 35 + 1)
        .map(|_| {
            let ty = *rng.choice(&[ValType::I32, ValType::I64, ValType::F32, ValType::F64]);
            rand_const(rng, ty)
        })
        .collect();

    let body: Vec<Instr> =
        std::iter::once(writer.write(format!("[i] Checking {} args\n", args.len())))
            .chain(args.iter().enumerate().map(|(i, &arg)| {
                gen_test(
                    writer,
                    &format!("Argument {} @ {} (v={})", valtype_of_const(arg), i, arg),
                    vec![
                        Instr::LocalGet(LocalIdx(i as u32)),
                        Instr::Const(arg),
                        cmpeq(valtype_of_const(arg)),
                    ],
                )
            }))
            .collect();

    (args, vec![], body)
}

fn get_module() -> wasm::syntax::Module {
    let mut rng = rng::Rng::new();
    let mut writer = WasmWriter::new();

    fn c<T, U, V>(v: V) -> (Vec<T>, Vec<U>, V) {
        (vec![], vec![], v)
    }

    build_module_from(
        vec![
            c(gen_iops(100, &mut rng, &mut writer)),
            c(check_memory_size(&mut rng, &mut writer)),
            c(check_memory_ops(&mut rng, &mut writer)),
            c(check_global_ops(&mut rng, &mut writer)),
            check_local_ops(&mut rng, &mut writer),
            check_loops(&mut rng, &mut writer),
            c(check_parametrics(&mut rng, &mut writer)),
            c(check_basic_floating_epsilon(&mut rng, &mut writer)),
            c(gen_fops(100, &mut rng, &mut writer)),
            check_args(&mut rng, &mut writer),
        ],
        writer,
        &mut rng,
    )
}

fn main() -> Maybe<()> {
    use std::io::Write;

    color_eyre::install()?;

    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 || !args[1].ends_with(".wasm") {
        println!("Usage:\n\t{} {{output.wasm}}", args[0]);
        std::process::exit(1);
    }

    let mut output_file = std::fs::File::create(&args[1])?;
    output_file.write_all(&*serializer::serialize_module(get_module()))?;

    println!("Done writing to {}", args[1]);

    Ok(())
}
