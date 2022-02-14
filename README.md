# Wasm Semantics Fuzzer

An aid for developing correct WebAssembly implementations through
generative fuzzing. Randomly generates valid Wasm programs that check
their own results during execution, in order to help identify
potentially flawed implemented semantics. The core idea of this fuzzer
is to generate code of the form `if 2 + 3 is not 5, exit with
failure`.

This fuzzer is built to focus fuzzing efforts on easy-to-get-wrong
corner cases (for example, the handling of [subnormal floating point
numbers](https://en.wikipedia.org/wiki/Subnormal_number)), but will
also test regular behavior for correctness. It supports testing the
following types of behaviors: integer operations (with additional
focus around wrapping behaviors), floating operations (including
behavior of floating epsilon, subnormal numbers, etc.), memory
sizing/resizing, memory loads/stores (including sign extension,
endianness checks, etc.), globals, function-locals behavior, looping
and branching, parametric operations, function argument counts,
etc. See the [Trophy Case](#trophy-case) for some of the bugs it has
helped find and fix.

## Usage

Requires a working Rust installation. See
[`rustup`](https://rustup.rs/) if you need to install Rust.

To produce a single randomly-generated self-checking program:
```sh
cargo run output-file-name.wasm
```

Often, however, you want to run it in a loop alongside the program
being tested (say, `runWasm`):

```sh
while cargo run temp.wasm && runWasm temp.wasm; do echo '--------'; sleep 0.1; done
```

## Trophy Case

If you find bugs in your Wasm implementation with this fuzzer, we'd
love to hear from you!

This fuzzer helped identify and fix multiple distinct semantic
correctness issues during the development of
[vWasm](https://github.com/secure-foundations/vWasm), listed
below. While none of these threatened sandbox safety, they could lead
to incorrect computation results.

+ Signed `Div`/`Rem` semantics: for approximately half of all possible
  inputs, `Div`/`Rem` could produce erroneous results.
+ Signed division floating point exception: under just the right
  circumstances (even if it wasn't a division by zero), the program
  might cause the CPU to raise a floating point exception.
+ `MemGrow` would not update the vWasm `MEM_PAGES` global.
+ vWasm `MEM_PAGES` initialization: virtual "accessible memory" would
  start at max size, thereby preventing growth.
+ Stack alignment: code without an explicit `exit` could sometimes
  crash at the final implicit exit from `main`.
+ Memory initialization bug: non-contiguous initialization of linear
  memory could sometimes be "forgotten".
+ `GlobalSet` semantics bug: the wrong offset from the stack was being
  stored into the global.
+ `BrTable` semantics bug: one of the compiler passes could forget to
  generate relevant IL for it.
+ Unaligned reads from indirect call / indirect jump tables due to
  typo in printer
+ Incorrect backwards branches produced in a compiler pass.
+ Non-power-2 sandboxing would produce safe but practically useless
  code.
+ Parametric `Select` conditional was accidentally flipped by a
  compiler pass.
+ Register allocator issue in the presence of just the "right" order
  of function argument types (mixing integer and floating arguments in
  alternation).
+ Register allocator stack-spilled arguments could sometimes clobber
  each other.
+ Incorrect type of `mov` for certain floating operations.
+ `Nearest` rounding on floats could, under certain situations, lead
  to incorrect results.

## Related Projects

+ [rWasm](https://github.com/secure-foundations/rWasm): a
  high-performance informally-verified provably-safe sandboxing
  compiler
+ [vWasm](https://github.com/secure-foundations/vWasm): a
  formally-verified provably-safe sandboxing compiler, built in F*

## License

BSD 3-Clause License. See [LICENSE](./LICENSE).

## Publications

**Provably-Safe Multilingual Software Sandboxing using WebAssembly**.
Jay Bosamiya, Wen Shih Lim, and Bryan Parno. To Appear in Proceedings
of the USENIX Security Symposium, August, 2022.

```bibtex
@inproceedings{provably-safe-sandboxing-wasm,
  author    = {Bosamiya, Jay and Lim, Wen Shih and Parno, Bryan},
  booktitle = {To Appear in Proceedings of the USENIX Security Symposium},
  month     = {August},
  title     = {Provably-Safe Multilingual Software Sandboxing using {WebAssembly}},
  year      = {2022}
}
```
