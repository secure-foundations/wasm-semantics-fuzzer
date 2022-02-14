//! Fast pseudo random number generation

pub type Rng = xoshiro::Rng;

#[cfg(all(target_arch = "x86_64"))]
pub fn rdtsc() -> u64 {
    // SAFETY: Valid in x86_64 in userland. If invalid on some x86_64
    // OS, it is guaranteed to #GP fault, which is noisy enough to
    // instantly be detected.
    unsafe { core::arch::x86_64::_rdtsc() }
}

mod xoshiro {
    use std::num::Wrapping;

    /// A pseudo random number generator state.
    ///
    /// Uses xoshiro256** to produce a stream of 64-bit random numbers
    /// with an internal state space of 256 bits.
    pub struct Rng {
        a: Wrapping<u64>,
        b: Wrapping<u64>,
        c: Wrapping<u64>,
        d: Wrapping<u64>,
    }

    impl Rng {
        /// Produces a new instance from a seed derived from current
        /// time stamp of the system.
        pub fn new() -> Self {
            Self::new_seeded(super::rdtsc())
        }

        /// Produces a new instance from a fixed 64-bit seed. Uses the
        /// SplitMix64 PRNG to initialize the internal 256-bit state to
        /// ensure sufficient randomness while still ensuring determinism.
        pub fn new_seeded(seed: u64) -> Self {
            fn splitmix64(state: &mut Wrapping<u64>) -> Wrapping<u64> {
                let mut result = *state;

                *state = result + Wrapping(0x9E3779B97f4A7C15);
                result = (result ^ (result >> 30)) * Wrapping(0xBF58476D1CE4E5B9);
                result = (result ^ (result >> 27)) * Wrapping(0x94D049BB133111EB);
                result = result ^ (result >> 31);

                assert_ne!(result.0, 0);

                result
            }

            let mut smstate = Wrapping(seed);
            let a = splitmix64(&mut smstate);
            let b = splitmix64(&mut smstate);
            let c = splitmix64(&mut smstate);
            let d = splitmix64(&mut smstate);

            Rng { a, b, c, d }
        }

        /// Generate a new pseudo-random `u64`
        pub fn next(&mut self) -> u64 {
            fn rol64(x: Wrapping<u64>, n: u32) -> Wrapping<u64> {
                Wrapping(x.0.rotate_left(n))
            }

            let result = rol64(self.b * Wrapping(5), 7) * Wrapping(9);
            let t = self.b << 17;

            self.c ^= self.a;
            self.d ^= self.b;
            self.b ^= self.c;
            self.a ^= self.d;

            self.c ^= t;
            self.d = rol64(self.d, 45);

            result.0
        }

        /// Pick an arbitrary element of `values`
        pub fn choice<'a, T>(&mut self, values: &'a [T]) -> &'a T {
            assert_ne!(values.len(), 0);
            let index = (self.next() as usize) % values.len();
            &values[index]
        }
    }
}
