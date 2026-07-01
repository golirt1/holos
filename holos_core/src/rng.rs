//! Small, dependency-free deterministic PRNG (SplitMix64).

/// A tiny, fast, deterministic pseudo-random number generator (SplitMix64).
///
/// Kept in-house so the crate has **zero dependencies**.
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Create a generator from a seed.
    pub fn new(seed: u64) -> Self {
        Rng { state: seed }
    }

    /// Next 64-bit pseudo-random value.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}
