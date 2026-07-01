//! BSC (Binary Spatter Code) model: binary hypervectors bit-packed into `u64` words.
//!
//! - bind      = XOR bit a bit         (invertible, orthogonal to both operands)
//! - bundle    = majority bit a bit    (similar to all its members)
//! - permute   = cyclic bit rotation   (encode order / sequences)
//! - similarity= Hamming via popcount  (equivalent to bipolar cosine in [-1, 1])

use crate::rng::Rng;

/// A binary hypervector (BSC model), bit-packed into 64-bit words.
///
/// Padding bits in the final word (beyond `d`) are always kept at 0, so they
/// never affect Hamming distance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hypervector {
    pub(crate) d: usize,
    pub(crate) words: Vec<u64>,
}

impl Hypervector {
    #[inline]
    pub(crate) fn n_words(d: usize) -> usize {
        (d + 63) / 64
    }

    #[inline]
    fn tail_mask(d: usize) -> u64 {
        let rem = d % 64;
        if rem == 0 {
            u64::MAX
        } else {
            (1u64 << rem) - 1
        }
    }

    /// All-zeros hypervector of dimension `d`.
    pub fn zeros(d: usize) -> Self {
        Hypervector {
            d,
            words: vec![0u64; Self::n_words(d)],
        }
    }

    /// A random hypervector (~50% of bits set). Padding bits are 0.
    pub fn random(d: usize, rng: &mut Rng) -> Self {
        let nw = Self::n_words(d);
        let mut words: Vec<u64> = (0..nw).map(|_| rng.next_u64()).collect();
        if let Some(last) = words.last_mut() {
            *last &= Self::tail_mask(d);
        }
        Hypervector { d, words }
    }

    /// Dimension (number of valid bits).
    #[inline]
    pub fn dim(&self) -> usize {
        self.d
    }

    /// Raw packed words (crate-internal, used by [`crate::ItemMemory`]).
    #[inline]
    pub(crate) fn words(&self) -> &[u64] {
        &self.words
    }

    /// Binding (`⊗`): XOR. Result is orthogonal to both operands and invertible.
    pub fn bind(&self, other: &Hypervector) -> Hypervector {
        assert_eq!(self.d, other.d, "dimension mismatch");
        let words = self
            .words
            .iter()
            .zip(&other.words)
            .map(|(a, b)| a ^ b)
            .collect();
        Hypervector { d: self.d, words }
    }

    /// Hamming distance (number of differing bits) via hardware popcount.
    pub fn hamming(&self, other: &Hypervector) -> u32 {
        self.words
            .iter()
            .zip(&other.words)
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }

    /// Bipolar-cosine-style similarity in `[-1, 1]`: `1 - 2·(hamming/d)`.
    /// ~0 = unrelated, 1 = identical, -1 = opposite.
    pub fn similarity(&self, other: &Hypervector) -> f64 {
        1.0 - 2.0 * self.hamming(other) as f64 / self.d as f64
    }

    /// Return a copy with exactly `fraction` of its (distinct) bits flipped.
    /// Useful for robustness tests: similarity to the original becomes `1 - 2·fraction`.
    pub fn add_noise(&self, fraction: f64, rng: &mut Rng) -> Hypervector {
        let mut out = self.clone();
        let flips = ((fraction * self.d as f64).round() as usize).min(self.d);
        // Partial Fisher-Yates: pick `flips` distinct positions, flip each once.
        let mut idx: Vec<usize> = (0..self.d).collect();
        for k in 0..flips {
            let j = k + (rng.next_u64() as usize) % (self.d - k);
            idx.swap(k, j);
            let pos = idx[k];
            out.words[pos / 64] ^= 1u64 << (pos % 64);
        }
        out
    }

    /// Serialize to a compact byte buffer: `d` (u64 LE) followed by the packed words.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + self.words.len() * 8);
        out.extend_from_slice(&(self.d as u64).to_le_bytes());
        for w in &self.words {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    }

    /// Deserialize a hypervector produced by [`Hypervector::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Option<Hypervector> {
        if bytes.len() < 8 {
            return None;
        }
        let d = u64::from_le_bytes(bytes[0..8].try_into().ok()?) as usize;
        let nw = Self::n_words(d);
        if bytes.len() != 8 + nw * 8 {
            return None;
        }
        let mut words = Vec::with_capacity(nw);
        for k in 0..nw {
            let off = 8 + k * 8;
            words.push(u64::from_le_bytes(bytes[off..off + 8].try_into().ok()?));
        }
        Some(Hypervector { d, words })
    }

    #[inline]
    pub(crate) fn get_bit(&self, i: usize) -> u64 {
        (self.words[i / 64] >> (i % 64)) & 1
    }

    /// (crate-internal) set the bit at `pos` to 1.
    #[inline]
    pub(crate) fn set_bit(&mut self, pos: usize) {
        self.words[pos / 64] |= 1u64 << (pos % 64);
    }

    /// (crate-internal) flip the bit at `pos`.
    #[inline]
    pub(crate) fn flip(&mut self, pos: usize) {
        self.words[pos / 64] ^= 1u64 << (pos % 64);
    }
}

/// Bundling (`⊕`): bit-wise majority over several hypervectors.
/// The result is similar to every member. Ties resolve to 0.
pub fn bundle(hvs: &[Hypervector]) -> Hypervector {
    assert!(!hvs.is_empty(), "bundle needs at least one hypervector");
    let d = hvs[0].d;
    let n = hvs.len();
    let mut out = Hypervector::zeros(d);
    for i in 0..d {
        let count: usize = hvs.iter().map(|hv| hv.get_bit(i) as usize).sum();
        if 2 * count > n {
            out.words[i / 64] |= 1u64 << (i % 64);
        }
    }
    out
}

/// Permutation (`ρ`): cyclic rotation of the bits by `k` positions.
/// Produces a vector quasi-orthogonal to the original; invert with `-k`.
pub fn permute(hv: &Hypervector, k: i64) -> Hypervector {
    let d = hv.d;
    let shift = (((k % d as i64) + d as i64) % d as i64) as usize;
    let mut out = Hypervector::zeros(d);
    for i in 0..d {
        if hv.get_bit(i) != 0 {
            let j = (i + shift) % d;
            out.words[j / 64] |= 1u64 << (j % 64);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const D: usize = 10_000;

    #[test]
    fn random_pairs_are_quasi_orthogonal() {
        let mut rng = Rng::new(42);
        let n = 300;
        let mean: f64 = (0..n)
            .map(|_| {
                let a = Hypervector::random(D, &mut rng);
                let b = Hypervector::random(D, &mut rng);
                a.similarity(&b)
            })
            .sum::<f64>()
            / n as f64;
        assert!(mean.abs() < 0.05, "mean random similarity = {mean}");
    }

    #[test]
    fn self_similarity_is_one() {
        let mut rng = Rng::new(1);
        let a = Hypervector::random(D, &mut rng);
        assert!((a.similarity(&a) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn bind_is_invertible_and_orthogonal() {
        let mut rng = Rng::new(7);
        let a = Hypervector::random(D, &mut rng);
        let b = Hypervector::random(D, &mut rng);
        let c = a.bind(&b);
        assert!(c.similarity(&a).abs() < 0.05);
        assert!(c.similarity(&b).abs() < 0.05);
        assert!((c.bind(&b).similarity(&a) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn bundle_is_similar_to_members() {
        let mut rng = Rng::new(9);
        let a = Hypervector::random(D, &mut rng);
        let b = Hypervector::random(D, &mut rng);
        let c = Hypervector::random(D, &mut rng);
        let s = bundle(&[a.clone(), b.clone(), c.clone()]);
        assert!(s.similarity(&a) > 0.3);
        assert!(s.similarity(&b) > 0.3);
        assert!(s.similarity(&c) > 0.3);
    }

    #[test]
    fn permute_is_orthogonal_and_invertible() {
        let mut rng = Rng::new(11);
        let a = Hypervector::random(D, &mut rng);
        let p = permute(&a, 1);
        assert!(p.similarity(&a).abs() < 0.05); // rotated is unlike original
        assert!((permute(&p, -1).similarity(&a) - 1.0).abs() < 1e-9); // invertible
    }

    #[test]
    fn add_noise_degrades_gracefully() {
        let mut rng = Rng::new(13);
        let a = Hypervector::random(D, &mut rng);
        let n = a.add_noise(0.25, &mut rng);
        // ~25% flips -> similarity ~0.5 (allowing for repeated positions)
        assert!(n.similarity(&a) > 0.4 && n.similarity(&a) < 0.6);
    }

    #[test]
    fn serialize_roundtrip() {
        let mut rng = Rng::new(15);
        let a = Hypervector::random(D, &mut rng);
        let b = Hypervector::from_bytes(&a.to_bytes()).unwrap();
        assert_eq!(a, b);
    }
}
