//! MAP model (Multiply-Add-Permute, Gayler): bipolar `{-1, +1}` hypervectors.
//!
//! Less memory-efficient than the bit-packed [`crate::Hypervector`] (BSC), but its
//! **linear** bundling (sum) makes it a natural fit for compositional reasoning
//! (records, analogies). Each vector is its own inverse under `bind`.

use crate::rng::Rng;

/// A bipolar hypervector (components in `{-1, +1}`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MapVector {
    data: Vec<i8>,
}

impl MapVector {
    /// A random bipolar hypervector of dimension `d`.
    pub fn random(d: usize, rng: &mut Rng) -> Self {
        let mut data = Vec::with_capacity(d);
        let mut bits = 0u64;
        let mut left = 0u32;
        for _ in 0..d {
            if left == 0 {
                bits = rng.next_u64();
                left = 64;
            }
            data.push(if bits & 1 == 1 { 1 } else { -1 });
            bits >>= 1;
            left -= 1;
        }
        MapVector { data }
    }

    /// Dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// Binding (`⊗`): element-wise product. Invertible (self-inverse).
    pub fn bind(&self, other: &MapVector) -> MapVector {
        assert_eq!(self.dim(), other.dim(), "dimension mismatch");
        MapVector {
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| a * b)
                .collect(),
        }
    }

    /// Permutation (`ρ`): cyclic shift by `k` (for sequences).
    pub fn permute(&self, k: i64) -> MapVector {
        let d = self.data.len();
        let shift = (((k % d as i64) + d as i64) % d as i64) as usize;
        let mut out = vec![0i8; d];
        for i in 0..d {
            out[(i + shift) % d] = self.data[i];
        }
        MapVector { data: out }
    }

    /// Cosine similarity in `[-1, 1]` (`dot / d`, since components are `±1`).
    pub fn similarity(&self, other: &MapVector) -> f64 {
        let dot: i64 = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| (*a as i64) * (*b as i64))
            .sum();
        dot as f64 / self.data.len() as f64
    }
}

/// Bundling (`⊕`): element-wise sum, then sign. Result similar to all members.
/// Ties resolve to `+1`.
pub fn bundle(vs: &[MapVector]) -> MapVector {
    assert!(!vs.is_empty(), "bundle needs at least one vector");
    let d = vs[0].data.len();
    let mut acc = vec![0i32; d];
    for v in vs {
        for i in 0..d {
            acc[i] += v.data[i] as i32;
        }
    }
    MapVector {
        data: acc.iter().map(|&s| if s >= 0 { 1 } else { -1 }).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rng;

    const D: usize = 10_000;

    #[test]
    fn quasi_orthogonal_and_self() {
        let mut rng = Rng::new(5);
        let a = MapVector::random(D, &mut rng);
        let b = MapVector::random(D, &mut rng);
        assert!(a.similarity(&b).abs() < 0.05);
        assert!((a.similarity(&a) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn bind_is_invertible() {
        let mut rng = Rng::new(6);
        let a = MapVector::random(D, &mut rng);
        let b = MapVector::random(D, &mut rng);
        let c = a.bind(&b);
        assert!(c.similarity(&a).abs() < 0.05);
        assert!((c.bind(&b).similarity(&a) - 1.0).abs() < 1e-9);
    }
}
