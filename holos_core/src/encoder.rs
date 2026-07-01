//! Encoders: turn raw numeric feature vectors into hypervectors.
//!
//! The standard HDC "spatial" encoding: for each feature, bind its *position*
//! hypervector with a *level* hypervector representing its value, then bundle them all.
//! Nearby values map to similar level hypervectors, so similar inputs produce similar
//! hypervectors — which is what makes downstream classification/cleanup work.

use crate::bsc::{bundle, Hypervector};
use crate::rng::Rng;

/// Encodes fixed-length numeric feature vectors into hypervectors.
pub struct LevelEncoder {
    d: usize,
    n_features: usize,
    min: f64,
    max: f64,
    positions: Vec<Hypervector>,
    levels: Vec<Hypervector>, // length = n_levels + 1
}

impl LevelEncoder {
    /// Build an encoder for `n_features` values in `[min, max]`, quantized to `n_levels`.
    pub fn new(
        d: usize,
        n_features: usize,
        min: f64,
        max: f64,
        n_levels: usize,
        rng: &mut Rng,
    ) -> Self {
        assert!(max > min, "max must be greater than min");
        assert!(n_levels >= 1, "need at least one level");
        let positions = (0..n_features)
            .map(|_| Hypervector::random(d, rng))
            .collect();
        let levels = make_levels(d, n_levels, rng);
        LevelEncoder {
            d,
            n_features,
            min,
            max,
            positions,
            levels,
        }
    }

    #[inline]
    fn level_index(&self, value: f64) -> usize {
        let n = self.levels.len() - 1;
        let t = ((value - self.min) / (self.max - self.min)).clamp(0.0, 1.0);
        (t * n as f64).round() as usize
    }

    /// Encode one feature vector (its length must equal `n_features`).
    pub fn encode(&self, features: &[f64]) -> Hypervector {
        assert_eq!(features.len(), self.n_features, "feature count mismatch");
        let terms: Vec<Hypervector> = features
            .iter()
            .enumerate()
            .map(|(i, &v)| self.positions[i].bind(&self.levels[self.level_index(v)]))
            .collect();
        bundle(&terms)
    }

    /// Hypervector dimension.
    pub fn dim(&self) -> usize {
        self.d
    }
}

/// Level hypervectors: level 0 is random; each subsequent level flips a fresh block of
/// bits, so adjacent levels are similar and the extremes are (near) orthogonal.
fn make_levels(d: usize, n_levels: usize, rng: &mut Rng) -> Vec<Hypervector> {
    let mut order: Vec<usize> = (0..d).collect();
    for k in 0..d {
        let j = k + (rng.next_u64() as usize) % (d - k);
        order.swap(k, j);
    }
    let mut current = Hypervector::random(d, rng);
    let mut out = Vec::with_capacity(n_levels + 1);
    out.push(current.clone());
    let step = (d / n_levels).max(1);
    for l in 0..n_levels {
        let start = (l * step).min(d);
        let end = ((l + 1) * step).min(d);
        for &pos in &order[start..end] {
            current.flip(pos);
        }
        out.push(current.clone());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rng;

    #[test]
    fn similar_inputs_encode_similarly() {
        let mut rng = Rng::new(20);
        let enc = LevelEncoder::new(10_000, 4, 0.0, 1.0, 20, &mut rng);
        let a = enc.encode(&[0.10, 0.50, 0.90, 0.30]);
        let near = enc.encode(&[0.12, 0.52, 0.88, 0.31]); // tiny change
        let far = enc.encode(&[0.90, 0.10, 0.20, 0.80]); // very different
        assert!(a.similarity(&near) > a.similarity(&far));
        assert!(a.similarity(&near) > 0.5);
    }
}
