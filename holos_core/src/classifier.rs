//! A simple one-shot HDC classifier.
//!
//! Training is just **counting bits**: each class prototype is the bit-wise majority of
//! all its (encoded) samples. There is no gradient descent and no epochs. Prediction is a
//! nearest-prototype [`crate::ItemMemory`] cleanup.

use crate::bsc::Hypervector;
use crate::memory::ItemMemory;
use std::collections::HashMap;

/// Accumulates per-class bit statistics and builds majority-vote prototypes.
pub struct Classifier {
    d: usize,
    names: Vec<String>,
    bit_counts: Vec<Vec<u32>>, // per class: count of 1-bits seen at each position
    totals: Vec<usize>,        // samples per class
    index: HashMap<String, usize>,
}

impl Classifier {
    /// Create a classifier for hypervectors of dimension `d`.
    pub fn new(d: usize) -> Self {
        Classifier {
            d,
            names: Vec::new(),
            bit_counts: Vec::new(),
            totals: Vec::new(),
            index: HashMap::new(),
        }
    }

    /// Add one encoded training sample with its class label.
    pub fn train(&mut self, sample: &Hypervector, class: &str) {
        assert_eq!(sample.dim(), self.d, "dimension mismatch");
        let ci = match self.index.get(class) {
            Some(&ci) => ci,
            None => {
                let ci = self.names.len();
                self.names.push(class.to_string());
                self.bit_counts.push(vec![0u32; self.d]);
                self.totals.push(0);
                self.index.insert(class.to_string(), ci);
                ci
            }
        };
        for i in 0..self.d {
            self.bit_counts[ci][i] += sample.get_bit(i) as u32;
        }
        self.totals[ci] += 1;
    }

    /// Number of distinct classes seen so far.
    pub fn n_classes(&self) -> usize {
        self.names.len()
    }

    /// Finalize the class prototypes into an [`ItemMemory`] for prediction.
    /// Each prototype bit is the majority vote across that class's samples.
    pub fn build(&self) -> ItemMemory {
        let mut mem = ItemMemory::new(self.d);
        for c in 0..self.names.len() {
            let mut proto = Hypervector::zeros(self.d);
            let total = self.totals[c];
            for i in 0..self.d {
                if 2 * self.bit_counts[c][i] as usize > total {
                    proto.set_bit(i);
                }
            }
            mem.add(self.names[c].clone(), &proto);
        }
        mem
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LevelEncoder, Rng};

    // Deterministic uniform in [0, 1).
    fn uni(rng: &mut Rng) -> f64 {
        (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    #[test]
    fn classifies_separable_synthetic_data() {
        let d = 10_000;
        let n_features = 16;
        let n_classes = 5;
        let mut rng = Rng::new(100);
        let enc = LevelEncoder::new(d, n_features, 0.0, 1.0, 20, &mut rng);

        // Random class means; samples = mean + small uniform noise.
        let means: Vec<Vec<f64>> = (0..n_classes)
            .map(|_| (0..n_features).map(|_| uni(&mut rng)).collect())
            .collect();

        let mut clf = Classifier::new(d);
        for c in 0..n_classes {
            for _ in 0..50 {
                let s: Vec<f64> = means[c]
                    .iter()
                    .map(|&m| (m + (uni(&mut rng) - 0.5) * 0.2).clamp(0.0, 1.0))
                    .collect();
                clf.train(&enc.encode(&s), &format!("class{c}"));
            }
        }
        let model = clf.build();

        let mut correct = 0;
        let per_class = 50;
        for c in 0..n_classes {
            for _ in 0..per_class {
                let s: Vec<f64> = means[c]
                    .iter()
                    .map(|&m| (m + (uni(&mut rng) - 0.5) * 0.2).clamp(0.0, 1.0))
                    .collect();
                if model.cleanup(&enc.encode(&s)).unwrap().0 == format!("class{c}") {
                    correct += 1;
                }
            }
        }
        let acc = correct as f64 / (n_classes * per_class) as f64;
        assert!(acc > 0.8, "accuracy too low: {acc}");
    }
}
