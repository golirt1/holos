//! A one-shot **and adaptive** HDC classifier.
//!
//! Base training is just **counting bits** (a signed vote per class — no gradients, no
//! epochs). Optional *retraining* then nudges the prototypes on mistakes — the standard
//! HDC trick that lifts accuracy several points. Prediction is a nearest-prototype
//! [`crate::ItemMemory`] cleanup.

use crate::bsc::Hypervector;
use crate::memory::ItemMemory;
use std::collections::HashMap;

/// Accumulates a signed per-class vote at each bit position and builds prototypes
/// (a prototype bit is 1 wherever the class's vote is positive).
pub struct Classifier {
    d: usize,
    names: Vec<String>,
    acc: Vec<Vec<i32>>, // per class: signed vote per bit (+1 for a set bit, -1 otherwise)
    index: HashMap<String, usize>,
}

impl Classifier {
    /// Create a classifier for hypervectors of dimension `d`.
    pub fn new(d: usize) -> Self {
        Classifier {
            d,
            names: Vec::new(),
            acc: Vec::new(),
            index: HashMap::new(),
        }
    }

    /// Class id for `class`, creating a new class if needed.
    fn class_id(&mut self, class: &str) -> usize {
        if let Some(&ci) = self.index.get(class) {
            return ci;
        }
        let ci = self.names.len();
        self.names.push(class.to_string());
        self.acc.push(vec![0i32; self.d]);
        self.index.insert(class.to_string(), ci);
        ci
    }

    /// Add one encoded training sample with its class label (single pass).
    pub fn train(&mut self, sample: &Hypervector, class: &str) {
        assert_eq!(sample.dim(), self.d, "dimension mismatch");
        let ci = self.class_id(class);
        for i in 0..self.d {
            self.acc[ci][i] += if sample.get_bit(i) == 1 { 1 } else { -1 };
        }
    }

    /// Train on all samples, then run `epochs` of **adaptive retraining**: for each
    /// misclassified sample, reinforce the correct class and penalize the predicted one.
    pub fn fit(&mut self, samples: &[Hypervector], labels: &[&str], epochs: usize) {
        assert_eq!(
            samples.len(),
            labels.len(),
            "samples/labels length mismatch"
        );
        for k in 0..samples.len() {
            self.train(&samples[k], labels[k]);
        }
        for _ in 0..epochs {
            let model = self.build();
            for k in 0..samples.len() {
                let s = &samples[k];
                let truth = labels[k];
                let pred = model.cleanup(s).unwrap().0;
                if pred != truth {
                    let pc = self.index[pred];
                    let tc = self.index[truth];
                    for i in 0..self.d {
                        let vote = if s.get_bit(i) == 1 { 1 } else { -1 };
                        self.acc[tc][i] += vote;
                        self.acc[pc][i] -= vote;
                    }
                }
            }
        }
    }

    /// Number of distinct classes seen so far.
    pub fn n_classes(&self) -> usize {
        self.names.len()
    }

    /// Finalize the class prototypes into an [`ItemMemory`] for prediction.
    pub fn build(&self) -> ItemMemory {
        let mut mem = ItemMemory::new(self.d);
        for c in 0..self.names.len() {
            let mut proto = Hypervector::zeros(self.d);
            for i in 0..self.d {
                if self.acc[c][i] > 0 {
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
