//! End-to-end: encode numeric features, train an HDC classifier (training = counting
//! bits), save/load the model, and measure accuracy on synthetic data.
//!
//! Run with:  cargo run --release --example clasificador

use holos_core::{Classifier, ItemMemory, LevelEncoder, Rng};

fn uni(rng: &mut Rng) -> f64 {
    (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64
}

fn sample(mean: &[f64], rng: &mut Rng) -> Vec<f64> {
    mean.iter()
        .map(|&m| (m + (uni(rng) - 0.5) * 0.2).clamp(0.0, 1.0))
        .collect()
}

fn main() {
    let d = 10_000;
    let n_features = 16;
    let n_classes = 5;
    let mut rng = Rng::new(2025);

    let enc = LevelEncoder::new(d, n_features, 0.0, 1.0, 20, &mut rng);
    let means: Vec<Vec<f64>> = (0..n_classes)
        .map(|_| (0..n_features).map(|_| uni(&mut rng)).collect())
        .collect();

    // Train: just count bits per class.
    let mut clf = Classifier::new(d);
    for c in 0..n_classes {
        for _ in 0..50 {
            let s = sample(&means[c], &mut rng);
            clf.train(&enc.encode(&s), &format!("class{c}"));
        }
    }
    println!(
        "Trained {} classes by COUNTING BITS (no gradients, no epochs).",
        clf.n_classes()
    );

    // Serialize the model and reload it (deploy-once, use-many).
    let bytes = clf.build().save();
    let model = ItemMemory::load(&bytes).unwrap();
    println!("Model serialized to {} bytes and reloaded OK.", bytes.len());

    // Test accuracy.
    let per_class = 100;
    let mut correct = 0;
    for c in 0..n_classes {
        for _ in 0..per_class {
            let s = sample(&means[c], &mut rng);
            if model.cleanup(&enc.encode(&s)).unwrap().0 == format!("class{c}") {
                correct += 1;
            }
        }
    }
    let total = n_classes * per_class;
    println!(
        "Accuracy on {} test samples: {:.1}%",
        total,
        100.0 * correct as f64 / total as f64
    );
}
