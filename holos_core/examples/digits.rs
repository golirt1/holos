//! Real-world benchmark: classify handwritten digits (8x8, the sklearn "digits" dataset)
//! with an HDC classifier — one-shot vs adaptive retraining.
//!
//! Run with:  cargo run --release --example digits

use holos_core::{Classifier, Hypervector, ItemMemory, LevelEncoder, Rng};

fn main() {
    // Dataset shipped with the crate: 64 pixel values (0..16) + label, per line.
    let csv = include_str!("data/digits.csv");
    let mut rows: Vec<(Vec<f64>, String)> = Vec::new();
    for line in csv.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let mut nums: Vec<i64> = line.split(',').map(|t| t.trim().parse().unwrap()).collect();
        let label = nums.pop().unwrap().to_string();
        let features: Vec<f64> = nums.iter().map(|&v| v as f64).collect();
        rows.push((features, label));
    }

    // Deterministic shuffle, then a 70/30 train/test split.
    let mut rng = Rng::new(7);
    for k in (1..rows.len()).rev() {
        let j = (rng.next_u64() as usize) % (k + 1);
        rows.swap(k, j);
    }
    let n_train = rows.len() * 7 / 10;
    let (train, test) = rows.split_at(n_train);

    // Encode: 64 features, pixel intensity 0..16 quantized to 16 levels.
    let d = 10_000;
    let enc = LevelEncoder::new(d, 64, 0.0, 16.0, 16, &mut rng);

    let train_hv: Vec<Hypervector> = train.iter().map(|(f, _)| enc.encode(f)).collect();
    let train_labels: Vec<&str> = train.iter().map(|(_, l)| l.as_str()).collect();
    let test_hv: Vec<Hypervector> = test.iter().map(|(f, _)| enc.encode(f)).collect();
    let test_labels: Vec<&str> = test.iter().map(|(_, l)| l.as_str()).collect();

    let accuracy = |model: &ItemMemory| -> f64 {
        let correct = test_hv
            .iter()
            .zip(&test_labels)
            .filter(|(hv, &truth)| model.cleanup(hv).unwrap().0 == truth)
            .count();
        correct as f64 / test_hv.len() as f64
    };

    // One-shot training (single pass).
    let mut clf = Classifier::new(d);
    for (hv, &l) in train_hv.iter().zip(&train_labels) {
        clf.train(hv, l);
    }
    let acc_oneshot = accuracy(&clf.build());

    // Adaptive retraining.
    let mut clf2 = Classifier::new(d);
    clf2.fit(&train_hv, &train_labels, 20);
    let acc_retrained = accuracy(&clf2.build());

    println!(
        "Handwritten digits: {} train / {} test, 10 classes",
        train.len(),
        test.len()
    );
    println!("HDC one-shot      : {:.1}%", 100.0 * acc_oneshot);
    println!("HDC + retraining  : {:.1}%", 100.0 * acc_retrained);
}
