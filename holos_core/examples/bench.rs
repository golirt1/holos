//! Benchmark simple: throughput de `bind` y `similarity` en el motor nativo.
//! Ejecutar con:  cargo run --release --example bench

use holos_core::{Hypervector, Rng};
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let d = 10_000;
    let n = 200_000;
    let mut rng = Rng::new(123);
    let a = Hypervector::random(d, &mut rng);
    let b = Hypervector::random(d, &mut rng);

    // --- similarity (la operacion del cleanup) ---
    let t = Instant::now();
    let mut acc = 0.0f64;
    for _ in 0..n {
        acc += black_box(&a).similarity(black_box(&b));
    }
    let dt = t.elapsed().as_secs_f64();
    black_box(acc);
    println!(
        "similarity: {n} ops en {dt:.4}s  ->  {:.0} ops/seg",
        n as f64 / dt
    );

    // --- bind ---
    let t = Instant::now();
    for _ in 0..n {
        black_box(black_box(&a).bind(black_box(&b)));
    }
    let dt = t.elapsed().as_secs_f64();
    println!(
        "bind:       {n} ops en {dt:.4}s  ->  {:.0} ops/seg",
        n as f64 / dt
    );
}
