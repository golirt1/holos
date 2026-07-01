//! Killer feature demo: associative memory + cleanup.
//!
//! Build a memory of many random symbols, corrupt one heavily, and recover it by
//! nearest-neighbor search. Then time serial vs multi-threaded cleanup.
//!
//! Run with:  cargo run --release --example memoria

use holos_core::{Hypervector, ItemMemory, Rng};
use std::time::Instant;

fn main() {
    let d = 10_000;
    let n = 50_000;
    let mut rng = Rng::new(7);

    let mut mem = ItemMemory::new(d);
    let mut originals = Vec::with_capacity(n);
    for i in 0..n {
        let hv = Hypervector::random(d, &mut rng);
        mem.add(format!("sym{i}"), &hv);
        originals.push(hv);
    }
    println!("Item memory: {n} symbols x {d} dims\n");

    // Robustness: flip 30% of a symbol's bits, then recover it via cleanup.
    let target = 1234usize;
    let noisy = originals[target].add_noise(0.30, &mut rng);
    let (name, sim) = mem.cleanup(&noisy).unwrap();
    println!("Query = sym{target} with 30% of bits flipped");
    println!("  cleanup recovered -> '{name}' (similarity {sim:.3})\n");

    // Timing: serial vs multi-threaded cleanup over the whole memory.
    let q = originals[target].add_noise(0.20, &mut rng);
    let reps = 50;

    let t = Instant::now();
    for _ in 0..reps {
        std::hint::black_box(mem.nearest(&q));
    }
    let serial = t.elapsed().as_secs_f64() / reps as f64;

    let threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);
    let t = Instant::now();
    for _ in 0..reps {
        std::hint::black_box(mem.nearest_parallel(&q, threads));
    }
    let par = t.elapsed().as_secs_f64() / reps as f64;

    println!(
        "cleanup over {n} items: serial {:.3} ms | parallel({threads}) {:.3} ms | {:.1}x speedup",
        serial * 1e3,
        par * 1e3,
        serial / par
    );
}
