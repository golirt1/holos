//! Analogical reasoning ("what is the dollar of Mexico?", Kanerva 2010) with the MAP
//! model — reasoning falls out of pure vector algebra, no rule was ever programmed.
//!
//! Run with:  cargo run --release --example dolar_de_mexico

use holos_core::map::{bundle, MapVector};
use holos_core::Rng;

fn main() {
    let d = 10_000;
    let mut rng = Rng::new(1);

    // Roles
    let pais = MapVector::random(d, &mut rng);
    let capital = MapVector::random(d, &mut rng);
    let moneda = MapVector::random(d, &mut rng);
    // Values
    let usa = MapVector::random(d, &mut rng);
    let washington = MapVector::random(d, &mut rng);
    let dolar = MapVector::random(d, &mut rng);
    let mexico = MapVector::random(d, &mut rng);
    let cdmx = MapVector::random(d, &mut rng);
    let peso = MapVector::random(d, &mut rng);

    // Two country records
    let usa_rec = bundle(&[
        pais.bind(&usa),
        capital.bind(&washington),
        moneda.bind(&dolar),
    ]);
    let mex_rec = bundle(&[pais.bind(&mexico), capital.bind(&cdmx), moneda.bind(&peso)]);

    // Transformation USA <-> Mexico, then ask for the analogue of the dollar.
    let t = usa_rec.bind(&mex_rec);
    let result = dolar.bind(&t);

    let candidates: [(&str, &MapVector); 6] = [
        ("peso", &peso),
        ("dolar", &dolar),
        ("usa", &usa),
        ("mexico", &mexico),
        ("washington", &washington),
        ("cdmx", &cdmx),
    ];
    let mut ranked: Vec<(&str, f64)> = candidates
        .iter()
        .map(|(n, v)| (*n, result.similarity(v)))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Question: what is the 'dollar of Mexico'?");
    println!(
        "Answer:   {} (similarity {:+.3})\n",
        ranked[0].0, ranked[0].1
    );
    println!("Ranking:");
    for (n, s) in &ranked {
        println!("  {n:12} {s:+.3}");
    }
}
