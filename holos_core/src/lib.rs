//! HOLOS — motor de Computacion Hiperdimensional (HDC / VSA), nativo y sin dependencias.
//!
//! Modelo BSC (Binary Spatter Codes): hipervectores binarios empaquetados en bits.
//!   - bind (X)  = XOR bit a bit           -> asociar (invertible, ortogonal)
//!   - bundle(+) = mayoria bit a bit       -> agrupar (parecido a todos)
//!   - similitud = Hamming/popcount        -> equivalente a coseno bipolar en [-1, 1]
//!
//! El empaquetado en bits hace que las operaciones sean XOR + popcount sobre
//! palabras de 64 bits: ordenes de magnitud mas rapido que un byte/float por bit.

/// PRNG SplitMix64 — determinista, rapido, sin dependencias externas.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng { state: seed }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Hipervector binario (modelo BSC) empaquetado en palabras de 64 bits.
/// Los bits de relleno de la ultima palabra (mas alla de `d`) se mantienen en 0.
#[derive(Clone, Debug)]
pub struct Hypervector {
    d: usize,
    words: Vec<u64>,
}

impl Hypervector {
    #[inline]
    fn n_words(d: usize) -> usize {
        (d + 63) / 64
    }

    /// Mascara de la ultima palabra: 1 en los bits validos, 0 en el relleno.
    #[inline]
    fn tail_mask(d: usize) -> u64 {
        let rem = d % 64;
        if rem == 0 {
            u64::MAX
        } else {
            (1u64 << rem) - 1
        }
    }

    /// Hipervector de ceros de dimension `d`.
    pub fn zeros(d: usize) -> Self {
        Hypervector {
            d,
            words: vec![0u64; Self::n_words(d)],
        }
    }

    /// Hipervector aleatorio (~50% de bits a 1). Bits de relleno = 0.
    pub fn random(d: usize, rng: &mut Rng) -> Self {
        let nw = Self::n_words(d);
        let mut words: Vec<u64> = (0..nw).map(|_| rng.next_u64()).collect();
        if let Some(last) = words.last_mut() {
            *last &= Self::tail_mask(d);
        }
        Hypervector { d, words }
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.d
    }

    /// X Binding: XOR bit a bit. Resultado ortogonal a ambos operandos, invertible.
    pub fn bind(&self, other: &Hypervector) -> Hypervector {
        assert_eq!(self.d, other.d, "dimensiones distintas");
        let words = self
            .words
            .iter()
            .zip(&other.words)
            .map(|(a, b)| a ^ b)
            .collect();
        Hypervector { d: self.d, words }
    }

    /// Distancia de Hamming (numero de bits distintos), via popcount hardware.
    pub fn hamming(&self, other: &Hypervector) -> u32 {
        self.words
            .iter()
            .zip(&other.words)
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }

    /// Similitud tipo coseno bipolar en [-1, 1]:  1 - 2*(hamming/d).
    /// ~0 = no se parecen ; 1 = identicos ; -1 = opuestos.
    pub fn similarity(&self, other: &Hypervector) -> f64 {
        let h = self.hamming(other) as f64;
        1.0 - 2.0 * h / self.d as f64
    }

    #[inline]
    fn get_bit(&self, i: usize) -> u64 {
        (self.words[i / 64] >> (i % 64)) & 1
    }
}

/// + Bundling: mayoria bit a bit sobre varios hipervectores.
/// El resultado se parece a todos sus miembros. Empates -> 0.
pub fn bundle(hvs: &[Hypervector]) -> Hypervector {
    assert!(!hvs.is_empty(), "bundle necesita al menos un hipervector");
    let d = hvs[0].d;
    let n = hvs.len();
    let mut out = Hypervector::zeros(d);
    for i in 0..d {
        let mut count = 0usize;
        for hv in hvs {
            count += hv.get_bit(i) as usize;
        }
        if 2 * count > n {
            out.words[i / 64] |= 1u64 << (i % 64);
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────
//  Tests: verifican las MISMAS propiedades que el prototipo Python.
// ─────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    const D: usize = 10_000;

    #[test]
    fn aleatorios_son_cuasi_ortogonales() {
        let mut rng = Rng::new(42);
        let n = 300;
        let media: f64 = (0..n)
            .map(|_| {
                let a = Hypervector::random(D, &mut rng);
                let b = Hypervector::random(D, &mut rng);
                a.similarity(&b)
            })
            .sum::<f64>()
            / n as f64;
        assert!(media.abs() < 0.05, "media de similitud aleatoria = {media}");
    }

    #[test]
    fn similitud_consigo_mismo_es_uno() {
        let mut rng = Rng::new(1);
        let a = Hypervector::random(D, &mut rng);
        assert!((a.similarity(&a) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn bind_es_invertible_y_ortogonal() {
        let mut rng = Rng::new(7);
        let a = Hypervector::random(D, &mut rng);
        let b = Hypervector::random(D, &mut rng);
        let c = a.bind(&b);
        // c no se parece ni a `a` ni a `b`
        assert!(c.similarity(&a).abs() < 0.05);
        assert!(c.similarity(&b).abs() < 0.05);
        // desatar: (a X b) X b == a
        let recuperado = c.bind(&b);
        assert!((recuperado.similarity(&a) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn bundle_se_parece_a_sus_miembros() {
        let mut rng = Rng::new(9);
        let a = Hypervector::random(D, &mut rng);
        let b = Hypervector::random(D, &mut rng);
        let c = Hypervector::random(D, &mut rng);
        let s = bundle(&[a.clone(), b.clone(), c.clone()]);
        assert!(s.similarity(&a) > 0.3);
        assert!(s.similarity(&b) > 0.3);
        assert!(s.similarity(&c) > 0.3);
    }
}
