//! Associative memory + fast nearest-neighbor **cleanup** over BSC hypervectors.
//!
//! Cleanup (find the stored symbol most similar to a noisy query) is the hot path
//! of most HDC workloads — classification, decoding, associative recall. This memory
//! stores all hypervectors in one contiguous, cache-friendly buffer and offers both a
//! single-threaded and a multi-threaded search (std threads only, no dependencies).

use crate::bsc::Hypervector;

/// A set of named hypervectors with fast nearest-neighbor search.
pub struct ItemMemory {
    d: usize,
    nw: usize,
    names: Vec<String>,
    data: Vec<u64>, // flat, row-major: item `i` occupies data[i*nw .. (i+1)*nw]
}

impl ItemMemory {
    /// Create an empty memory for hypervectors of dimension `d`.
    pub fn new(d: usize) -> Self {
        ItemMemory {
            d,
            nw: Hypervector::n_words(d),
            names: Vec::new(),
            data: Vec::new(),
        }
    }

    /// Add a named hypervector.
    pub fn add(&mut self, name: impl Into<String>, hv: &Hypervector) {
        assert_eq!(hv.dim(), self.d, "dimension mismatch");
        self.names.push(name.into());
        self.data.extend_from_slice(hv.words());
    }

    /// Number of stored items.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Whether the memory is empty.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Name of item `i`.
    pub fn name(&self, i: usize) -> &str {
        &self.names[i]
    }

    #[inline]
    fn hamming_row(nw: usize, data: &[u64], q: &[u64], i: usize) -> u32 {
        let row = &data[i * nw..(i + 1) * nw];
        let mut h = 0u32;
        for k in 0..nw {
            h += (q[k] ^ row[k]).count_ones();
        }
        h
    }

    #[inline]
    fn sim(&self, h: u32) -> f64 {
        1.0 - 2.0 * h as f64 / self.d as f64
    }

    /// Nearest item to `query` as `(index, similarity)`. Single-threaded, `O(n·d)`.
    pub fn nearest(&self, query: &Hypervector) -> Option<(usize, f64)> {
        if self.is_empty() {
            return None;
        }
        let q = query.words();
        let mut best_i = 0usize;
        let mut best_h = u32::MAX;
        for i in 0..self.len() {
            let h = Self::hamming_row(self.nw, &self.data, q, i);
            if h < best_h {
                best_h = h;
                best_i = i;
            }
        }
        Some((best_i, self.sim(best_h)))
    }

    /// Nearest item, searched across `threads` OS threads (std only, no dependencies).
    /// On a multi-core CPU this gives a near-linear speedup for large memories.
    pub fn nearest_parallel(&self, query: &Hypervector, threads: usize) -> Option<(usize, f64)> {
        if self.is_empty() {
            return None;
        }
        let threads = threads.max(1);
        let q = query.words();
        let n = self.len();
        let chunk = (n + threads - 1) / threads;
        let nw = self.nw;
        let data = &self.data;

        let best = std::thread::scope(|s| {
            let mut handles = Vec::new();
            for t in 0..threads {
                let start = t * chunk;
                let end = ((t + 1) * chunk).min(n);
                if start >= end {
                    break;
                }
                handles.push(s.spawn(move || {
                    let mut bi = start;
                    let mut bh = u32::MAX;
                    for i in start..end {
                        let h = Self::hamming_row(nw, data, q, i);
                        if h < bh {
                            bh = h;
                            bi = i;
                        }
                    }
                    (bi, bh)
                }));
            }
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .min_by_key(|&(_, h)| h)
                .unwrap()
        });
        Some((best.0, self.sim(best.1)))
    }

    /// Best match as `(name, similarity)`.
    pub fn cleanup(&self, query: &Hypervector) -> Option<(&str, f64)> {
        self.nearest(query)
            .map(|(i, s)| (self.names[i].as_str(), s))
    }

    /// Top-`k` matches as `(name, similarity)`, best first.
    pub fn rank(&self, query: &Hypervector, k: usize) -> Vec<(&str, f64)> {
        let q = query.words();
        let mut all: Vec<(usize, u32)> = (0..self.len())
            .map(|i| (i, Self::hamming_row(self.nw, &self.data, q, i)))
            .collect();
        all.sort_by_key(|&(_, h)| h);
        all.into_iter()
            .take(k)
            .map(|(i, h)| (self.names[i].as_str(), self.sim(h)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Hypervector, Rng};

    const D: usize = 10_000;

    #[test]
    fn cleanup_recovers_noisy_symbol() {
        let mut rng = Rng::new(3);
        let mut mem = ItemMemory::new(D);
        let mut items = Vec::new();
        for i in 0..500 {
            let hv = Hypervector::random(D, &mut rng);
            mem.add(format!("s{i}"), &hv);
            items.push(hv);
        }
        let noisy = items[250].add_noise(0.30, &mut rng);
        let (name, sim) = mem.cleanup(&noisy).unwrap();
        assert_eq!(name, "s250");
        assert!(sim > 0.3);
    }

    #[test]
    fn parallel_matches_serial() {
        let mut rng = Rng::new(4);
        let mut mem = ItemMemory::new(D);
        for i in 0..300 {
            let hv = Hypervector::random(D, &mut rng);
            mem.add(format!("s{i}"), &hv);
        }
        let q = Hypervector::random(D, &mut rng);
        assert_eq!(mem.nearest(&q), mem.nearest_parallel(&q, 4));
    }
}
