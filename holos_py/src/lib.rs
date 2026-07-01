//! Python bindings for HOLOS (module `holos`).
//!
//! Exposes the BSC hypervector algebra and the fast associative-memory cleanup so the
//! (Python-based) HDC community can use the native engine as a drop-in speedup.

use holos_core as hc;
use pyo3::prelude::*;

/// A binary hypervector (BSC model).
#[pyclass]
#[derive(Clone)]
struct Hypervector {
    inner: hc::Hypervector,
}

#[pymethods]
impl Hypervector {
    /// A random hypervector of dimension `d` (deterministic given `seed`).
    #[staticmethod]
    fn random(d: usize, seed: u64) -> Self {
        let mut rng = hc::Rng::new(seed);
        Hypervector {
            inner: hc::Hypervector::random(d, &mut rng),
        }
    }

    /// bind (XOR) — associate two hypervectors.
    fn bind(&self, other: &Hypervector) -> Hypervector {
        Hypervector {
            inner: self.inner.bind(&other.inner),
        }
    }

    /// Cosine-style similarity in [-1, 1].
    fn similarity(&self, other: &Hypervector) -> f64 {
        self.inner.similarity(&other.inner)
    }

    /// Dimension.
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn __repr__(&self) -> String {
        format!("Hypervector(d={})", self.inner.dim())
    }
}

/// bundle (majority) — superpose several hypervectors into one similar to all.
#[pyfunction]
fn bundle(hvs: Vec<Hypervector>) -> PyResult<Hypervector> {
    if hvs.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bundle needs at least one hypervector",
        ));
    }
    let inners: Vec<hc::Hypervector> = hvs.into_iter().map(|h| h.inner).collect();
    Ok(Hypervector {
        inner: hc::bundle(&inners),
    })
}

/// Associative memory with fast nearest-neighbor cleanup.
#[pyclass]
struct ItemMemory {
    inner: hc::ItemMemory,
}

#[pymethods]
impl ItemMemory {
    #[new]
    fn new(d: usize) -> Self {
        ItemMemory {
            inner: hc::ItemMemory::new(d),
        }
    }

    /// Store a named hypervector.
    fn add(&mut self, name: String, hv: &Hypervector) {
        self.inner.add(name, &hv.inner);
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Nearest stored symbol to `query` as `(name, similarity)`.
    /// Set `threads > 1` to search in parallel (best for large memories).
    #[pyo3(signature = (query, threads = 1))]
    fn cleanup(&self, query: &Hypervector, threads: usize) -> Option<(String, f64)> {
        let res = if threads > 1 {
            self.inner.nearest_parallel(&query.inner, threads)
        } else {
            self.inner.nearest(&query.inner)
        };
        res.map(|(i, s)| (self.inner.name(i).to_string(), s))
    }
}

#[pymodule]
fn holos(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Hypervector>()?;
    m.add_class::<ItemMemory>()?;
    m.add_function(wrap_pyfunction!(bundle, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
