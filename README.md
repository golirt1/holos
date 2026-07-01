# HOLOS

**A fast, dependency-free Hyperdimensional Computing (HDC / VSA) engine** — native Rust core,
Python bindings, runs on a CPU.

HDC is a "sleeping beauty" of computer science: a paradigm theorized in the 1980s–2000s
(Kanerva, Plate, Gayler) that represents *everything* — data, memory, whole data structures —
as high-dimensional vectors (~10,000-D) and computes with three cheap algebraic operations
plus an associative memory. It is **efficient, noise-robust, interpretable, learns from few
examples, and needs no GPU**.

> **Thesis:** HDC's barrier is engineering, not theory. HOLOS turns the dormant theory into a
> fast, usable, embeddable engine — and is a place to attack the field's central open problem:
> *encoding* (how to turn raw data into good hypervectors).

---

## Why HOLOS (honest positioning)

The reference ecosystem (e.g. TorchHD) is Python on top of PyTorch: great for research, but a
heavy dependency, not embeddable, and not tuned for pure-CPU / edge use. HOLOS fills that gap:

- **Fast** — BSC hypervectors are bit-packed into `u64`; `bind`/`similarity`/`cleanup` become
  XOR + hardware popcount. **5–14× faster cleanup than optimized NumPy** (see below).
- **Zero dependencies** — the core crate pulls in *nothing*. Small, auditable, embeddable.
- **Parallel** — multi-threaded cleanup using the standard library only.
- **Usable from Python** — PyO3 bindings, so the existing HDC community gets a drop-in speedup.

**What HOLOS is not:** HDC is not a drop-in replacement for deep learning on hard perceptual
tasks — it trades some accuracy for efficiency, robustness, few-shot learning and
interpretability. Its sweet spot is edge / low-power / robust / interpretable computing.

## Benchmarks

Measured on a 12-thread CPU. Cleanup = nearest-neighbor over an item memory (the hot path of most
HDC workloads).

| Task (D = 10,000) | Optimized NumPy | **HOLOS (serial)** | **HOLOS (parallel)** |
|---|---|---|---|
| Cleanup over 50,000 items | 53.98 ms/query | **9.79 ms** (5.5×) | **3.90 ms** (13.8×) |

Reproduce: `cargo run --release --example memoria` and `python prototipo_py/bench_cleanup.py`.
*(NumPy here is the optimized bit-packed approach with `np.bitwise_count`, not a naive baseline.
A direct TorchHD comparison is future work — TorchHD requires PyTorch, the dependency HOLOS avoids.)*

## Install

**Rust:**
```bash
cargo add holos_core   # once published; meanwhile use a git or path dependency
```

**Python:**
```bash
pip install holos       # once published on PyPI
# or build from source:
pip install maturin && maturin build --release --manifest-path holos_py/Cargo.toml
```

## Usage

**Rust:**
```rust
use holos_core::{Hypervector, Rng, ItemMemory};

let mut rng = Rng::new(42);
let d = 10_000;

// bind (associate) and unbind
let role = Hypervector::random(d, &mut rng);
let value = Hypervector::random(d, &mut rng);
let bound = role.bind(&value);
assert!(bound.bind(&role).similarity(&value) > 0.99);

// associative memory + fast cleanup
let mut mem = ItemMemory::new(d);
mem.add("value", &value);
let (name, sim) = mem.cleanup(&bound.bind(&role)).unwrap(); // -> ("value", ~1.0)
```

**Python:**
```python
import holos

role  = holos.Hypervector.random(10_000, seed=1)
value = holos.Hypervector.random(10_000, seed=2)
bound = role.bind(value)
assert bound.bind(role).similarity(value) > 0.99

mem = holos.ItemMemory(10_000)
mem.add("value", value)
name, sim = mem.cleanup(bound.bind(role), threads=8)   # ("value", ~1.0)
```

## Repository layout

```
holos/
├── holos_core/          # the engine (Rust, zero dependencies)
│   ├── src/             #   rng · bsc (bit-packed) · memory (cleanup) · map
│   └── examples/        #   memoria (cleanup), dolar_de_mexico (analogy), bench
├── holos_py/            # Python bindings (PyO3 + maturin)
├── prototipo_py/        # NumPy reference prototypes (ground truth + benchmarks)
├── .github/workflows/   # CI: fmt/clippy/test + build wheels (linux/mac/win) + PyPI publish
├── HOLOS_investigacion_tecnica.md   # full technical design & roadmap
└── PUBLISHING.md        # how to publish to crates.io / PyPI
```

## Status & roadmap

Working: native BSC + MAP algebra, associative-memory cleanup (serial + parallel), Python
bindings, CI, benchmarks. Next (see [design doc](HOLOS_investigacion_tecnica.md) §15):

- Trainable / adaptive **encoders** — the field's central open problem, and the path to closing
  the accuracy gap with deep learning.
- Batched / streaming APIs and a persistent thread pool for cleanup.
- Direct TorchHD benchmark; more VSA models (FHRR).

## References

- Kanerva (2009), *Hyperdimensional Computing: An Introduction…*; Kanerva (2010), *"…dollar of Mexico?"*
- Plate (HRR/FHRR); Gayler (VSA / MAP); Smolensky (Tensor Product Representations)
- HDC/VSA survey — [arXiv:2111.06077](https://arxiv.org/abs/2111.06077)
- [TorchHD](https://github.com/hyperdimensional-computing/torchhd) (reference library)

## License

MIT — see [LICENSE](LICENSE).
