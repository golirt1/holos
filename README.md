# HOLOS

A from-scratch, CPU-first engine for **Hyperdimensional Computing / Vector Symbolic
Architectures (HDC/VSA)**.

HDC is a "sleeping beauty" of computer science: a computing paradigm theorized in the
1980s–2000s (Kanerva, Plate, Gayler) that represents *everything* — data, memory, whole
data structures — as high-dimensional vectors (~10,000-D) and computes with three simple,
cheap operations. It is **efficient, noise-robust, interpretable, learns from few examples,
and runs great on a CPU (no GPU required)**.

**Thesis:** HDC's barrier is engineering, not theory. HOLOS aims to (1) build a fast native
(Rust/C++) HDC engine, and (2) attack the field's central open problem — *encoding* (how to
turn raw data into good hypervectors).

## Status

Research phase. Currently: Python reference prototypes (for learning and as ground truth for
the future native engine). Full technical design and roadmap:
[HOLOS_investigacion_tecnica.md](HOLOS_investigacion_tecnica.md).

## What's here

`prototipo_py/` — a small NumPy reference implementation and demos:

| File | What it shows |
|---|---|
| `hdc.py` | Core MAP model: `bind` / `bundle` / `permute` / `similarity` / item memory |
| `01_cuasi_ortogonalidad.py` | Quasi-orthogonality: random hypervectors are ~perpendicular; graceful noise degradation |
| `02_record_y_consulta.py` | A whole key–value record stored in **one** hypervector, queried by algebra |
| `03_dolar_de_mexico.py` | Kanerva's analogical reasoning ("dollar of Mexico" → peso) |
| `04_clasificador.py` | HDC classifier on handwritten digits — **training = summing vectors** |

## Results (reproducible)

- **Handwritten digits** (8×8, 10 classes): **90.6% accuracy**, trained in **~1.7 s** by only
  *adding* hypervectors. No GPU, no backpropagation, no epochs.

## Run

```bash
pip install numpy scikit-learn
cd prototipo_py
python 01_cuasi_ortogonalidad.py
python 03_dolar_de_mexico.py
python 04_clasificador.py
```

## Roadmap

See [HOLOS_investigacion_tecnica.md](HOLOS_investigacion_tecnica.md) §15. Next up: the native
Rust/C++ engine (SIMD, bit-packing, fast cleanup), then research on trainable encoders to push
classification accuracy past the current baseline.

## References

- Kanerva (2009), *Hyperdimensional Computing: An Introduction…*
- Plate (HRR), Gayler (VSA / MAP model)
- HDC/VSA survey — arXiv:2111.06077
- [TorchHD](https://github.com/hyperdimensional-computing/torchhd) (reference library)

## License

MIT — see [LICENSE](LICENSE).
