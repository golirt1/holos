# holos

Python bindings for **HOLOS** — a fast, dependency-free
Hyperdimensional Computing / Vector Symbolic Architectures (HDC/VSA) engine,
implemented in Rust.

```python
import holos

# Two random hypervectors
role  = holos.Hypervector.random(10_000, seed=1)
value = holos.Hypervector.random(10_000, seed=2)

# bind (associate) and unbind
bound = role.bind(value)
assert bound.bind(role).similarity(value) > 0.99

# Associative memory + fast cleanup
mem = holos.ItemMemory(10_000)
for i in range(50_000):
    mem.add(f"sym{i}", holos.Hypervector.random(10_000, seed=i))
name, sim = mem.cleanup(role, threads=8)   # nearest stored symbol
```

Why HOLOS: the bit-packed BSC engine makes `bind`/`similarity`/`cleanup` run as
XOR + hardware popcount — **5–14× faster cleanup than optimized NumPy**, with **zero
dependencies** and an embeddable native core (unlike PyTorch-based libraries).

See the [project repository](https://github.com/USER/holos) for the full design.

License: MIT.
