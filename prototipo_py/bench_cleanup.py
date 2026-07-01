"""
Benchmark HONESTO del cleanup: numpy OPTIMIZADO (bit-packed, np.bitwise_count).
Es la forma rapida de hacerlo en Python. Se compara contra holos_core (Rust):
  cargo run --release --example memoria
"""
import time
import numpy as np

D = 10_000
N = 50_000
NW = (D + 63) // 64

rng = np.random.default_rng(0)

# Memoria empaquetada en bits: N vectores de NW palabras u64.
M = np.frombuffer(rng.bytes(N * NW * 8), dtype=np.uint64).reshape(N, NW).copy()
q = np.frombuffer(rng.bytes(NW * 8), dtype=np.uint64).copy()


def cleanup(M, q):
    # Hamming a cada fila via XOR + popcount vectorizado, luego el minimo.
    return int(np.bitwise_count(M ^ q).sum(axis=1).argmin())


# warmup (compila rutas internas de numpy)
_ = cleanup(M[:200], q)

reps = 50
t = time.perf_counter()
for _ in range(reps):
    idx = cleanup(M, q)
dt = (time.perf_counter() - t) / reps

print(f"numpy (bit-packed, optimizado)")
print(f"  cleanup sobre {N:,} items x {D:,} dims:  {dt * 1e3:.3f} ms/query")
print(f"  (comparar contra holos_core: cargo run --release --example memoria)")
