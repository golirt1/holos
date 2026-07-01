"""
Benchmark del prototipo Python (numpy, modelo MAP) para comparar contra el
motor nativo Rust. Mide throughput de bind y similarity.
"""
import time
import hdc

N = 200_000
a = hdc.aleatorio()
b = hdc.aleatorio()

# --- similarity ---
t = time.perf_counter()
acc = 0.0
for _ in range(N):
    acc += hdc.similitud(a, b)
dt = time.perf_counter() - t
print(f"similarity: {N} ops en {dt:.4f}s  ->  {N / dt:,.0f} ops/seg")

# --- bind ---
t = time.perf_counter()
for _ in range(N):
    c = hdc.bind(a, b)
dt = time.perf_counter() - t
print(f"bind:       {N} ops en {dt:.4f}s  ->  {N / dt:,.0f} ops/seg")
