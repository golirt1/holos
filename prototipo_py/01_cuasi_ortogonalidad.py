"""
01 - Ver la CUASI-ORTOGONALIDAD con tus propios ojos.

En dimension alta, dos hipervectores aleatorios son casi perpendiculares
(similitud ~ 0). Y un hipervector aguanta MUCHO ruido antes de volverse
irreconocible. Esta es la base de por que HDC funciona.
"""
import numpy as np
import hdc

# --- 1. Similitud entre pares ALEATORIOS ---
N = 1000
sims = np.array([hdc.similitud(hdc.aleatorio(), hdc.aleatorio()) for _ in range(N)])

print(f"Dimension D = {hdc.D:,}")
print(f"\nSimilitud entre {N} pares de hipervectores ALEATORIOS:")
print(f"   media    = {sims.mean():+.4f}   <- esperado ~0 (cuasi-ortogonales)")
print(f"   std      = {sims.std():.4f}")
print(f"   min/max  = {sims.min():+.4f} / {sims.max():+.4f}")

# --- 2. Robustez: un vector vs versiones corrompidas de si mismo ---
a = hdc.aleatorio()
print(f"\nSimilitud de un hipervector consigo mismo y con copias corrompidas:")
print(f"    0% bits volteados -> {hdc.similitud(a, a):+.4f}  (identico)")
rng = np.random.default_rng(0)
for pct in (0.10, 0.25, 0.40, 0.49):
    corr = a.copy()
    idx = rng.choice(hdc.D, size=int(hdc.D * pct), replace=False)
    corr[idx] *= -1
    print(f"   {int(pct*100):>2}% bits volteados -> {hdc.similitud(a, corr):+.4f}")

print("\n=> Aleatorios ~0, identico =1, y degrada SUAVE con el ruido. Esa es la magia.")
