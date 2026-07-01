"""
04 - Un CLASIFICADOR HDC real sobre datos reales (digitos escritos a mano 8x8).

Lo asombroso: "entrenar" = SUMAR hipervectores. Sin backpropagation, sin GPU,
sin epocas. Un prototipo por clase = el bundle de todos sus ejemplos.
"""
import time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import hdc

rng = np.random.default_rng(7)

NUM_PIXELES = 64          # imagen 8x8
NUM_NIVELES = 17          # intensidad de pixel 0..16

# --- Hipervectores de POSICION (uno por pixel) ---
POS = np.stack([hdc.aleatorio() for _ in range(NUM_PIXELES)]).astype(np.int8)  # (64, D)

# --- Hipervectores de NIVEL (intensidad) ---
# Niveles cercanos = parecidos; extremos = ortogonales (voltear bits progresivamente).
def crear_niveles(n):
    orden = rng.permutation(hdc.D)
    paso = hdc.D // (n - 1)
    actual = hdc.aleatorio().copy()
    niveles = [actual.copy()]
    for i in range(1, n):
        idx = orden[(i - 1) * paso: i * paso]
        actual = actual.copy()
        actual[idx] *= -1
        niveles.append(actual.copy())
    return np.stack(niveles).astype(np.int8)  # (n, D)

NIVEL = crear_niveles(NUM_NIVELES)

def encode(img):
    """imagen (64 enteros 0..16) -> un hipervector (vectorizado)."""
    lv = NIVEL[img.astype(int)]          # (64, D)
    return (POS * lv).sum(axis=0)         # (D,)  suma de los 64 pares atados

# --- Datos (incluidos en sklearn, sin descargas) ---
digits = load_digits()
X = digits.images.reshape(len(digits.images), -1)   # (n, 64), valores 0..16
y = digits.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# --- ENTRENAR = sumar hipervectores por clase ---
t0 = time.time()
prototipos = np.zeros((10, hdc.D))
for img, c in zip(Xtr, ytr):
    prototipos[c] += encode(img)
t_train = time.time() - t0

# --- PREDECIR = clase cuyo prototipo es mas parecido (coseno) ---
def predecir(img):
    h = encode(img)
    return max(range(10), key=lambda c: hdc.similitud(h, prototipos[c]))

t0 = time.time()
pred = np.array([predecir(img) for img in Xte])
t_test = time.time() - t0

acc = (pred == yte).mean()
print(f"Dataset: digitos 8x8  |  {len(Xtr)} train / {len(Xte)} test  |  10 clases")
print(f"Dimension HDC: {hdc.D:,}")
print(f"\nEntrenamiento (SOLO sumando vectores): {t_train:.2f}s")
print(f"Prediccion de {len(Xte)} muestras:        {t_test:.2f}s")
print(f"\n>>> PRECISION: {acc * 100:.1f}% <<<")
print("\n=> 'Entrenar' fue literalmente sumar hipervectores. Sin GPU, sin backprop.")
