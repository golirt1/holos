"""
hdc.py - Nucleo de Computacion Hiperdimensional (modelo MAP bipolar)

Un HIPERVECTOR es un vector de dimension D (p.ej. 10.000) con valores {-1, +1}.
Todo el paradigma son TRES operaciones + una memoria asociativa:

    bind   (X)  asociar   -> multiplicacion elemento a elemento
    bundle (+)  agrupar   -> suma
    permute(p)  ordenar   -> desplazamiento circular

Modelo MAP (Multiply-Add-Permute, Gayler): elegido por ser el mas intuitivo.
Propiedad clave: cada vector es su propio inverso bajo bind  (a * a = +1),
asi que "desatar" es volver a atar con la misma clave.
"""

import numpy as np

D = 10_000                        # dimension de los hipervectores
_rng = np.random.default_rng(42)  # semilla fija -> resultados reproducibles


def aleatorio():
    """Un hipervector aleatorio bipolar {-1, +1}."""
    return _rng.choice(np.array([-1, 1], dtype=np.int8), size=D)


def bind(a, b):
    """X Binding: asocia dos hipervectores. Resultado ortogonal a ambos, invertible."""
    return a * b


def bundle(*vs):
    """+ Bundling: superpone varios. Resultado PARECIDO a todos sus ingredientes."""
    return np.sum(np.stack(vs), axis=0)


def permute(a, k=1):
    """p Permutacion: desplazamiento circular de k posiciones (para secuencias)."""
    return np.roll(a, k)


def similitud(a, b):
    """Similitud coseno en [-1, 1].  ~0 = no se parecen ; 1 = identicos."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class MemoriaItems:
    """Guarda simbolos atomicos por nombre y hace 'cleanup' (vecino mas cercano)."""

    def __init__(self):
        self._nombres = []
        self._vectores = []

    def nuevo(self, nombre):
        """Crea un simbolo atomico aleatorio, lo registra y lo devuelve."""
        v = aleatorio()
        self._nombres.append(nombre)
        self._vectores.append(v)
        return v

    def cleanup(self, x):
        """Devuelve (nombre, similitud) del simbolo mas parecido a x."""
        mejor_n, mejor_s = None, -2.0
        for n, v in zip(self._nombres, self._vectores):
            s = similitud(x, v)
            if s > mejor_s:
                mejor_n, mejor_s = n, s
        return mejor_n, mejor_s

    def ranking(self, x, top=4):
        """Lista los 'top' simbolos mas parecidos a x, ordenados."""
        sims = [(n, similitud(x, v)) for n, v in zip(self._nombres, self._vectores)]
        sims.sort(key=lambda t: t[1], reverse=True)
        return sims[:top]
