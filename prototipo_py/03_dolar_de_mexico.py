"""
03 - El famoso "¿cual es el DOLAR DE MEXICO?" (Kanerva, 2010).
     Razonamiento ANALOGICO emergiendo de pura algebra vectorial.
"""
import hdc

mem = hdc.MemoriaItems()
PAIS   = mem.nuevo("PAIS")
MONEDA = mem.nuevo("MONEDA")
usa    = mem.nuevo("usa");    dolar = mem.nuevo("dolar")
mexico = mem.nuevo("mexico"); peso  = mem.nuevo("peso")

# Dos "paises" como registros
USA    = hdc.bundle(hdc.bind(PAIS, usa),    hdc.bind(MONEDA, dolar))
MEXICO = hdc.bundle(hdc.bind(PAIS, mexico), hdc.bind(MONEDA, peso))

# Transformacion que mapea USA <-> MEXICO
T = hdc.bind(USA, MEXICO)

# "Lo que el dolar es para USA, ¿que es para Mexico?"
resultado = hdc.bind(dolar, T)
nombre, sim = mem.cleanup(resultado)

print("Pregunta:  ¿cual es el 'dolar de Mexico'?")
print(f"Respuesta:  {nombre.upper()}   (similitud {sim:+.3f})\n")
print("Ranking de candidatos:")
for n, s in mem.ranking(resultado, top=6):
    marca = "  <--" if n == nombre else ""
    print(f"   {n:10s} {s:+.3f}{marca}")

print("\n=> El sistema dedujo PESO sin que nadie programara esa pregunta. Eso es HDC.")
