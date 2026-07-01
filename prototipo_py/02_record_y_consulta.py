"""
02 - Un REGISTRO entero (pais/capital/moneda) guardado en UN solo hipervector,
     y consultado con pura algebra (sin bases de datos, sin if/else).
"""
import hdc

mem = hdc.MemoriaItems()

# Roles (las "columnas")
PAIS    = mem.nuevo("PAIS")
CAPITAL = mem.nuevo("CAPITAL")
MONEDA  = mem.nuevo("MONEDA")

# Valores
usa        = mem.nuevo("usa")
washington = mem.nuevo("washington")
dolar      = mem.nuevo("dolar")

# Construir el registro:  bundle de (rol X valor)
registro = hdc.bundle(
    hdc.bind(PAIS,    usa),
    hdc.bind(CAPITAL, washington),
    hdc.bind(MONEDA,  dolar),
)
print("Registro {PAIS:usa, CAPITAL:washington, MONEDA:dolar} guardado en 1 hipervector.\n")

# Consultar: para leer el valor de un rol, se ata el registro con ese rol
for rol, ROL in [("MONEDA", MONEDA), ("CAPITAL", CAPITAL), ("PAIS", PAIS)]:
    respuesta = hdc.bind(ROL, registro)
    nombre, sim = mem.cleanup(respuesta)
    print(f"   Query {rol:8s} -> {nombre:12s} (similitud {sim:+.3f})")

print("\n=> Consultamos una estructura de datos entera con UNA multiplicacion de vectores.")
