# HOLOS — Motor de Computación Hiperdimensional (HDC / VSA)
## Documento de Investigación Técnica Profunda

> **Estado:** Fase de Investigación
> **Versión:** 1.0
> **Fecha:** 2026-07-01
> **Codename:** HOLOS *(placeholder — renombrable)*
> **Clasificación:** Documento interno de arquitectura
>
> **Tesis del proyecto:** La Computación Hiperdimensional es una "bella durmiente" — un paradigma
> de cómputo teorizado en los años 80-90 que quedó dormido bajo la sombra del deep learning, y que
> hoy podría despertar. Es **ultra-eficiente, robusta, interpretable, corre en CPU**, y el campo
> está tan poco poblado que **una persona sola puede dejar marca real.**

---

## Tabla de Contenidos

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [La Bella Durmiente — Por Qué HDC y Por Qué Ahora](#2-la-bella-durmiente--por-qué-hdc-y-por-qué-ahora)
3. [La Idea Central — Computar con Vectores Gigantes](#3-la-idea-central--computar-con-vectores-gigantes)
4. [El Fenómeno Clave — Cuasi-Ortogonalidad](#4-el-fenómeno-clave--cuasi-ortogonalidad)
5. [Las Tres Operaciones del Álgebra Hiperdimensional](#5-las-tres-operaciones-del-álgebra-hiperdimensional)
6. [Memoria de Ítems y "Cleanup"](#6-memoria-de-ítems-y-cleanup)
7. [Los Modelos VSA (BSC, MAP, HRR, FHRR, Sparse)](#7-los-modelos-vsa-bsc-map-hrr-fhrr-sparse)
8. [Ejemplos Canónicos — Ver el Paradigma en Acción](#8-ejemplos-canónicos--ver-el-paradigma-en-acción)
9. [Estado del Arte — Librerías, Aplicaciones, Hardware](#9-estado-del-arte--librerías-aplicaciones-hardware)
10. [Honestidad — Limitaciones y Problemas Abiertos](#10-honestidad--limitaciones-y-problemas-abiertos)
11. [La Tesis de HOLOS — Dónde Aportamos](#11-la-tesis-de-holos--dónde-aportamos)
12. [Elección de Lenguaje y Stack](#12-elección-de-lenguaje-y-stack)
13. [Arquitectura Técnica de HOLOS](#13-arquitectura-técnica-de-holos)
14. [Estructura del Repositorio](#14-estructura-del-repositorio)
15. [Roadmap de Implementación](#15-roadmap-de-implementación)
16. [Desafíos Técnicos Críticos](#16-desafíos-técnicos-críticos)
17. [Métricas de Éxito](#17-métricas-de-éxito)
18. [El Experimento que Valida o Refuta el Proyecto](#18-el-experimento-que-valida-o-refuta-el-proyecto)
19. [Referencias y Recursos Clave](#19-referencias-y-recursos-clave)

---

## 1. Resumen Ejecutivo

La **Computación Hiperdimensional** (HDC), también llamada **Arquitecturas Vectoriales Simbólicas**
(VSA, *Vector Symbolic Architectures*), es un paradigma de cómputo radicalmente distinto al que
domina hoy. En vez de manipular números uno por uno, o de entrenar millones de pesos como el deep
learning, **representa toda la información — conceptos, datos, memoria, estructuras — como vectores
enormes** (típicamente 10.000 dimensiones) y computa con ellos usando un álgebra de tres operaciones
simples.

**Lo que lo hace extraordinario:**
- **Corre en CPU, rapidísimo y con poca energía** — las operaciones son sumas, multiplicaciones
  elemento a elemento, o XOR de bits. No necesita GPU.
- **Es robusto al ruido y a fallos** — puedes corromper el 30% de un hipervector y aún recuperar la
  información. La representación está distribuida por todo el vector, no localizada.
- **Aprende con uno o pocos ejemplos** (*one-shot / few-shot*), a diferencia del deep learning que
  necesita montañas de datos.
- **Es transparente y algebraico** — puedes *razonar simbólicamente* sobre lo que representa, no es
  una caja negra.

**El insight histórico (la tesis del proyecto):**
> HDC fue teorizada por Kanerva, Plate y Gayler entre 1988 y 2003. Llegó justo cuando las redes
> neuronales empezaban su ascenso imparable, y quedó **dormida en la academia** durante décadas.
> Igual que las redes neuronales durmieron desde 1943 hasta 2012 esperando su momento, HDC podría
> estar esperando el suyo — impulsada ahora por la necesidad de IA eficiente en el borde (edge),
> por los límites energéticos del cómputo, y por la búsqueda de alternativas interpretables al
> deep learning.

**Qué es HOLOS:** un motor de HDC/VSA construido desde cero en un lenguaje de sistemas, optimizado
para CPU, que implementa el álgebra completa (los modelos, las operaciones, la memoria asociativa) y
ataca **el problema abierto más importante del campo: la codificación** (cómo convertir datos crudos
en hipervectores útiles). No competimos con la teoría — **la llevamos de la teoría dormida a
ingeniería viva y usable.**

**Veredicto técnico:** El paradigma es real, tiene 35 años de fundamento matemático sólido, un
resurgimiento activo (2024-2026), y — crucialmente — es un campo **poco poblado** donde el trabajo
de ingeniería serio escasea. Es el terreno ideal para que un ingeniero de sistemas con voluntad deje
una huella real, todo corriendo en su propia máquina.

---

## 2. La Bella Durmiente — Por Qué HDC y Por Qué Ahora

### 2.1 El patrón de las ideas prematuras

La historia de la computación está llena de ideas teorizadas décadas antes de que el mundo pudiera
usarlas:

| Idea | Teorizada | Despertó | Espera |
|---|---|---|---|
| **Redes neuronales** | 1943 (McCulloch-Pitts), 1958 (Perceptrón) | ~2012 (deep learning) | ~70 años |
| **Lógica lineal** (Girard) | 1987 | ~2010s (ownership de Rust) | ~25 años |
| **Computación hiperdimensional** | 1988-2003 (Kanerva, Plate, Gayler) | **¿ahora?** | ~35 años |

### 2.2 Por qué HDC se durmió
- Nació en los años 88-2003, justo cuando las redes neuronales empezaban a demostrar resultados
  espectaculares. Toda la atención, el talento y el dinero se fueron al deep learning.
- HDC no ganaba en las tareas "de moda" (reconocer imágenes), donde las redes brillaban.
- Se quedó como una curiosidad elegante de la ciencia cognitiva y unos pocos laboratorios (sobre
  todo el Redwood Center de Berkeley).

### 2.3 Por qué podría despertar AHORA
- **El fin del escalar barato:** el deep learning es carísimo en energía y cómputo. HDC es
  ridículamente barato. En cargas móviles, **mover datos hacia/desde memoria consume el 62% de la
  energía** — y HDC minimiza justamente eso.
- **Edge AI / TinyML:** hay una necesidad enorme de IA que corra en dispositivos pequeños (sensores,
  wearables, implantes médicos) sin nube ni GPU. HDC encaja perfecto.
- **Hardware emergente:** computación en memoria (*in-memory computing*), neuromórfico, y procesadores
  RISC-V dedicados a HDC están apareciendo (2024-2025). El paradigma vuelve a ser relevante.
- **La sed de interpretabilidad:** el mundo desconfía de las cajas negras. HDC es algebraico y
  transparente.
- **Campo poco poblado:** mientras millones trabajan en deep learning, HDC tiene una comunidad
  pequeña. Eso significa **fruta al alcance de la mano** para quien haga ingeniería seria.

### 2.4 La honestidad desde el principio
HDC **no** es un reemplazo probado del deep learning — hoy pierde en precisión en tareas perceptuales
difíciles (§10). Es una apuesta a un paradigma con ventajas únicas (eficiencia, robustez,
interpretabilidad, few-shot) cuyo momento *podría* haber llegado. Es exactamente eso: una **bella
durmiente**, no una certeza. Ese es el tipo de riesgo que buscabas.

---

## 3. La Idea Central — Computar con Vectores Gigantes

### 3.1 El cambio de mentalidad

```
Computación clásica:          El significado vive en UN lugar (una variable, un byte, una neurona).
                              Si ese lugar se corrompe, pierdes la información.

Computación hiperdimensional: El significado vive DISTRIBUIDO en un vector de 10.000 números.
                              Ningún número individual significa nada. El concepto emerge del
                              patrón completo. Corrompe muchos y aún lo recuperas.
```

Un **hipervector** es un vector de dimensión muy alta — típicamente **D = 10.000**. Sus componentes,
según el modelo, pueden ser bits `{0,1}`, valores bipolares `{-1,+1}`, números reales, o números
complejos.

### 3.2 Todo es un hipervector
La idea unificadora: **absolutamente todo** se representa como un hipervector del mismo tamaño:
- Un símbolo atómico ("rojo", "perro", "Q") → un hipervector aleatorio.
- Un dato (un píxel, una señal, una letra) → un hipervector.
- Una estructura compleja (una frase, un registro, un grafo, una secuencia) → **también** un
  hipervector del mismo tamaño, construido combinando otros.

Esa es la magia: una estructura de datos entera (un record `{país: USA, moneda: dólar}`) se comprime
en un solo vector de 10.000 números, y sigues pudiendo consultarla.

### 3.3 De dónde salen los hipervectores "atómicos"
Los símbolos base se generan **aleatoriamente**. Suena raro, pero es la clave: en dimensión alta, dos
vectores aleatorios son casi siempre **cuasi-ortogonales** (§4) — es decir, "no se parecen en nada".
Esto le da al sistema un vocabulario prácticamente infinito de símbolos distinguibles.

---

## 4. El Fenómeno Clave — Cuasi-Ortogonalidad

Este es el fenómeno matemático que hace que todo funcione. Merece entenderse bien.

### 4.1 La rareza de los espacios de alta dimensión

En 2D o 3D, dos vectores aleatorios pueden apuntar en direcciones parecidas con facilidad. Pero a
medida que sube la dimensión, ocurre algo contraintuitivo:

> En dimensión muy alta (D = 10.000), **dos vectores aleatorios son casi siempre casi
> perpendiculares** entre sí. Su similitud (coseno) es prácticamente 0.

```
Similitud de dos hipervectores aleatorios (D=10.000):
   ≈ 0.0  ± 0.01     → "no tienen NADA que ver"

Similitud de un hipervector consigo mismo:
   = 1.0             → "idéntico"

Similitud de un hipervector con una versión con 30% de bits corrompidos:
   ≈ 0.4             → "aún claramente reconocible"
```

### 4.2 Por qué esto lo cambia todo
- **Capacidad casi infinita:** puedes generar millones de símbolos aleatorios y todos serán
  distinguibles entre sí (todos cuasi-ortogonales). El "vocabulario" es enorme.
- **Robustez:** como la información está esparcida en 10.000 componentes, corromper muchos no destruye
  el significado. La similitud degrada suave, no se rompe de golpe.
- **Separabilidad:** distinguir "esto es A" de "esto es B" es trivial, porque A y B están lejísimos en
  el espacio.

Esta es la razón por la que HDC es **tan robusto al ruido y a fallos de hardware** — una propiedad que
el deep learning no tiene de forma natural, y que es oro para dispositivos edge y entornos hostiles.

---

## 5. Las Tres Operaciones del Álgebra Hiperdimensional

Todo el poder de HDC viene de **tres operaciones**. Con ellas construyes cualquier estructura y
cualquier consulta. Son baratísimas (elemento a elemento), lo que hace todo el paradigma eficiente en
CPU.

### 5.1 Binding (⊗) — "asociar / atar"
Combina dos hipervectores en uno **nuevo, distinto a ambos**. Sirve para **asociar** cosas: una clave
con un valor, un rol con su relleno.

```
C = A ⊗ B
Propiedades:
  • C NO se parece ni a A ni a B (similitud ≈ 0)
  • Es invertible: si conoces A, recuperas B  →  B ≈ A ⊗ C
  • Preserva distancias (distribuye sobre el bundling)
```

**Implementación según modelo:**
- Binario (BSC): **XOR** bit a bit.
- Bipolar (MAP): **multiplicación** elemento a elemento.
- Real (HRR): **convolución circular**.

**Uso típico:** representar un par clave-valor. `pais ⊗ USA` = "el rol PAÍS tiene el valor USA".

### 5.2 Bundling / Superposición (⊕) — "juntar en un conjunto"
Combina varios hipervectores en uno que **se parece a todos ellos**. Sirve para representar
**conjuntos, memoria, superposición** de cosas.

```
S = A ⊕ B ⊕ C
Propiedades:
  • S SÍ se parece a A, a B y a C (similitud > 0 con cada uno)
  • Es como una "media" — todos sus ingredientes siguen ahí, detectables
  • Tiene capacidad limitada: si juntas demasiados, empiezan a interferir (crosstalk)
```

**Implementación:** suma elemento a elemento (con umbral/mayoría para volver a binario/bipolar).

**Uso típico:** representar "el conjunto que contiene A, B y C" en un solo vector.

### 5.3 Permutación (ρ) — "ordenar / proteger"
Reordena los componentes del hipervector (típicamente un **desplazamiento cíclico**). Produce algo
cuasi-ortogonal al original. Sirve para representar **orden y secuencias**, y para "proteger" un
hipervector de mezclarse.

```
ρ(A)  = A desplazado una posición
ρ²(A) = A desplazado dos posiciones
Uso: representar la secuencia [A, B, C] como  ρ²(A) ⊕ ρ(B) ⊕ C
     (la posición queda codificada por cuántas veces permutas)
```

### 5.4 El álgebra completa, en una imagen

```
   BINDING (⊗)          BUNDLING (⊕)           PERMUTACIÓN (ρ)
   asociar               agrupar                 ordenar
   ───────────           ───────────             ───────────
   clave↔valor           conjuntos               secuencias
   distinto a ambos      parecido a todos        distinto, reversible
   XOR / multiplicar     sumar                   desplazar

   Con estas tres operaciones + memoria de ítems (§6),
   representas: registros, listas, árboles, grafos, secuencias,
   estados, y consultas sobre todos ellos. Todo en vectores del mismo tamaño.
```

---

## 6. Memoria de Ítems y "Cleanup"

### 6.1 El problema
Cuando haces operaciones (sobre todo bundling y binding), el resultado acumula "ruido": no es
exactamente ninguno de los hipervectores originales, sino una versión aproximada. Para **decodificar**
—recuperar qué símbolo atómico es lo más parecido— necesitas una memoria.

### 6.2 La memoria de ítems (Item Memory / Cleanup Memory)
Es simplemente una **tabla de todos los hipervectores atómicos conocidos** ("rojo", "perro", "USA"...).
Dado un hipervector ruidoso `x`, la operación de **cleanup** busca el vecino más cercano:

```
cleanup(x) = argmax_{v ∈ memoria}  similitud(x, v)
```

Es una **búsqueda por vecino más cercano** (nearest neighbor). Devuelve el símbolo "limpio" más
parecido. Es el puente entre el mundo ruidoso del cómputo vectorial y el mundo limpio de los símbolos.

### 6.3 Por qué importa para el rendimiento
La memoria de ítems es donde se va **la mayor parte del coste computacional** en muchas aplicaciones
HDC: comparar `x` contra miles o millones de hipervectores almacenados. Optimizar esta búsqueda
(SIMD, bit-packing, indexado) es un objetivo central de ingeniería — y una de las áreas donde HOLOS
puede brillar (§11).

---

## 7. Los Modelos VSA (BSC, MAP, HRR, FHRR, Sparse)

No existe "un" HDC. Existe una **familia de modelos** que difieren en (a) qué tipo de números usan los
hipervectores y (b) cómo implementan las tres operaciones. Un motor serio debe entender los grandes.

### 7.1 Tabla comparativa

| Modelo | Autor / Año | Componentes | Binding | Bundling | Similitud |
|---|---|---|---|---|---|
| **BSC** (Binary Spatter Codes) | Kanerva, ~1994 | bits `{0,1}` | XOR | mayoría | Hamming |
| **MAP** (Multiply-Add-Permute) | Gayler, ~1998 | bipolar `{-1,+1}` | multiplicar | sumar (+signo) | coseno / producto |
| **HRR** (Holographic Reduced Rep.) | Plate, 1994 | reales | convolución circular | sumar | coseno |
| **FHRR** (Frequency HRR) | Plate, 2003 | complejos (fase) | mult. compleja (suma de ángulos) | sumar | coseno de fase |
| **Sparse (SBDR)** | Rachkovskij/Kussul | binarios dispersos | (varía) | OR / disjunción | overlap |
| **TPR** (Tensor Product Rep.) | Smolensky, 1990 | reales (tensor) | producto tensorial | sumar | — |

### 7.2 Notas clave
- **MAP bipolar es isomorfo a BSC.** Son esencialmente el mismo modelo en dos vestidos (`{-1,+1}` vs
  `{0,1}`). MAP suele ser el más cómodo para empezar (multiplicar y sumar es intuitivo).
- **BSC tiene binding auto-inverso** (XOR consigo mismo da identidad): `A ⊗ A = 0`. Útil y elegante.
- **HRR/FHRR** manejan reales/complejos — más expresivos, pero más caros. FHRR (fase compleja) es muy
  usado en trabajo cognitivo moderno.
- **TPR** (Smolensky) es el ancestro teórico: potentísimo pero la dimensión **explota** con cada
  binding (producto tensorial), por eso los demás modelos son "reduced" (comprimen el tensor de vuelta
  al tamaño original).

### 7.3 Decisión de diseño para HOLOS
Empezar por **MAP bipolar** (intuitivo, eficiente, isomorfo a BSC) y **BSC binario** (máxima eficiencia
en CPU: XOR y popcount sobre bits empaquetados). Añadir **FHRR** después para trabajo cognitivo. Un
motor multi-modelo es parte del valor: pocas librerías los tratan a todos con rigor.

---

## 8. Ejemplos Canónicos — Ver el Paradigma en Acción

La mejor forma de "sentir" HDC es ver dos ejemplos clásicos.

### 8.1 Representar un registro (record) y consultarlo

Queremos representar `{país: USA, capital: Washington, moneda: dólar}` en **un solo hipervector**.

```
Símbolos atómicos (aleatorios): PAIS, CAPITAL, MONEDA, USA, WASHINGTON, DOLAR

Construir el registro (binding de cada rol con su valor, luego bundling):
   USA_record = (PAIS ⊗ USA) ⊕ (CAPITAL ⊗ WASHINGTON) ⊕ (MONEDA ⊗ DOLAR)

Consultar "¿cuál es la moneda?" (binding con el inverso del rol):
   respuesta = MONEDA ⊗ USA_record
             ≈ DOLAR  (+ ruido)
   cleanup(respuesta) = DOLAR  ✅
```

Un registro entero vive en un vector, y lo consultas con una multiplicación. **Sin bases de datos, sin
if/else — pura álgebra.**

### 8.2 El famoso "¿Cuál es el dólar de México?" (Kanerva, 2010)

Este ejemplo hizo famoso a HDC porque muestra **razonamiento analógico** emergiendo de álgebra vectorial.

```
Tenemos dos registros:
   USA    = (PAIS ⊗ USA)    ⊕ (MONEDA ⊗ DOLAR)
   MEXICO = (PAIS ⊗ MEXICO) ⊕ (MONEDA ⊗ PESO)

Pregunta: "¿Qué es, para México, lo que el dólar es para USA?"

Construimos la transformación de USA→MEXICO:
   T = USA ⊗ MEXICO         (mapea un país en el otro)

Aplicamos T al dólar:
   resultado = T ⊗ DOLAR
   cleanup(resultado) = PESO  ✅
```

El sistema "razonó" que el análogo del dólar en México es el peso — **sin haber sido programado para
esa pregunta**, solo con las tres operaciones. Esto es lo que emocionó a la comunidad cognitiva: la
analogía como operación algebraica.

### 8.3 Codificar una secuencia
```
Secuencia [A, B, C] usando permutación para el orden:
   seq = ρ²(A) ⊕ ρ(B) ⊕ C
Consultar "¿qué había 2 posiciones antes del final?":
   ρ⁻²(seq) contiene A recuperable por cleanup.
```

> Estos tres patrones —records, analogías, secuencias— son los ladrillos con los que se construyen las
> aplicaciones reales: clasificadores, memorias asociativas, razonadores.

---

## 9. Estado del Arte — Librerías, Aplicaciones, Hardware

### 9.1 Librerías existentes (sobre las que aprender / con las que comparar)

| Librería | Base | Notas |
|---|---|---|
| **TorchHD** | Python / PyTorch | La referencia moderna (JMLR 2023). Multi-modelo, CPU+GPU. **24× (CPU) y 54× (GPU)** más rápida que el código original de los papers. Trae datasets de benchmark. |
| **OpenHD / HDTorch** | Python/CUDA | Enfocadas en aceleración. |
| **ScalableHD** | CPU multinúcleo | Inferencia HDC de alto throughput en CPUs multi-core (2025). |
| **HPVM-HDC** | Sistema heterogéneo | Compilación para acelerar HDC (2024). |

> **Observación estratégica:** casi todo el ecosistema es **Python sobre PyTorch** (pensado para
> investigación, no para máxima eficiencia). Un **motor nativo en C++/Rust, optimizado para CPU con
> SIMD y bit-packing**, es un hueco real. Ahí puede vivir HOLOS.

### 9.2 Aplicaciones demostradas
- **Clasificación:** identificación de idioma, gestos por EMG, emociones por EEG, datos biológicos y
  genómicos, clasificación de grafos, texto.
- **Razonamiento:** analogías (dólar de México), razonamiento abductivo, memoria de trabajo.
- **Robótica y fusión de sensores.**
- **Reconocimiento visual de lugares** (visual place recognition).
- **Decodificar representaciones de LLMs** (Hyperdimensional Probe, 2025 — usar VSA para interpretar
  lo que "piensa" un modelo de lenguaje: un puente fascinante entre HDC e interpretabilidad de IA).

### 9.3 Benchmarks de referencia
- **7 datasets** populares de la literatura HDC/VSA para clasificación.
- **121 datasets** del repositorio UCI (benchmark amplio y consistente, 2024).
- Estos son nuestro campo de pruebas para medir HOLOS con rigor.

### 9.4 El ángulo de hardware / eficiencia (por qué el mundo vuelve a mirar HDC)
- **Edge AI:** aceleradores HDC para EEG en tiempo real, procesadores RISC-V dedicados a HDC (2024-25).
- **In-memory computing:** motores HDC en NAND flash 3D para genómica energéticamente eficiente.
- **Compresión:** técnicas como DPQ-HD y DecoHD para HDC bajo presupuestos de memoria extremos.
- **El dato que lo explica:** mover datos a/desde memoria = **62% de la energía** en cargas móviles.
  HDC minimiza justo eso. Por eso es candidato fuerte para la era post-Moore.

> Todo este trabajo de hardware **valida el paradigma** — pero la capa de software/algoritmos (donde
> vive HOLOS) es **100% CPU y pura CS**. No necesitas el hardware exótico para contribuir a la ciencia.

---

## 10. Honestidad — Limitaciones y Problemas Abiertos

Para no repetir el optimismo ciego, aquí está lo que HDC **todavía no** resuelve. Esto no son razones
para no hacerlo — son **exactamente los problemas donde hay contribución que hacer.**

### 10.1 Precisión menor que el deep learning (en tareas perceptuales duras)
En clasificación de imágenes complejas, HDC "puro" pierde frente a las redes neuronales. Suele
necesitar un extractor de características neuronal por delante. **HDC no es hoy un reemplazo universal
del deep learning** — es fuerte donde importan eficiencia, robustez, few-shot e interpretabilidad.

### 10.2 La codificación es EL problema abierto central ⭐
Este es el más importante y donde HOLOS puede aportar de verdad:
> Convertir datos crudos (una imagen, una señal, un texto) en hipervectores útiles se hace hoy con
> **codificadores fijos y aleatorios, elegidos empíricamente**, sin buena teoría de por qué funcionan.
> La calidad del codificador **domina** la precisión final, y está poco entendido.

Es la pregunta abierta #1 del campo. Quien la avance, avanza HDC entero.

### 10.3 Progreso reciente que apunta el camino
- **Codificadores entrenables (THDC, 2026):** hacer el codificador aprendible vía backpropagation, en
  vez de aleatorio. Resultado notable: **igual o mejor precisión bajando la dimensión de 10.000 a 64.**
  Esto sugiere que gran parte de la "necesidad" de dimensión alta era por codificadores pobres.
- **Encoders adaptativos / holográficos**, tasas de aprendizaje adaptativas.

### 10.4 Capacidad y crosstalk
El bundling tiene límite: juntas demasiados hipervectores y empiezan a interferir. Cuántos "caben"
antes de que el cleanup falle es un problema de teoría de la capacidad que hay que respetar al diseñar.

### 10.5 El trade-off precisión/eficiencia
La precisión sube con la dimensión y luego se estanca; la latencia y la energía suben sin parar. Elegir
la dimensión correcta es un arte con tensiones reales.

> **En una frase honesta:** HDC es un paradigma con ventajas únicas y un problema central sin resolver
> (la codificación). No es una bala de plata garantizada — es una apuesta de alto techo en un campo
> con espacio. Justo lo que pediste: teoría vieja, real, que podría despertar.

---

## 11. La Tesis de HOLOS — Dónde Aportamos

### 11.1 Lo que NO hacemos
- ❌ Inventar un modelo VSA nuevo desde la teoría pura (es investigación de años; no es el cuello de botella).
- ❌ Reescribir TorchHD en Python (ya existe y es bueno para investigación).
- ❌ Prometer que HDC vence al deep learning en todo (no lo hace, hoy).

### 11.2 Lo que SÍ hacemos — foco láser en dos huecos reales

```
   ┌─────────────────────────────────────────────────────────────┐
   │  HUECO 1 — INGENIERÍA (rendimiento) ⭐                       │
   │  Un motor HDC/VSA NATIVO en C++/Rust, optimizado para CPU:   │
   │  SIMD, bit-packing (BSC = bits reales + XOR + popcount),     │
   │  búsqueda de cleanup ultra-rápida. El ecosistema es Python;  │
   │  un motor de sistemas serio es un hueco.                     │
   ├─────────────────────────────────────────────────────────────┤
   │  HUECO 2 — CIENCIA (el problema de la codificación) ⭐       │
   │  Experimentar con codificadores: aleatorios vs entrenables,  │
   │  medir en los benchmarks (UCI, 7 datasets), y buscar         │
   │  mejores formas de mapear datos crudos → hipervectores.      │
   │  Es EL problema abierto del campo.                           │
   └─────────────────────────────────────────────────────────────┘
```

### 11.3 El principio rector
> Construir el motor nativo primero (rápido y correcto, verificado contra TorchHD), y usarlo como
> laboratorio para atacar el problema de la codificación con experimentos medibles.

### 11.4 Por qué esta tesis encaja contigo
- **Rendimiento y bajo nivel** (SIMD, bit-packing, búsqueda optimizada) → tu zona, tu talento de PSEG.
- **100% CPU, corre en tu Mac** → sin depender de nube ni GPU.
- **Campo poco poblado** → tu trabajo destaca en vez de perderse entre millones.
- **Un problema abierto real** (codificación) → posibilidad de contribución científica de verdad, no
  solo un clon.
- **Puente a la IA:** el trabajo de "sondear LLMs con VSA" conecta HDC con interpretabilidad — una
  puerta a los temas de más alto techo si quisieras cruzarla después.

---

## 12. Elección de Lenguaje y Stack

### 12.1 El debate

| Criterio | C++ / SIMD | Rust | Python (prototipo) |
|---|---|---|---|
| Rendimiento CPU | ✅ Máximo | ✅ Máximo | ❌ Lento |
| Bit-packing / intrínsecos SIMD | ✅ Nativo | ✅ Nativo (`std::simd`, `packed_simd`) | ❌ |
| Seguridad de memoria | ❌ Manual | ✅ Garantizada | ✅ |
| Tu experiencia previa | ✅ (PSEG) | ➖ | ➖ |
| Ecosistema HDC de referencia | ➖ | ➖ | ✅ (TorchHD) |

### 12.2 Decisión

**Fase de prototipo/aprendizaje: Python.** Para *entender* el paradigma rápido y sin fricción (usando
NumPy o TorchHD como referencia y verdad de terreno). Ver funcionar el "dólar de México" en 20 líneas.

**Motor de producción: Rust** (recomendado) **o C++20.**
- **Rust** encaja especialmente bien aquí: seguridad de memoria + rendimiento nativo + SIMD, y es un
  lenguaje moderno donde crecerás. Sin la complejidad de interop-GPU que hacía a C++ obligatorio en
  proyectos anteriores.
- **C++20** si prefieres apoyarte en tu experiencia de PSEG.

### 12.3 Stack concreto
```
Prototipo:      Python 3.12 + NumPy  (+ TorchHD como referencia)
Motor:          Rust (edición 2021+) o C++20
SIMD:           std::simd (Rust) / intrínsecos AVX2 (C++)   [tu Mac Xeon soporta AVX/AVX2]
Build:          Cargo (Rust) o CMake (C++)
Benchmark:      criterion (Rust) / Google Benchmark (C++)
Verificación:   tests que comparan contra TorchHD/NumPy (verdad de terreno)
Datos:          benchmarks UCI + los 7 datasets clásicos de HDC
Gráficas:       Python + matplotlib para visualizar resultados de experimentos
```

> **Nota de hardware:** todo esto corre perfecto en tu Mac Pro (Xeon multinúcleo, con AVX). HDC es
> *embarazosamente paralelo* — tus múltiples núcleos son una ventaja real, no una limitación.

---

## 13. Arquitectura Técnica de HOLOS

### 13.1 Vista de alto nivel

```
═══════════════════════════════════════════════════════════════════════
                     HOLOS — Motor de Computación Hiperdimensional
═══════════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────────┐
  │  API PÚBLICA                                                     │
  │  Hypervector · bind() · bundle() · permute() · similarity()     │
  │  ItemMemory · encode() · Classifier                             │
  └───────────────────────────────┬─────────────────────────────────┘
                                  │
  ┌───────────────────────────────▼─────────────────────────────────┐
  │  CAPA DE APLICACIONES (Nivel 3)                                 │
  │  ┌──────────────┐ ┌───────────────┐ ┌────────────────────────┐ │
  │  │ Clasificador │ │ Razonador     │ │ Codificadores          │ │
  │  │ (few-shot)   │ │ (analogías)   │ │ (texto, señal, imagen) │ │
  │  └──────────────┘ └───────────────┘ └────────────────────────┘ │
  └───────────────────────────────┬─────────────────────────────────┘
                                  │
  ┌───────────────────────────────▼─────────────────────────────────┐
  │  CAPA DE MEMORIA (Nivel 2)                                      │
  │  ┌────────────────────────┐ ┌──────────────────────────────┐   │
  │  │ Item Memory            │ │ Cleanup (nearest neighbor)   │   │
  │  │ (tabla de símbolos)    │ │ búsqueda SIMD optimizada ⭐   │   │
  │  └────────────────────────┘ └──────────────────────────────┘   │
  └───────────────────────────────┬─────────────────────────────────┘
                                  │
  ┌───────────────────────────────▼─────────────────────────────────┐
  │  CAPA DE ÁLGEBRA (Nivel 1) ⭐ EL CORAZÓN                        │
  │  ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌──────────────────┐  │
  │  │ bind ⊗   │ │ bundle ⊕ │ │ permute ρ │ │ similarity       │  │
  │  │ XOR/mult │ │ sum/maj  │ │ shift     │ │ Hamming/coseno   │  │
  │  └──────────┘ └──────────┘ └───────────┘ └──────────────────┘  │
  │  Implementadas por modelo: BSC · MAP · (FHRR)                   │
  └───────────────────────────────┬─────────────────────────────────┘
                                  │
  ┌───────────────────────────────▼─────────────────────────────────┐
  │  CAPA DE REPRESENTACIÓN (Nivel 0)                               │
  │  ┌──────────────────────────┐ ┌──────────────────────────────┐ │
  │  │ Hypervector packed (bits)│ │ SIMD kernels (AVX2) ⭐        │ │
  │  │ bit-packing para BSC     │ │ popcount, XOR, add vectorial │ │
  │  │ layouts alineados        │ │ paralelismo multinúcleo      │ │
  │  └──────────────────────────┘ └──────────────────────────────┘ │
  └───────────────────────────────┬─────────────────────────────────┘
                                  │
  ┌───────────────────────────────▼─────────────────────────────────┐
  │  VERIFICACIÓN Y BENCHMARK (transversal)                        │
  │  • Verdad de terreno contra TorchHD / NumPy (correctitud)      │
  │  • criterion / Google Benchmark (rendimiento)                  │
  │  • Datasets UCI + 7 clásicos (precisión en tareas reales)      │
  └─────────────────────────────────────────────────────────────────┘
```

### 13.2 Decisiones de diseño clave

**1. Bit-packing para BSC desde el día uno.**
Un hipervector binario de 10.000 bits cabe en **1.250 bytes** si empaquetas bits. El binding (XOR) y la
similitud (Hamming = popcount de XOR) se vuelven operaciones sobre palabras de 64 bits + `popcount`
hardware. Esto es órdenes de magnitud más rápido que representar cada bit como un byte o un float.

**2. La búsqueda de cleanup es el kernel que más se optimiza.**
Comparar un vector contra miles de símbolos es el cuello de botella de muchas apps. SIMD + paralelismo
multinúcleo + (más tarde) indexado aproximado. Es tu "NTT" de este proyecto: la primitiva estrella.

**3. Multi-modelo, pero empezando por uno.**
Arquitectura que permita BSC, MAP y FHRR, pero implementamos y validamos **MAP** primero (intuitivo),
luego **BSC** (máxima eficiencia), luego FHRR.

**4. Correctitud antes que velocidad.**
Cada operación se verifica contra NumPy/TorchHD. Un bind rapidísimo pero incorrecto no vale nada.

### 13.3 Flujo de un clasificador HDC (ejemplo concreto)
```
Entrenar (one-shot / few-shot):
  Para cada clase c:
    prototipo[c] = bundle( encode(x) para cada ejemplo x de la clase c )
    → un solo hipervector por clase. Entrenamiento = sumar. Rapidísimo.

Predecir una muestra x:
  h = encode(x)
  clase = argmax_c  similarity(h, prototipo[c])   ← búsqueda de cleanup
```
Fíjate: **entrenar es sumar vectores.** Sin backpropagation, sin épocas, sin GPU. Esa es la promesa de
eficiencia de HDC hecha código.

---

## 14. Estructura del Repositorio

```
holos/
├── Cargo.toml                     # (o CMakeLists.txt si C++)
├── README.md
│
├── prototipo_py/                  # Fase 0: entender el paradigma en Python
│   ├── 01_cuasi_ortogonalidad.py  # ver que vectores aleatorios son ⊥
│   ├── 02_record_y_consulta.py    # el registro país/moneda
│   ├── 03_dolar_de_mexico.py      # razonamiento analógico
│   └── 04_clasificador.py         # clasificar un dataset UCI
│
├── holos_core/                    # El motor nativo (Rust o C++)
│   ├── src/
│   │   ├── hypervector.rs         # representación + bit-packing
│   │   ├── models/
│   │   │   ├── map.rs             # modelo MAP (bipolar)
│   │   │   ├── bsc.rs             # modelo BSC (binario, XOR+popcount) ⭐
│   │   │   └── fhrr.rs            # modelo FHRR (fase compleja)
│   │   ├── algebra.rs             # bind, bundle, permute, similarity
│   │   ├── item_memory.rs         # memoria de símbolos
│   │   ├── cleanup.rs             # búsqueda vecino más cercano (SIMD) ⭐
│   │   └── simd/
│   │       └── kernels.rs         # XOR, popcount, add vectorial (AVX2)
│   │
├── holos_apps/                    # Aplicaciones
│   ├── classifier.rs              # clasificador few-shot
│   ├── reasoner.rs                # analogías
│   └── encoders/
│       ├── text.rs                # texto → hipervector (n-gramas)
│       ├── signal.rs              # señal → hipervector
│       └── trainable.rs           # ⭐ codificador entrenable (el problema abierto)
│
├── holos_verify/                  # Correctitud contra TorchHD/NumPy
│   └── vs_torchhd.rs
│
├── holos_bench/                   # Rendimiento
│   ├── bench_bind.rs
│   ├── bench_cleanup.rs           # el kernel estrella
│   └── results/
│
├── datasets/                      # benchmarks (UCI, 7 clásicos)
│
├── docs/
│   ├── HOLOS_investigacion_tecnica.md   # este documento
│   └── notes/                     # bitácora de experimentos
│
└── tests/
```

---

## 15. Roadmap de Implementación

> Filosofía: **cada fase produce algo que funciona y se puede medir.** Concepto en Python primero,
> luego motor nativo, luego ciencia. Ver §18 para el experimento que va *antes* de la Fase 0.

### Fase 0 — Sentir el paradigma en Python (Semanas 1-2)
**Objetivo:** entender HDC con las manos, sin optimizar nada.
- [ ] Generar hipervectores aleatorios y **ver la cuasi-ortogonalidad** con tus ojos (histograma de similitudes).
- [ ] Implementar bind/bundle/permute/similarity en NumPy (~30 líneas).
- [ ] Construir el **registro país/moneda** y consultarlo.
- [ ] Reproducir el **"dólar de México"** — ver el razonamiento analógico funcionar.
- [ ] **Hito:** "el sistema respondió PESO sin que yo programara esa pregunta."

### Fase 1 — Clasificador real en Python (Semanas 3-4)
**Objetivo:** HDC haciendo algo útil sobre datos reales.
- [ ] Codificador simple (n-gramas para texto, o binning para señales).
- [ ] Clasificador few-shot (prototipos por clase = bundling).
- [ ] Medir precisión en 1-2 datasets UCI; comparar contra TorchHD.
- [ ] **Hito:** "clasifiqué un dataset real con entrenamiento = sumar vectores."

### Fase 2 — Motor nativo: el álgebra (Semanas 5-10) ⭐
**Objetivo:** las operaciones core en Rust/C++, correctas y rápidas.
- [ ] Representación de hipervector con **bit-packing** (BSC).
- [ ] bind (XOR), similarity (Hamming/popcount), bundle, permute — con SIMD.
- [ ] Verificar **bit-a-bit contra el prototipo Python/TorchHD**.
- [ ] Benchmark: objetivo, superar ampliamente a TorchHD-CPU en las primitivas.
- [ ] **Hito:** "mi álgebra hiperdimensional nativa es correcta y N× más rápida que Python."

### Fase 3 — Memoria y cleanup optimizado (Semanas 11-14)
**Objetivo:** la búsqueda asociativa, el kernel estrella.
- [ ] Item memory + cleanup (nearest neighbor) con SIMD y multinúcleo.
- [ ] Clasificador nativo end-to-end sobre datasets UCI.
- [ ] **Hito:** "clasifico miles de muestras por segundo en mi CPU, con precisión igual al prototipo."

### Fase 4 — El problema abierto: codificación (Mes 4+) ⭐
**Objetivo:** aportar a la ciencia, no solo a la ingeniería.
- [ ] Implementar y comparar codificadores: aleatorio vs adaptativo vs **entrenable** (estilo THDC).
- [ ] Medir en el benchmark de 121 datasets UCI.
- [ ] Buscar una mejora reproducible (mejor precisión, o igual precisión con menor dimensión).
- [ ] **Hito:** "encontré un codificador que mejora el estado del arte en X, con datos que lo prueban."

### Fase 5 — Contribución y comunidad (Mes 6+)
**Objetivo:** que HOLOS exista para el mundo, no aislado.
- [ ] Publicar el motor open-source con benchmarks reproducibles.
- [ ] Escribir un blog/paper técnico de la mejor optimización o hallazgo.
- [ ] (Opcional/ambicioso) puente HDC ↔ interpretabilidad de LLMs.
- [ ] **Hito:** "otra persona usó HOLOS, o citó mi resultado."

---

## 16. Desafíos Técnicos Críticos

### 16.1 La codificación (el gran reto, §10.2)
Convertir datos crudos en buenos hipervectores es un arte poco entendido y domina la precisión. Es el
desafío científico central — y la mayor oportunidad de contribución.

### 16.2 Bit-packing correcto y rápido
Empaquetar 10.000 bits en palabras de 64 y hacer XOR + popcount sin bugs sutiles de alineación o de
manejo del "resto" (bits sobrantes de la última palabra). Un error aquí corrompe la similitud silenciosamente.

### 16.3 La capacidad del bundling
Cuántos hipervectores puedes superponer antes de que el cleanup falle. Hay que respetar la teoría de la
capacidad al diseñar (dimensión vs número de ítems vs tasa de error tolerable).

### 16.4 Verificación rigurosa
Como todo es "aproximado y ruidoso", es fácil tener un bug que *parece* funcionar (da resultados
plausibles pero peores). Sin comparación continua contra TorchHD/NumPy y sin medir precisión en
datasets reales, estás ciego.

### 16.5 Elegir bien la dimensión y el modelo
Dimensión alta = más precisión pero más coste. Modelo binario (BSC) = máxima velocidad; real/complejo
(HRR/FHRR) = más expresividad. Elegir según la tarea es parte del oficio.

---

## 17. Métricas de Éxito

### 17.1 Métricas técnicas objetivo

| Métrica | Objetivo MVP | Objetivo Ambicioso |
|---|---|---|
| Correctitud vs TorchHD/NumPy | 100% (dentro de tolerancia de ruido) | 100% |
| Speedup de bind/similarity vs TorchHD-CPU | > 10× | > 50× |
| Throughput de cleanup (muestras/seg) | Interactivo | Tiempo real sobre millones de símbolos |
| Precisión en datasets UCI | Igual a TorchHD | Igual o mejor con menor dimensión |
| Entrenamiento de un clasificador | Segundos en CPU | Sub-segundo |

### 17.2 Métricas de proyecto (igual de importantes)
- **Reproducibilidad:** cualquiera corre `holos_bench` y obtiene tus números.
- **Contribución real:** un motor open-source usable, o un hallazgo sobre codificación con datos.
- **Comunicación:** un post/paper que explique una optimización o un resultado con rigor.

---

## 18. El Experimento que Valida o Refuta el Proyecto

> Antes de comprometer meses, gástate **dos semanas** en responder: *"¿me obsesiona este paradigma o
> solo me parece elegante?"* — y *"¿tengo la intuición para el problema de la codificación?"*

**El experimento de 2 semanas (Fase 0, todo en tu Mac, sin nube):**

1. **Semana 1 — Ver la magia.**
   - En Python + NumPy, genera hipervectores aleatorios y **grafica la cuasi-ortogonalidad** (vas a
     *ver* que todos los aleatorios son ⊥). Ese histograma es el "clic" del paradigma.
   - Implementa las tres operaciones y construye el **registro país/moneda**. Consúltalo.

2. **Semana 2 — Ver el razonamiento y clasificar.**
   - Reproduce el **"dólar de México"**. Cuando el sistema responda `PESO` sin que lo programaras para
     eso, vas a sentir por qué esto emocionó a la gente.
   - Codifica un dataset UCI sencillo y clasifícalo con prototipos (bundling). Mide la precisión.

**El veredicto:**
- Si al final quieres **bajar al motor nativo y optimizar el cleanup con SIMD** → este es tu proyecto.
  Adelante con la Fase 2.
- Si el paradigma te dejó frío → lo descubriste en 2 semanas y no en 6 meses. Sin nube, sin gastar un peso.

**Barato, honesto, y 100% en tu hardware.**

---

## 19. Referencias y Recursos Clave

### Aprender HDC/VSA (empezar aquí)
- **Kanerva (2009)** — *Hyperdimensional Computing: An Introduction to Computing in Distributed
  Representation with High-Dimensional Random Vectors* (el paper fundacional, muy legible).
- **Kanerva (2010)** — *What we mean when we say "What's the dollar of Mexico?"* (el razonamiento analógico).
- **Survey en dos partes (2021-2022, ACM Computing Surveys):**
  - Part I: *Models and Data Transformations* — https://arxiv.org/abs/2111.06077
  - Part II: *Applications, Cognitive Models, and Challenges*
- **A comparison of Vector Symbolic Architectures** (Schlegel et al.) — https://arxiv.org/pdf/2001.11797
- **hd-computing.com** — portal de la comunidad · **Redwood Center, Berkeley** (Kleyko, Frady, Sommer, Olshausen).

### Librerías (aprender / comparar contra)
- **TorchHD** — https://github.com/hyperdimensional-computing/torchhd (la referencia, JMLR 2023)
- **ScalableHD** — inferencia HDC en CPUs multinúcleo (2025)

### Fundamentos (papers seminales)
- **Kanerva (1988)** — *Sparse Distributed Memory*
- **Plate (1994/2003)** — *Holographic Reduced Representations* (HRR / FHRR)
- **Gayler (2003)** — *Vector Symbolic Architectures answer Jackendoff's challenges* (acuña "VSA", modelo MAP)
- **Smolensky (1990)** — *Tensor Product Representations*

### Frontera actual (2024-2026)
- **THDC (2026)** — *Training Hyperdimensional Computing Models with Backpropagation* (codificadores entrenables)
- **Hyperdimensional Probe (2025)** — decodificar representaciones de LLMs con VSA (puente a interpretabilidad)
- Surveys recientes de arquitecturas HDC para IA, edge y hardware.

### Herramientas de desarrollo
- **Python 3.12 + NumPy** (prototipo) · **matplotlib** (visualización)
- **Rust** (Cargo, criterion, std::simd) o **C++20** (CMake, Google Benchmark, intrínsecos AVX2)

---

> **Documento generado durante la fase de investigación del proyecto HOLOS.**
> *HDC es una bella durmiente: teoría vieja, sólida y dormida, que podría despertar. La barrera es
> ingeniería y una buena idea sobre codificación — ambas al alcance de una persona con una CPU y
> voluntad. Este documento debe actualizarse conforme avance la implementación.*
