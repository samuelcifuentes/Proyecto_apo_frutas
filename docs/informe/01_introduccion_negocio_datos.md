# Secciones del informe — Samuel Cifuentes

> Borrador para pegar en la plantilla IEEE (`conference-template-letter.docx`).
> Secciones: I. Introducción · II-A. Comprensión del negocio · III. Datos (comprensión y preparación).
> Las cifras salen de `notebooks/01_eda.ipynb`, `data/splits.csv` y `data/processed/stats.json`;
> si se vuelve a correr el preprocesamiento con más etiquetas "regular", actualizar los números marcados con ⟳.

---

## Resumen (aporte para el abstract)

La clasificación manual de frutas por calidad es lenta, subjetiva y contribuye a la pérdida
poscosecha de alimentos. Este trabajo propone un sistema de visión por computador que, a partir de
la fotografía de una fruta individual, asigna una de tres categorías de calidad (buena, regular,
mala) y estima su tamaño relativo. Se utilizó el conjunto FruitNet (19.526 imágenes de seis frutas),
complementado con fotografías propias y recortes etiquetados manualmente, siguiendo la metodología
CRISP-DM.

---

## I. Introducción

En mercados y agroindustrias la clasificación de productos frescos según su madurez, tamaño o
defectos visibles se hace mayoritariamente a mano. Este proceso es lento, depende del criterio de
cada operario y es difícil de estandarizar, lo que genera pérdidas económicas y desperdicio de
alimentos. A nivel mundial, alrededor del 14 % de los alimentos se pierde entre la cosecha y la venta
minorista [1]. En Colombia se pierden o desperdician 9,76 millones de toneladas de alimentos al año
(34 % de la oferta disponible), de las cuales cerca de 6,1 millones corresponden a frutas y verduras
[2].

La visión por computador permite automatizar esta inspección: una cámara y un modelo de aprendizaje
automático pueden evaluar cada producto con el mismo criterio, de forma rápida y a bajo costo. El
reto es que la calidad de una fruta se manifiesta en señales visuales sutiles (manchas, cambios de
color, textura de la piel) que varían entre especies, iluminación y fondo de la fotografía.

En este proyecto se desarrolla un sistema que recibe la fotografía de una fruta individual sobre un
fondo simple y entrega: (i) su clase de calidad —buena, regular o mala— y (ii) una estimación de su
tamaño (pequeño, mediano o grande). Se comparan dos modelos de aprendizaje automático clásico sobre
características extraídas manualmente y una red neuronal convolucional (CNN) entrenada desde cero, y
el mejor modelo se despliega en una interfaz gráfica con cámara.

## II-A. Comprensión del negocio

**Problema.** Un comerciante o una planta de empaque necesita separar las frutas aptas para la venta
a precio completo, las que pueden venderse con descuento o procesarse, y las que deben descartarse.
Por eso se definieron **tres** clases de calidad en lugar de las dos (buena/mala) que trae el
conjunto de datos original: la clase intermedia *regular* corresponde a una decisión de negocio real
(vender con descuento o transformar) y es justamente donde la inspección humana es más subjetiva.

**Criterios de calidad** (detallados en `docs/guia_etiquetado.md`):
- *Buena*: sin defectos visibles, color uniforme y propio de la madurez.
- *Regular*: defectos leves que cubren menos del ~20 % de la superficie visible (manchas pequeñas,
  rayones, golpe leve, poca o excesiva madurez).
- *Mala*: podredumbre, moho, zonas negras extensas o defectos en más del ~20 % de la superficie.

**Criterio de éxito.** Como un error de clasificación tiene costos distintos según la clase (vender
una fruta mala daña la confianza del cliente; descartar una buena es pérdida directa), se evalúa con
**F1-macro**, que pondera por igual las tres clases aunque estén desbalanceadas, además de la
exactitud (*accuracy*) y la matriz de confusión.

## III. Datos

### A. Fuentes

1. **FruitNet** [3] (publicado en Kaggle como *Fruit Quality Classification*): 19.526 imágenes de seis
   frutas —manzana, banano, guayaba, limón, naranja y granada— tomadas con celular en distintos
   fondos e iluminaciones, organizadas en tres carpetas: *Good* (11.664), *Bad* (6.788) y *Mixed*
   (1.074 fotos con varias frutas).
2. **Fotografías propias**: fotos de una fruta por imagen sobre mesa, tomadas por el grupo siguiendo
   una guía común (distancia ~30 cm, buena luz, sin manos), etiquetadas con la misma guía. ⟳
3. **Recortes de *Mixed*** (2.570, generados automáticamente) e **imágenes re-etiquetadas**: se
   revisaron imágenes de *Bad* con defectos leves para construir la clase *regular* (287 imágenes en
   total, sumando fotos propias y recortes). ⟳

### B. Análisis exploratorio (comprensión de los datos)

**Distribución de clases** (Fig. X — `experiments/results/eda/01_distribucion_clases.svg`). Las clases
tienen ≈1.100 imágenes cada una, excepto *granada buena* con 5.940 imágenes, 5,5 veces la clase más
pequeña. Globalmente, 63 % de las imágenes son *buenas* y 37 % *malas*. Sin corrección, un modelo
tendería a sobrepredecir esa clase.

**Resolución.** El 64 % de las imágenes es de 256×256 px y la mayoría del resto de 192×256 px, pero
hay 564 fotografías de 3000×4000 y 8000×6000 px. Todas se llevaron a un tamaño fijo.

**Color** (Fig. Y — `experiments/results/eda/04_hsv_por_clase.svg`). Se calculó la media de los canales
del espacio HSV (tono H, saturación S, brillo V) en una muestra de 150 imágenes por clase y fruta.
En las seis frutas las frutas buenas son **más saturadas** que las malas (ΔS = S̄_buena − S̄_mala > 0,
hasta 0,13 en naranja): la podredumbre "apaga" el color. En cambio, las imágenes malas resultaron
**más brillantes** (ΔV < 0 en cinco de seis frutas). Como la estadística se calculó sobre toda la
imagen, esto indica que las fotos malas se tomaron sobre fondos más claros: **el fondo está
correlacionado con la clase**, y un modelo podría aprender a reconocer fondos en lugar de frutas.
Esta observación motivó segmentar la fruta antes de extraer características y evaluar con fotos
propias en un fondo distinto.

**Otros hallazgos.** Muchas fotos incluyen la mano que sostiene la fruta, y algunas imágenes de
*Good* y *Bad* contienen varias frutas. En *Bad* conviven defectos leves y graves, lo que justifica
la clase *regular*.

### C. Preparación de los datos

**1) Segmentación de la fruta.** Se probaron dos enfoques:

- *Clásico*: umbral de Otsu [4] sobre la saturación y sobre la distancia de color al fondo en el
  espacio CIELAB, refinado con GrabCut [5]. Otsu elige el umbral t que maximiza la varianza entre
  clases:
  
  t* = argmax_t  ω₀(t) ω₁(t) [μ₀(t) − μ₁(t)]²        (1)

  donde ω₀, ω₁ son las proporciones de píxeles a cada lado del umbral y μ₀, μ₁ sus medias. La
  distancia al fondo es d(p) = ‖Lab(p) − Lab_fondo‖₂, con Lab_fondo la mediana del borde de la imagen.
- *U²-Net* [6] preentrenada (biblioteca `rembg`), una red de detección de objetos salientes que
  predice para cada píxel la probabilidad de pertenecer al objeto principal.

En las fotografías propias (fruta sobre mesa de madera) el método clásico falló en 9 de 10 imágenes
—la madera es tan saturada como la fruta— mientras que U²-Net acertó en las 10 (Fig. Z —
`experiments/results/segmentacion/fotos_propias.jpg`); por eso se adoptó U²-Net. Para las fotos
*Mixed*, las frutas que se tocan se separaron con la transformada de distancia y el algoritmo
*watershed*.

**2) Estimación de tamaño.** A partir del área A (en píxeles) de la máscara de la fruta se calcula el
diámetro equivalente y se normaliza por el lado menor de la imagen:

  d_eq = √(4A/π),    d_norm = d_eq / min(alto, ancho)        (2)

Las categorías pequeño/mediano/grande se obtienen con los terciles de d_norm **por tipo de fruta**,
pues un limón grande sigue siendo más pequeño que una manzana pequeña.

**3) Balanceo.** Se limitó cada combinación (fruta, calidad) a 1.200 imágenes mediante submuestreo
aleatorio; a *granada buena* se le retiraron 4.740 imágenes.

**4) División sin fuga de datos.** FruitNet contiene ráfagas: varias fotos consecutivas de la misma
fruta con un leve cambio de ángulo. Si una foto de la ráfaga queda en entrenamiento y otra en prueba,
el modelo es evaluado con una fruta que ya vio y las métricas se inflan. Para evitarlo se calculó un
*hash* perceptual por diferencias (dHash, 64 bits) de cada imagen y se unieron en un mismo grupo las
fotos consecutivas de una carpeta cuya distancia de Hamming es ≤ 10 bits (entre fotos tomadas al azar
la distancia típica es de 27 bits). Se obtuvieron 5.074 grupos para 13.712 imágenes. La división
70/15/15 se hizo **por grupos** y estratificada por (fruta, calidad): todas las fotos de una ráfaga van
al mismo subconjunto. Resultado: 9.719 imágenes de entrenamiento, 2.051 de validación y 1.942 de
prueba; la clase *regular* quedó con 224 / 30 / 33. ⟳

**5) Redimensionamiento.** Cada imagen se escaló para que su lado mayor midiera 128 px, rellenando
con negro hasta 128×128 (*letterbox*), con factor de escala s = 128 / max(alto, ancho). Así no se
deforma la fruta, cuya forma es relevante para la calidad y el tamaño.

**6) Normalización.** Se aplicó estandarización z-score por canal RGB:

  x' = (x/255 − μ_c) / σ_c,    σ_c² = E[x²] − (E[x])²        (3)

con μ = (0,559; 0,537; 0,492) y σ = (0,294; 0,295; 0,310) calculadas **solo** con el conjunto de
entrenamiento, para no filtrar información de validación y prueba. ⟳

**7) Aumento de datos** (solo en entrenamiento): volteos horizontal y vertical, rotación aleatoria de
0–360° y variación de brillo de ±15 %. No se altera el tono, porque el color es información clave
para la calidad.

**8) Privacidad.** A las fotografías propias se les eliminaron los metadatos EXIF (que pueden incluir
la ubicación GPS) antes de incorporarlas al repositorio.

---

## Referencias (formato IEEE)

[1] FAO, *The State of Food and Agriculture 2019. Moving Forward on Food Loss and Waste Reduction*.
Rome, Italy: FAO, 2019.

[2] Departamento Nacional de Planeación (DNP), *Pérdida y desperdicio de alimentos en Colombia:
Estudio de la Dirección de Seguimiento y Evaluación de Políticas Públicas*. Bogotá, Colombia: DNP, 2016.

[3] V. Meshram and K. Patil, "FruitNet: Indian fruits image dataset with quality for machine learning
applications," *Data in Brief*, vol. 40, p. 107686, 2022, doi: 10.1016/j.dib.2021.107686.

[4] N. Otsu, "A threshold selection method from gray-level histograms," *IEEE Trans. Syst., Man,
Cybern.*, vol. 9, no. 1, pp. 62–66, 1979.

[5] C. Rother, V. Kolmogorov, and A. Blake, "'GrabCut': Interactive foreground extraction using iterated
graph cuts," *ACM Trans. Graph.*, vol. 23, no. 3, pp. 309–314, 2004.

[6] X. Qin, Z. Zhang, C. Huang, M. Dehghan, O. R. Zaiane, and M. Jagersand, "U²-Net: Going deeper with
nested U-structure for salient object detection," *Pattern Recognition*, vol. 106, p. 107404, 2020,
doi: 10.1016/j.patcog.2020.107404.

[7] P. Chapman *et al.*, *CRISP-DM 1.0: Step-by-step data mining guide*. SPSS, 2000.
