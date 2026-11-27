# Formulación matemática y modelos de aprendizaje automático clásico

> Borrador para la plantilla IEEE. Sección II-B (formulación del problema) y III-D (modelos clásicos).
> Evidencia de PI3: formulación, estrategia de solución y fundamentos matemáticos.
> Código: `src/features/extraction.py`, `src/features/size_estimation.py`, `src/models/train_classical.py`.

## II-B. Formulación del problema

Sea 𝒟 = {(Iₙ, yₙ)}ₙ₌₁ᴺ el conjunto de imágenes Iₙ de una fruta, con etiqueta de calidad
yₙ ∈ 𝒴 = {buena, regular, mala}. El objetivo es aprender un clasificador f: ℐ → 𝒴 que minimice el
riesgo esperado ponderado por clase:

  f* = argmin_f  Σ_c w_c · E[ 𝟙{f(I) ≠ y} | y = c ],    w_c = N / (K · N_c)        (4)

donde K = 3 es el número de clases y N_c el número de ejemplos de la clase c. Los pesos w_c
compensan el desbalance: *regular* representa ≈2 % de los datos, y sin ellos un modelo podría
ignorarla y aun así obtener alta exactitud. Por la misma razón se optimiza y reporta el **F1-macro**:

  F1_c = 2 P_c R_c / (P_c + R_c),    F1-macro = (1/K) Σ_c F1_c        (5)

con P_c (precisión) y R_c (sensibilidad) de la clase c.

Para los modelos clásicos la imagen se transforma en un vector de características
x = φ(I, M) ∈ ℝ²³, calculado únicamente sobre la máscara M de la fruta segmentada (Sección III-C),
y el problema pasa a ser f: ℝ²³ → 𝒴. La estimación de tamaño es un problema aparte, resuelto con
el diámetro normalizado de la ecuación (2).

**¿Por qué es un problema complejo de ingeniería?**
1. *Alta variabilidad intraclase y baja variabilidad interclase*: una manzana buena y una regular
   pueden diferir en una mancha de pocos píxeles, mientras dos manzanas buenas difieren en color,
   variedad, iluminación y ángulo.
2. *Etiquetas subjetivas*: el límite entre regular y mala (≈20 % de superficie dañada) depende del
   criterio del anotador; en la revisión manual, 40 de 50 etiquetas *regular* hubo que corregirlas.
3. *Sesgo del fondo*: el EDA mostró que el fondo se correlaciona con la clase, lo que obliga a
   segmentar antes de modelar.
4. *Desbalance severo* de la clase intermedia y *seis especies* con apariencias distintas en un
   mismo modelo.

## III-D. Características (φ)

Todas se calculan sobre la imagen de 128×128 px del preprocesamiento y solo sobre los píxeles de
la máscara M, para que el fondo no influya.

**Color (6).** Media y desviación estándar de cada canal HSV (escalado a [0, 1]) dentro de M:
μ_k = (1/|M|) Σ_{p∈M} I_k(p),  σ_k = √[(1/|M|) Σ_{p∈M} (I_k(p) − μ_k)²], k ∈ {H, S, V}.
El EDA mostró que la saturación separa buenas de malas en las seis frutas.

**Textura (4).** Matriz de co-ocurrencia de niveles de gris (GLCM) con 32 niveles, distancia 1 y
cuatro direcciones (0°, 45°, 90°, 135°), contando solo pares de píxeles dentro de M. Con P(i, j) la
probabilidad normalizada de que un píxel de nivel i tenga un vecino de nivel j:

  contraste = Σ (i − j)² P(i,j),   homogeneidad = Σ P(i,j) / (1 + |i − j|),
  energía = Σ P(i,j)²,   correlación = Σ (i − μᵢ)(j − μⱼ) P(i,j) / (σᵢ σⱼ)        (6)

Las manchas y la podredumbre producen cambios bruscos de intensidad: aumentan el contraste y
reducen la homogeneidad.

**Forma (13).** Sobre el contorno principal con área A, perímetro L y envolvente convexa A_hull:
área relativa A/(alto·ancho), perímetro relativo L/min(alto, ancho), circularidad 4πA/L²,
razón de aspecto del rectángulo envolvente, solidez A/A_hull, excentricidad √(1 − λ₂/λ₁) (con
λ₁ ≥ λ₂ valores propios de la matriz de covarianza de los momentos centrales) y los siete momentos
invariantes de Hu en escala logarítmica, −sgn(h)·log₁₀|h|. El área y el perímetro se normalizan
para que no dependan de la resolución de la foto original. Una fruta golpeada o arrugada pierde
convexidad (menor solidez) y circularidad.

Antes de la regresión logística y la SVM, cada característica se estandariza con la media y la
desviación del conjunto de entrenamiento: x'_j = (x_j − μ_j)/σ_j, porque ambos modelos dependen de
distancias o de la magnitud de los coeficientes.

## III-E. Modelos y estrategia de solución

**Línea base.** Un clasificador que siempre predice la clase más frecuente de entrenamiento; fija
el mínimo que cualquier modelo debe superar.

**Regresión logística multinomial.** Modela la probabilidad de cada clase con la función softmax
y minimiza la entropía cruzada ponderada con regularización L2:

  P(y = c | x) = exp(w_cᵀx + b_c) / Σ_k exp(w_kᵀx + b_k)
  min_W  ½‖W‖² + C Σₙ w_{yₙ} [−log P(yₙ | xₙ)]        (7)

Es un modelo lineal, interpretable y rápido; sirve como referencia de qué tanto se puede separar
la calidad con fronteras lineales.

**Máquina de soporte vectorial (SVM).** Busca el hiperplano de margen máximo con margen blando:

  min_{w,b,ξ}  ½‖w‖² + C Σₙ w_{yₙ} ξₙ   s.a.  yₙ(wᵀϕ(xₙ) + b) ≥ 1 − ξₙ,  ξₙ ≥ 0        (8)

Con el kernel RBF, K(x, x') = exp(−γ‖x − x'‖²), separa clases con fronteras no lineales sin
calcular ϕ explícitamente. Para obtener probabilidades se calibró con el método de Platt.

**Bosque aleatorio (Random Forest).** Promedia B árboles de decisión, cada uno entrenado con una
muestra *bootstrap* y probando en cada nodo solo √d características al azar. Cada árbol elige la
partición que más reduce la impureza de Gini, G = 1 − Σ_c p_c². Promediar árboles poco
correlacionados reduce la varianza de un árbol individual, y la importancia de cada característica
(reducción media de impureza) permite interpretar el modelo.

**Ajuste de hiperparámetros.** Búsqueda en rejilla (*GridSearchCV*) con validación cruzada
estratificada de 5 pliegues sobre el conjunto de entrenamiento, maximizando el F1-macro:
- Regresión logística: C ∈ {0,1; 1; 10}.
- SVM: C ∈ {0,1; 1; 10}, kernel ∈ {lineal, RBF}, γ ∈ {scale, auto}.
- Random Forest: árboles ∈ {100, 250}, profundidad máxima ∈ {sin límite, 15}, mínimo de muestras
  por hoja ∈ {1, 2}.

Los tres modelos usan los pesos de clase de la ecuación (4). El mejor modelo de cada familia se
evalúa una sola vez en validación y prueba, que no participan en el ajuste.

**Alternativas descartadas.**
- *k vecinos más cercanos (KNN)*: con 23 dimensiones de escalas distintas la distancia euclidiana
  pierde poder discriminante, es sensible a las clases desbalanceadas (los vecinos de una fruta
  regular suelen ser malas) y la predicción exige comparar con todo el entrenamiento.
- *Árbol de decisión individual*: alta varianza; el bosque aleatorio lo contiene como caso
  particular y la reduce.
- *XGBoost*: rendimiento similar al bosque aleatorio en conjuntos tabulares pequeños, pero con
  muchos más hiperparámetros que ajustar en el tiempo del proyecto.

## Referencias (continuar la numeración del informe)

[10] R. M. Haralick, K. Shanmugam, and I. Dinstein, "Textural features for image classification,"
*IEEE Trans. Syst., Man, Cybern.*, vol. SMC-3, no. 6, pp. 610–621, 1973.

[11] M.-K. Hu, "Visual pattern recognition by moment invariants," *IRE Trans. Inf. Theory*, vol. 8,
no. 2, pp. 179–187, 1962.

[12] C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, vol. 20, no. 3,
pp. 273–297, 1995.

[13] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.

[14] J. Platt, "Probabilistic outputs for support vector machines and comparisons to regularized
likelihood methods," in *Advances in Large Margin Classifiers*, MIT Press, 1999, pp. 61–74.

[15] F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*,
vol. 12, pp. 2825–2830, 2011.
