# Comparación de modelos y selección del modelo final

> Borrador para la plantilla IEEE (sección de Resultados y Análisis). Complementa
> `04_resultados_cnn.md`. Cifras de `experiments/results/evaluation/` (métricas, matrices de
> confusión, `selection.json`) y `src/models/saved/*_best_params.json`. ⟳ = actualizar si cambian
> las etiquetas *regular*.

## Resultados

Todos los modelos usan las mismas particiones (agrupadas por ráfaga) y los mismos pesos de clase.
Los hiperparámetros de los modelos clásicos se eligieron con validación cruzada estratificada
**por grupos** de 5 pliegues; el modelo final se eligió por F1-macro en **validación**, y el
conjunto de prueba se usó una sola vez para reportar el desempeño. ⟳

**Tabla I.** Exactitud y F1-macro por partición.

| Modelo | Train F1 | Val F1 | Test exactitud | Test F1-macro | F1 *regular* (test) |
|---|---:|---:|---:|---:|---:|
| Línea base (clase mayoritaria) | 0,223 | 0,224 | 50,8 % | 0,225 | 0,00 |
| Regresión logística (C = 10) | 0,543 | 0,500 | 65,1 % | 0,516 | 0,10 |
| SVM (RBF, C = 10) | 0,897 | 0,698 | 88,8 % | 0,648 | 0,13 |
| **Random Forest** (100 árboles, prof. 15) | 0,967 | **0,774** | **90,0 %** | 0,638 | 0,10 |
| CNN desde cero | 0,718 | 0,661 | 86,7 % | 0,640 | 0,16 |

[Métricas](../../experiments/results/evaluation/metrics.csv) ·
[Comparación](../../experiments/results/evaluation/comparison.svg) ·
[Sobreajuste](../../experiments/results/evaluation/overfitting.csv) ·
[Importancia de características](../../experiments/results/figures/random_forest_feature_importance.svg)

## Análisis

**Todos los modelos aprenden algo útil.** Los cuatro superan ampliamente la línea base
(F1-macro 0,225). SVM, Random Forest y CNN separan bien la fruta buena de la mala: F1 de 0,88 a
0,92 en ambas clases y exactitud cercana al 90 %.

**La clase *regular* es el cuello de botella.** Ningún modelo supera un F1 de 0,16 en esa clase
(33 imágenes en prueba). Lo que más la limita no es el modelo sino los datos: es la clase más
pequeña (2 % del entrenamiento), su límite con *mala* es subjetivo, y la revisión manual de una
muestra obligó a corregir 40 de 50 etiquetas. Además, en 7 de las 17 ráfagas de prueba conviven
etiquetas *regular* y *mala* para la misma fruta. La regresión logística tiene la mayor
sensibilidad en *regular* (0,73), pero a costa de una precisión de 0,06: marca como regulares
cientos de frutas buenas y malas, porque una frontera lineal no separa esa clase.

**Sobreajuste.** La brecha de F1-macro entre entrenamiento y validación es de 0,19 en Random Forest
y 0,20 en la SVM, frente a 0,06 en la CNN y 0,04 en la regresión logística. Los modelos con
mayor capacidad memorizan parte del entrenamiento, sobre todo en *regular*.

**Efecto de la fuga por ráfagas en la validación cruzada.** En una primera corrida la validación
cruzada no agrupaba las ráfagas: fotos casi idénticas quedaban en entrenamiento y validación del
mismo pliegue. El F1 de validación cruzada fue de 0,81–0,83, muy por encima del 0,70–0,74 obtenido
luego en la partición de validación real, y la búsqueda eligió un Random Forest sin límite de
profundidad (F1 de 0,998 en entrenamiento). Al agrupar por ráfaga, el F1 de validación cruzada
(0,72 para Random Forest y 0,71 para SVM) quedó en línea con la validación real y la búsqueda
eligió un bosque más regularizado (profundidad 15, mínimo 2 muestras por hoja), que mejoró el F1
de prueba de 0,618 a 0,638. Esto muestra que, con imágenes tomadas en ráfaga, una validación sin
grupos sobreestima el desempeño y lleva a elegir modelos que memorizan.

**Qué características usa el modelo.** En el Random Forest las características más importantes son
de color: la media del tono H (0,14), la media de la saturación S (0,09), la desviación del tono
(0,09) y la media del brillo V (0,06), seguidas del perímetro relativo y la homogeneidad de la
textura (≈0,05). Coincide con el EDA, donde la saturación separaba buenas de malas en las seis
frutas.

**Errores por especie.** El limón (47 errores en prueba) y la naranja (43) concentran más fallos;
en ambos los defectos son manchas pequeñas sobre una cáscara de color parecido.

**Comparación con la literatura.** Morshed *et al.* [9] reportan 99,67 % de exactitud en FruitNet
con DenseNet201 preentrenada, pero su tercera clase (*mixed*) son fotos con varias frutas, usan
preentrenamiento y una partición aleatoria que no separa ráfagas. Nuestra partición por grupos
es más exigente: evita evaluar con vistas casi idénticas de frutas ya vistas.

## Selección del modelo final

Por F1-macro de validación el mejor modelo es **Random Forest** (0,774), seguido de la SVM
(0,698), la CNN (0,661) y la regresión logística (0,500). En prueba, SVM, CNN y Random Forest
quedan prácticamente empatados (0,648, 0,640 y 0,638); con solo 33 imágenes *regular*, una o dos
predicciones cambian el F1-macro en varias centésimas, así que esas diferencias no son
concluyentes. Se mantiene el **Random Forest** como modelo final porque fue el elegido por el
criterio definido de antemano (validación), tiene la mayor exactitud de prueba (90,0 %) y es
interpretable mediante la importancia de sus características. Su principal limitación es la clase
*regular*, que debe mejorarse con más ejemplos y etiquetas revisadas antes que con otro
modelo. ⟳

**Fotos propias.** De las 5 fotos propias que cayeron en prueba, el Random Forest acertó 3; la
muestra es demasiado pequeña para estimar el desempeño en un entorno de captura distinto.
