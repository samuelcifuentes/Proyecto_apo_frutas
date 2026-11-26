# Evaluación de la CNN de calidad de frutas

## Resultados y sobreajuste

La CNN se entrenó desde cero sobre las particiones de la versión `3f699dd`, con cada grupo de ráfaga en una única partición. Se ponderaron las clases según su frecuencia en entrenamiento. La pérdida de validación alcanzó su mínimo de 0,3170 en la época 19. Tras cinco épocas sin mejora, el entrenamiento terminó en la época 24. Se evaluaron los pesos de la época 19 sobre imágenes sin aumento. [Particiones utilizadas](https://github.com/samuelcifuentes/Proyecto_apo_frutas/blob/3f699dd/data/splits.csv) · [Curva de pérdida](../../experiments/results/evaluation/cnn_loss.svg) · [Curva de exactitud](../../experiments/results/evaluation/cnn_accuracy.svg)

| Partición | Fotos | Exactitud | F1-macro |
|---|---:|---:|---:|
| Entrenamiento | 9.719 | 89,39 % | 0,7180 |
| Validación | 2.051 | 87,91 % | 0,6605 |
| Prueba | 1.942 | 86,66 % | 0,6398 |

La exactitud disminuyó 1,48 puntos porcentuales de entrenamiento a validación y el F1-macro bajó 0,0575. La brecha fue mayor para la clase *regular*, cuyo F1 pasó de 0,3322 a 0,1915. Esto sugiere que la CNN generaliza peor en la clase intermedia. En prueba acertó 1.683 de 1.942 imágenes. Predecir siempre *buena*, la clase mayoritaria de entrenamiento, habría dado un 50,82 % de exactitud y un F1-macro de 0,2247. [Métricas por partición](../../experiments/results/evaluation/metrics.csv)

## Errores por calidad y especie

En prueba, la CNN identificó 830 de las 987 frutas buenas, 847 de las 922 malas y **6 de las 33 regulares**. De las otras 27 regulares, clasificó 24 como malas y 3 como buenas. También marcó como regulares 31 buenas y 6 malas. Así, sólo 6 de sus 43 predicciones de *regular* fueron correctas. Para esa clase, la precisión fue 13,95 %, la sensibilidad 18,18 % y el F1 0,1579. En un puesto de venta, el error más frecuente sobre fruta regular podría llevar a descartar piezas vendibles con descuento. [Matriz de confusión](../../experiments/results/evaluation/cnn_test_confusion.svg) · [Métricas por clase](../../experiments/results/evaluation/cnn_test_report.csv)

La naranja concentró la mayor proporción de fallos (69 de 336), seguida de la manzana (61 de 335), el banano (45 de 302), la guayaba (33 de 338), el limón (23 de 279) y la granada (28 de 352). En naranja, 29 piezas malas se clasificaron como buenas. En manzana predominó el error inverso, con 37 buenas clasificadas como malas. Los seis aciertos en *regular* se repartieron entre guayaba y naranja, con tres en cada especie. [Errores por especie](../../experiments/results/evaluation/cnn_error_groups.csv)

## Comparación con la literatura

Morshed et al. [9] obtuvieron un 99,67 % de exactitud con DenseNet201 preentrenada y ajustada sobre FruitNet. Su categoría *mixed* identifica fotografías con frutas de distintas calidades. Aquí, *regular* identifica una sola pieza con defectos leves. La distinta definición de la tercera clase, el uso de preentrenamiento y la partición 60/20/20 del estudio limitan la comparación directa de las dos cifras. [Artículo de los autores](https://arxiv.org/abs/2212.04255)

## Conclusiones

La CNN superó la referencia mayoritaria, pero su desempeño en *regular* limita su uso para separar fruta con defectos leves. Las 33 imágenes regulares de prueba corresponden a 17 grupos de ráfaga. En siete de esos grupos conviven etiquetas *regular* y *mala*, por lo que revisar esas vistas de la misma pieza ayudaría a distinguir efectos del ángulo y del criterio de etiquetado. La CNN acertó en una de cinco fotos propias de prueba, una muestra insuficiente para estimar su desempeño en otro entorno de captura. Esta corrida precede a la incorporación de las fotos de Andrés. La elección del modelo final requiere las predicciones de los tres modelos clásicos sobre las mismas particiones. [Estado de la comparación](../../experiments/results/evaluation/selection.json)

## Referencia

[9] M. S. Morshed, S. Ahmed, T. Ahmed, M. U. Islam y A. B. M. A. Rahman, “Fruit Quality Assessment with Densely Connected Convolutional Neural Network”, en *2022 12th International Conference on Electrical and Computer Engineering (ICECE)*, Daca, Bangladés, 2022, pp. 1–4. [doi:10.1109/ICECE57408.2022.10088873](https://doi.org/10.1109/ICECE57408.2022.10088873)
