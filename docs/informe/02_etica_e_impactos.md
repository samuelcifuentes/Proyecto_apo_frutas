# Ética e impactos de la clasificación de frutas

## Decisiones éticas (PI1)

| Dilema de ingeniería | Principio del código de la ACM [8] | Decisión del proyecto |
|---|---|---|
| Una fotografía tomada en el puesto de venta puede identificar a quien atiende o compra. | 1.6, privacidad | La guía de captura pide encuadrar una sola fruta sobre fondo uniforme y sin manos. Si aparece una persona identificable, se debe repetir la toma antes de compartirla. |
| Una mancha superficial puede causar el rechazo de fruta que aún tiene valor comercial. | 1.2, prevención del daño | La guía reserva la clase *regular* para defectos leves y distingue las imágenes inservibles mediante la etiqueta `discard`. El operario debe revisar la fruta antes de descartarla. |
| El fondo se correlaciona con la calidad en los datos. Un productor que use otra mesa o cámara podría recibir una evaluación distinta para la misma fruta. | 1.4, trato justo, y 2.5, evaluación de riesgos | El equipo incorporó segmentación y fotografías propias. Para comprobar si persiste el sesgo, se propone fotografiar la misma pieza sobre dos fondos y contar los cambios de clase. |
| Un diámetro aparente puede influir en el precio aunque cambie al acercar la cámara. | 1.3, honestidad | La guía define el tamaño como relativo al encuadre, fija una distancia de captura de unos 30 cm y establece comparaciones por especie. Para pactar un precio por calibre físico se requiere una medida con escala. |

Samuel Lasso revisó a mano las 325 imágenes de su lote que se habían etiquetado con asistencia de IA. Primero revisó una muestra de 50, cada una de un grupo de ráfaga distinto: confirmó 10 como *regulares* y reclasificó 10 como *buenas* y 30 como *malas*. Después revisó una por una las 275 restantes: de las 271 marcadas como *regulares*, confirmó 20 y reclasificó 248 como *malas* y 7 como *buenas*. En total, solo 30 de las 321 imágenes que la IA había marcado como *regulares* conservaron esa clase, lo que muestra que la clase *regular* no puede etiquetarse sin supervisión humana. En `labels.csv`, `samuell` identifica todas estas decisiones manuales. La clase *regular* quedó con 41 imágenes, por lo que conviene ampliarla con fotos propias de frutas con defectos leves. Los tamaños de las fotos propias siguen vacíos conforme a la guía del repositorio. La sección 2.3 de la consigna pide al grupo anotar también el tamaño de esas fotos.

## Matriz de impactos potenciales (PI2)

| Dimensión | Beneficio y riesgo en un puesto de venta | Mitigación y comprobación propuestas |
|---|---|---|
| Social | Una guía común puede facilitar el acuerdo entre operarios. Una cámara dirigida al pasillo puede captar clientes y generar percepción de vigilancia. | Orientar la cámara hacia la mesa, explicar cuándo se toma la foto y comprobar el encuadre al inicio de cada turno. |
| Económica | Identificar fruta *regular* permite ofrecerla con descuento. Confundir *buena* con *mala* reduce ingresos y el error inverso puede generar devoluciones. | En un lote piloto, contrastar con inspección manual ambos errores por separado y estimar su costo por cada 100 piezas. |
| Ambiental | Recuperar fruta con defectos leves puede reducir residuos. Mantener la inferencia activa entre piezas consume energía y renovar cámaras genera residuos electrónicos. | Capturar bajo demanda con la cámara disponible. Comparar los kg descartados por lote y los Wh por cada 100 piezas con la clasificación habitual. |
| Global | La captura con celular facilita probar el sistema en otros mercados. Las variedades, los criterios de madurez y las reglas sobre imágenes de personas pueden cambiar entre regiones. | Antes de introducirlo en otra plaza, acordar criterios con productores locales, validar una muestra etiquetada allí y revisar las condiciones de uso de las imágenes. |

Las medidas de la matriz permiten evaluar un piloto futuro. El proyecto aún carece de mediciones en un puesto de venta que permitan cuantificar ahorro, consumo o reducción de desperdicio.

## Referencia

[8] Association for Computing Machinery, *Código de Ética y Conducta Profesional de ACM*, 2018. [Texto oficial en español](https://www.acm.org/binaries/content/assets/about/codigo-de-etica-y-conducta-profesional-de-acm.pdf).
