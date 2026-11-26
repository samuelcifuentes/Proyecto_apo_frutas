# CRISP-DM aplicado a la clasificación de frutas

```mermaid
flowchart TD
    N["Comprensión del negocio<br/>Distinguir fruta buena, regular y mala para orientar la clasificación en el puesto de venta"]
    D["Comprensión de los datos<br/>Examinar el desbalance por especie y la relación entre el fondo y la clase"]
    P["Preparación de los datos<br/>Recortar las fotos Mixed, revisar etiquetas y generar las particiones"]
    M["Modelado<br/>Entrenar regresión logística, SVM, Random Forest y CNN"]
    E["Evaluación pendiente<br/>Comparar predicciones con F1-macro, exactitud y matrices de confusión"]
    A["Prototipo actual de Streamlit<br/>Probar carga de fotos y cámara con un modelo de demostración"]
    F["Despliegue final pendiente<br/>Conectar el modelo seleccionado y mostrar el tamaño relativo"]

    N --> D --> P --> M --> E
    P --> A
    E -->|"Modelo seleccionado"| F
    A --> F
    E -->|"Si los errores revelan etiquetas dudosas, volver a revisarlas"| P
```

**Evidencia en el repositorio.** El planteamiento del negocio y el EDA explican la clase *regular* y los posibles efectos del fondo. La preparación aplica una guía común a las fotos propias y al conjunto público. [Planteamiento](01_introduccion_negocio_datos.md) · [EDA](../../notebooks/01_eda.ipynb) · [Guía de etiquetado](../guia_etiquetado.md) · [Preprocesamiento](../../src/data/preprocess.py)

Los programas de entrenamiento y comparación definen cómo se obtendrán las predicciones y métricas. El prototipo permite probar la interacción mientras se selecciona el modelo. [Modelos clásicos](../../src/models/train_classical.py) · [Entrenamiento de la CNN](../../src/training/train_cnn.py) · [Evaluación](../../src/evaluation/compare_models.py) · [Interfaz](../../app/app.py)

**Revisión de las particiones.** En `Lime_Bad`, `IMG_7279.JPG` quedó en entrenamiento y `IMG_7280.JPG` en prueba. Las dos fotos muestran el mismo limón con las mismas manchas y un leve cambio de ángulo. Repartir una ráfaga entre particiones puede elevar el desempeño aparente sobre frutas nuevas. Antes de interpretar las métricas, conviene identificar estas secuencias, asignar cada ejemplar a una sola partición y repetir la evaluación. [Manifiesto de particiones](../../data/splits.csv)
