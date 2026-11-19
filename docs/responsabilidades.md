# Responsabilidades del equipo

✅ **Ya hecho:** estructura del repositorio, dataset en `data/raw/`, guía de etiquetado (`docs/guia_etiquetado.md`), EDA (`notebooks/01_eda.ipynb`) y segmentación v1 (`src/data/segmentation.py`).

## Samuel Cifuentes — Preparación de datos (CRISP-DM fase 3)
- Segmentación de la fruta (`src/data/segmentation.py`) — usada por todo el equipo.
- Recortes individuales de las frutas de `Mixed Qualit_Fruits`.
- Preprocesamiento: redimensionar, normalizar, balancear (granada buena ≈5,5× más imágenes) y dividir en train/val/test (`src/data/preprocess.py`).
- **Informe:** introducción, comprensión del negocio y sección de datos/EDA.

## Juan Camilo — Machine learning clásico (fase 4)
- Extracción de características **sobre la fruta segmentada**: histogramas HSV, textura, forma (`src/features/`).
- Estimación de tamaño (diámetro normalizado → terciles por fruta).
- Entrenar **SVM y Random Forest** (línea base: regresión logística) con GridSearch + validación cruzada.
- **Informe:** formulación matemática (PI3), fundamentos teóricos y justificación de modelos / alternativas descartadas.

## Andrés Fajardo — Deep learning + evaluación (fases 4 y 5)
- **CNN desde cero** (2–3 capas conv + pooling + densas); curvas de pérdida y precisión.
- Comparación de todos los modelos: accuracy, F1-macro, matriz de confusión, análisis de errores y sobreajuste (`src/evaluation/`).
- **Informe:** resultados, análisis de resultados y conclusiones.

## Samuel Lasso — Etiquetado, ética y despliegue (fase 6)
- Re-etiquetar **300–500 imágenes como "Regular"** a partir de `Bad Quality_Fruits` y recortes de `Mixed`, siguiendo la guía. Mantener `data/own_images/labels.csv`.
- **App en Streamlit**: subir una foto o usar la cámara y ver calidad + tamaño (`app/`).
- **Informe:** tabla de dilemas éticos vs. código de ética (PI1) y matriz de impactos social, económico, ambiental y global con mitigación (PI2).

## Todos
- ~12 fotos propias cada uno (≈48 en total), siguiendo `docs/guia_etiquetado.md`.
- Diagrama CRISP-DM, referencias IEEE, video (máx. 10 min) y revisión final del informe.

> **Orden de dependencias:** la segmentación y el preprocesamiento (Samuel C.) van primero; Juan Camilo y Andrés los necesitan para entrenar. Mientras tanto, cada uno puede avanzar en su sección del informe.
