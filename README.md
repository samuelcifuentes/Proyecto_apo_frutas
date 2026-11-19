# Clasificación de calidad de frutas con visión por computador

Proyecto final de **Algoritmos y Programación III** — Universidad Icesi, 2026-2.

#### Estado del proyecto: Activo

## Integrantes

| Nombre | Rol |
|--------|-----|
| Samuel Cifuentes | Datos: EDA, segmentación, recortes y preprocesamiento |
| Juan Camilo | Extracción de características, estimación de tamaño y modelos de ML clásico |
| Andrés Fajardo | CNN, evaluación y comparación de modelos |
| Samuel Lasso | Etiquetado de la clase "Regular", ética e impactos, despliegue (Streamlit) |

El detalle de las responsabilidades está en [`docs/responsabilidades.md`](docs/responsabilidades.md).

## Objetivo
Construir un sistema que, a partir de la foto de una fruta individual sobre fondo uniforme,
prediga su **categoría de calidad** (3 o más clases) y estime su **tamaño relativo**
(pequeño / mediano / grande), siguiendo la metodología CRISP-DM.

## Datos
- **Dataset base:** [Fruit Quality Classification (Kaggle)](https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification) → descargar en `data/raw/` (no se sube al repo).
- **Imágenes propias:** 30–50 fotos recolectadas y etiquetadas por el grupo → `data/own_images/`.

## Métodos
- Modelos de ML: SVM, Random Forest (línea base: regresión logística) sobre características de color, textura y forma.
- Deep learning: CNN entrenada desde cero.
- Despliegue: aplicación en Streamlit con carga de imagen y cámara.

## Estructura del repositorio
```
data/            raw/ (Kaggle), own_images/ (propias), processed/
notebooks/       EDA y experimentos
src/data/        carga y preprocesamiento
src/features/    extracción de características y estimación de tamaño
src/models/      definición de modelos (ML y CNN)
src/training/    entrenamiento y ajuste de hiperparámetros
src/evaluation/  métricas, matrices de confusión, comparación
app/             interfaz Streamlit
experiments/     resultados, modelos guardados, logs
docs/            informe IEEE, diagramas, video
tests/           pruebas
```

## Cómo ejecutar
```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Entregables
- Informe IEEE (máx. 7 páginas) — `docs/`
- Video (máx. 10 min) — enlace por definir
