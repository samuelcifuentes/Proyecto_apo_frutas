"""Estimación del tamaño de una fruta a partir de su máscara binaria."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import cv2
import numpy as np

from src.data.segmentation import segment_fruit

LOGGER = logging.getLogger(__name__)
DEFAULT_THRESHOLDS = (1.0 / 3.0, 2.0 / 3.0)


def fit_size_thresholds(training_diameters: Sequence[float]) -> tuple[float, float]:
    """Calcula los límites de tamaño pequeño/mediano/grande con terciles de train.

    Los valores de entrada deben ser diámetros normalizados calculados sobre el
    conjunto de entrenamiento (preferiblemente agrupados por tipo de fruta).
    """
    values = np.asarray(training_diameters, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Se necesita al menos un diámetro normalizado válido")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Los diámetros normalizados deben estar entre 0 y 1")
    lower, upper = np.quantile(values, (1.0 / 3.0, 2.0 / 3.0))
    return float(lower), float(upper)


def fit_size_thresholds_by_fruit(
    fruits: Sequence[str], training_diameters: Sequence[float]
) -> dict[str, tuple[float, float]]:
    """Terciles (P33, P66) del diámetro normalizado de train, calculados POR FRUTA.

    Un limón grande sigue siendo más pequeño que una manzana pequeña, así que los
    límites pequeño/mediano/grande deben depender de la especie
    (ver docs/guia_etiquetado.md, sección 2).
    """
    fruits = np.asarray(fruits)
    values = np.asarray(training_diameters, dtype=np.float64)
    return {
        str(fruit): fit_size_thresholds(values[fruits == fruit])
        for fruit in np.unique(fruits)
    }


def save_size_thresholds(thresholds: dict[str, tuple[float, float]], path) -> None:
    """Guarda los límites por fruta en JSON (los usa la app)."""
    import json
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({k: list(v) for k, v in thresholds.items()}, indent=2))


def load_size_thresholds(path) -> dict[str, tuple[float, float]]:
    """Lee los límites por fruta guardados con save_size_thresholds."""
    import json
    from pathlib import Path

    return {k: tuple(v) for k, v in json.loads(Path(path).read_text()).items()}


def _validate_thresholds(thresholds: tuple[float, float]) -> None:
    """Valida que los umbrales estén ordenados dentro del rango normalizado."""
    if len(thresholds) != 2:
        raise ValueError("thresholds debe contener dos valores")
    lower, upper = thresholds
    if not (0.0 <= lower <= upper <= 1.0):
        raise ValueError("Los umbrales deben cumplir 0 <= t1 <= t2 <= 1")


def estimate_size(
    image: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    thresholds: tuple[float, float] | None = None,
    training_diameters: Sequence[float] | None = None,
    cm_per_pixel: float | None = None,
    segmentation_method: str = "rembg",
) -> dict[str, object]:
    """Estima área y diámetro de una fruta, normalizados y opcionalmente calibrados.

    Se puede proporcionar la imagen BGR, la máscara binaria, o ambas. Si solo se
    da la imagen, se usa ``segment_fruit``. Los umbrales se pueden obtener con
    :func:`fit_size_thresholds` a partir de los diámetros del conjunto de train.
    ``training_diameters`` hace que los terciles de train se calculen directamente;
    también se pueden pasar límites ya ajustados con ``thresholds``. Sin datos de
    train ni límites configurados, se usan terciles iguales como valor inicial.
    Sin calibración física, los tamaños absolutos se expresan en píxeles.

    Args:
        image: Imagen BGR de OpenCV. Opcional si se proporciona ``mask``.
        mask: Máscara binaria; cualquier valor distinto de cero es fruta.
        thresholds: Límites ``(P33, P66)`` configurados para clasificar el diámetro.
        training_diameters: Diámetros normalizados del train para calcular los
            terciles de clasificación automáticamente.
        cm_per_pixel: Escala opcional de centímetros por píxel, medida con una
            referencia física.
        segmentation_method: Método enviado a ``segment_fruit`` si falta la máscara.

    Returns:
        Diccionario con estado, área, diámetro equivalente en píxeles y
        normalizado, categoría, caja delimitadora y medidas físicas opcionales.
        En una máscara vacía las medidas son ``None`` y la categoría es
        ``"no_disponible"``.
    """
    if image is None and mask is None:
        raise ValueError("Se debe proporcionar image, mask o ambos")
    if thresholds is not None and training_diameters is not None:
        raise ValueError("Usa thresholds o training_diameters, pero no ambos")
    if training_diameters is not None:
        thresholds = fit_size_thresholds(training_diameters)
        threshold_source = "terciles_train"
    elif thresholds is not None:
        threshold_source = "configurados"
    else:
        thresholds = DEFAULT_THRESHOLDS
        threshold_source = "terciles_iguales_iniciales"
    _validate_thresholds(thresholds)
    if cm_per_pixel is not None and (
        not np.isfinite(cm_per_pixel) or cm_per_pixel <= 0
    ):
        raise ValueError("cm_per_pixel debe ser mayor que cero")
    if image is not None and (image.ndim != 3 or image.shape[2] != 3):
        raise ValueError("image debe ser una imagen BGR de tres canales")

    if mask is None:
        assert image is not None
        segmentation = segment_fruit(image, method=segmentation_method)
        mask = segmentation["mask"]
    elif mask.ndim == 3:
        if mask.shape[2] != 1:
            raise ValueError("mask debe ser bidimensional o tener un solo canal")
        mask = mask[:, :, 0]

    if mask.ndim != 2:
        raise ValueError("mask debe ser bidimensional")
    if image is not None and mask.shape != image.shape[:2]:
        raise ValueError("mask debe tener las mismas dimensiones que image")

    binary_mask = mask > 0
    if not binary_mask.any():
        LOGGER.warning("No se pudo estimar el tamaño: máscara vacía")
        return {
            "ok": False,
            "area_px": None,
            "diameter_px": None,
            "diameter_normalized": None,
            "category": "no_disponible",
            "thresholds": thresholds,
            "threshold_source": threshold_source,
            "bbox": None,
            "area_cm2": None,
            "diameter_cm": None,
            "bbox_cm": None,
        }

    area_px = int(np.count_nonzero(binary_mask))
    diameter_px = float(np.sqrt(4.0 * area_px / np.pi))
    side_length = min(binary_mask.shape)
    diameter_normalized = min(1.0, diameter_px / side_length)
    lower, upper = thresholds
    if diameter_normalized < lower:
        category = "pequeño"
    elif diameter_normalized < upper:
        category = "mediano"
    else:
        category = "grande"

    x, y, width, height = cv2.boundingRect(binary_mask.astype(np.uint8))
    bbox = (int(x), int(y), int(width), int(height))
    result: dict[str, object] = {
        "ok": True,
        "area_px": area_px,
        "diameter_px": diameter_px,
        "diameter_normalized": float(diameter_normalized),
        "category": category,
        "thresholds": thresholds,
        "threshold_source": threshold_source,
        "bbox": bbox,
        "area_cm2": None,
        "diameter_cm": None,
        "bbox_cm": None,
    }

    if cm_per_pixel is not None:
        result.update(
            {
                "area_cm2": float(area_px * cm_per_pixel**2),
                "diameter_cm": float(diameter_px * cm_per_pixel),
                "bbox_cm": (
                    float(width * cm_per_pixel),
                    float(height * cm_per_pixel),
                ),
            }
        )
    return result
