"""Extracción de características de color, textura y forma de frutas segmentadas."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycoprops

from src.data.segmentation import segment_fruit

LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
GLCM_LEVELS = 32
GLCM_OFFSETS = ((0, 1), (-1, 1), (-1, 0), (-1, -1))

FEATURE_NAMES = (
    "h_mean",
    "h_std",
    "s_mean",
    "s_std",
    "v_mean",
    "v_std",
    "texture_contrast",
    "texture_homogeneity",
    "texture_energy",
    "texture_correlation",
    "shape_area_px",
    "shape_perimeter_px",
    "shape_circularity",
    "shape_aspect_ratio",
    "shape_solidity",
    "shape_eccentricity",
    "shape_hu_1",
    "shape_hu_2",
    "shape_hu_3",
    "shape_hu_4",
    "shape_hu_5",
    "shape_hu_6",
    "shape_hu_7",
)


def _masked_glcm_features(gray: np.ndarray, mask: np.ndarray) -> tuple[float, ...]:
    """Calcula propiedades GLCM usando solo pares de píxeles dentro de la máscara."""
    quantized = (gray.astype(np.uint16) * GLCM_LEVELS // 256).astype(np.uint8)
    height, width = mask.shape
    glcm = np.zeros(
        (GLCM_LEVELS, GLCM_LEVELS, 1, len(GLCM_OFFSETS)), dtype=np.float64
    )

    for angle_index, (dy, dx) in enumerate(GLCM_OFFSETS):
        y_start, y_stop = max(0, -dy), min(height, height - dy)
        x_start, x_stop = max(0, -dx), min(width, width - dx)
        first = quantized[y_start:y_stop, x_start:x_stop]
        second = quantized[y_start + dy:y_stop + dy, x_start + dx:x_stop + dx]
        valid = (
            mask[y_start:y_stop, x_start:x_stop]
            & mask[y_start + dy:y_stop + dy, x_start + dx:x_stop + dx]
        )
        if not valid.any():
            continue

        counts = np.bincount(
            first[valid].astype(np.int32) * GLCM_LEVELS
            + second[valid].astype(np.int32),
            minlength=GLCM_LEVELS * GLCM_LEVELS,
        ).reshape(GLCM_LEVELS, GLCM_LEVELS)
        symmetric = counts + counts.T
        total = symmetric.sum()
        if total:
            glcm[:, :, 0, angle_index] = symmetric / total

    if not glcm.any():
        return (0.0, 0.0, 0.0, 0.0)

    values = tuple(
        float(np.nan_to_num(graycoprops(glcm, prop).mean(), nan=0.0))
        for prop in ("contrast", "homogeneity", "energy", "correlation")
    )
    return values


def _shape_features(mask: np.ndarray) -> tuple[float, ...]:
    """Calcula descriptores geométricos de la región principal de la máscara."""
    binary = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return (float("nan"),) * 13

    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    x, y, width, height = cv2.boundingRect(contour)
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    circularity = (
        4.0 * np.pi * contour_area / perimeter**2 if perimeter > 0 else 0.0
    )
    solidity = contour_area / hull_area if hull_area > 0 else 0.0

    moments = cv2.moments(contour)
    if moments["m00"] > 0:
        covariance = np.array(
            [
                [moments["mu20"], moments["mu11"]],
                [moments["mu11"], moments["mu02"]],
            ],
            dtype=np.float64,
        ) / moments["m00"]
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
        eccentricity = (
            float(np.sqrt(1.0 - eigenvalues[0] / eigenvalues[1]))
            if eigenvalues[1] > 0
            else 0.0
        )
    else:
        eccentricity = 0.0

    hu = cv2.HuMoments(moments).reshape(-1)
    hu_signed_log = -np.sign(hu) * np.log10(np.maximum(np.abs(hu), 1e-30))
    hu_signed_log = np.nan_to_num(hu_signed_log, nan=0.0, posinf=0.0, neginf=0.0)

    return (
        float(np.count_nonzero(mask)),
        perimeter,
        float(circularity),
        float(width / height) if height > 0 else 0.0,
        float(solidity),
        eccentricity,
        *(float(value) for value in hu_signed_log),
    )


def extract_features(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Extrae características de una imagen BGR y su máscara binaria.

    El color HSV y las parejas GLCM se calculan solo en los píxeles de la fruta.
    Las características de forma se calculan sobre la región externa principal.
    Una máscara vacía produce un vector de NaN y genera un aviso en el log.

    Args:
        image: Imagen BGR de OpenCV con forma (alto, ancho, 3).
        mask: Máscara binaria alineada con la imagen; cualquier valor distinto de
            cero se considera parte de la fruta.

    Returns:
        Vector float64 ordenado según ``FEATURE_NAMES``.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image debe ser una imagen BGR de tres canales")
    if mask.ndim == 3:
        if mask.shape[2] != 1:
            raise ValueError("mask debe ser bidimensional o tener un solo canal")
        mask = mask[:, :, 0]
    if mask.ndim != 2 or mask.shape != image.shape[:2]:
        raise ValueError("mask debe tener las mismas dimensiones que image")

    binary_mask = mask > 0
    if not binary_mask.any():
        LOGGER.warning("No se extrajeron características: máscara vacía")
        return np.full(len(FEATURE_NAMES), np.nan, dtype=np.float64)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] /= 180.0
    hsv[..., 1:] /= 255.0
    color_features = tuple(
        float(value)
        for channel in range(3)
        for value in (
            hsv[..., channel][binary_mask].mean(),
            hsv[..., channel][binary_mask].std(),
        )
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    texture_features = _masked_glcm_features(gray, binary_mask)
    shape_features = _shape_features(binary_mask)

    return np.asarray(
        (*color_features, *texture_features, *shape_features), dtype=np.float64
    )


def extract_features_from_dataframe(
    dataframe: pd.DataFrame,
    output_csv: str | Path,
    image_column: str = "path",
    segmentation_method: str = "rembg",
) -> pd.DataFrame:
    """Segmenta imágenes referenciadas por un DataFrame y guarda sus características.

    Las demás columnas de entrada (por ejemplo, ``quality`` y ``split``) se
    conservan para facilitar la preparación posterior de los datos de modelos.
    Las rutas que no se puedan leer se registran y reciben características NaN.
    """
    if image_column not in dataframe.columns:
        raise KeyError(f"No existe la columna de imágenes {image_column!r}")

    rows: list[dict[str, Any]] = []
    for _, source_row in dataframe.iterrows():
        row = source_row.to_dict()
        image_path = Path(str(row[image_column]))
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            LOGGER.warning("No se pudo leer la imagen: %s", image_path)
            features = np.full(len(FEATURE_NAMES), np.nan, dtype=np.float64)
            segmentation_ok = False
        else:
            segmentation = segment_fruit(image, method=segmentation_method)
            segmentation_ok = bool(segmentation.get("ok", False))
            if not segmentation_ok:
                LOGGER.warning("Falló la segmentación: %s", image_path)
            features = extract_features(image, segmentation["mask"])

        row["image_path"] = str(image_path)
        row["segmentation_ok"] = segmentation_ok
        row.update(dict(zip(FEATURE_NAMES, features)))
        rows.append(row)

    result = pd.DataFrame(rows)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def extract_features_from_folder(
    image_dir: str | Path,
    output_csv: str | Path,
    segmentation_method: str = "rembg",
) -> pd.DataFrame:
    """Procesa recursivamente las imágenes de una carpeta y guarda un CSV."""
    root = Path(image_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"No existe la carpeta de imágenes: {root}")
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    dataframe = pd.DataFrame({"image_path": [str(path) for path in paths]})
    return extract_features_from_dataframe(
        dataframe,
        output_csv,
        image_column="image_path",
        segmentation_method=segmentation_method,
    )
