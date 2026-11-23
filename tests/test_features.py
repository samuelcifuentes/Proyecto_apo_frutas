"""Pruebas rápidas de extracción de características y tamaño."""

import numpy as np

from src.features import (
    FEATURE_NAMES,
    estimate_size,
    extract_features,
    fit_size_thresholds,
)


def _synthetic_image_and_mask() -> tuple[np.ndarray, np.ndarray]:
    """Construye una fruta rectangular sobre un fondo oscuro uniforme."""
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[16:48, 18:46] = (20, 120, 230)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 18:46] = 255
    return image, mask


def test_extract_features_returns_finite_vector_for_masked_fruit() -> None:
    """Las características tienen longitud estable y son finitas para una máscara."""
    image, mask = _synthetic_image_and_mask()

    features = extract_features(image, mask)

    assert features.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(features).all()


def test_extract_features_handles_empty_mask() -> None:
    """Una máscara vacía se registra como características NaN, sin excepción."""
    image, mask = _synthetic_image_and_mask()

    features = extract_features(image, np.zeros_like(mask))

    assert np.isnan(features).all()


def test_estimate_size_returns_pixel_and_calibrated_measurements() -> None:
    """La estimación respeta área, caja delimitadora, normalización y calibración."""
    _, mask = _synthetic_image_and_mask()

    result = estimate_size(mask=mask, thresholds=(0.25, 0.75), cm_per_pixel=0.1)

    assert result["ok"] is True
    assert result["area_px"] == 32 * 28
    assert result["bbox"] == (18, 16, 28, 32)
    assert 0.0 <= result["diameter_normalized"] <= 1.0
    assert result["category"] == "mediano"
    assert result["diameter_cm"] is not None


def test_estimate_size_and_thresholds_handle_empty_or_training_data() -> None:
    """La máscara vacía es segura y los límites se ajustan con los terciles de train."""
    _, mask = _synthetic_image_and_mask()
    empty_result = estimate_size(mask=np.zeros_like(mask))
    train_result = estimate_size(
        mask=mask, training_diameters=[0.1, 0.2, 0.3, 0.6, 0.9]
    )
    thresholds = fit_size_thresholds([0.1, 0.2, 0.3, 0.6, 0.9])

    assert empty_result["ok"] is False
    assert empty_result["category"] == "no_disponible"
    assert np.allclose(thresholds, (0.2333333333, 0.5))
    assert train_result["threshold_source"] == "terciles_train"
    assert np.allclose(train_result["thresholds"], thresholds)
