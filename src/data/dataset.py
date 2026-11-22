"""
Carga de los datos ya preprocesados — lo que usan los modelos.

Responsable: Samuel Cifuentes. Usado por: Juan Camilo (ML clásico) y Andrés (CNN).

Requisito: haber corrido antes `python -m src.data.preprocess`.

Ejemplo
-------
    from src.data.dataset import load_split, normalize, QUALITY_TO_ID

    X_train, y_train, meta_train = load_split("train")   # X: uint8 (N, 128, 128, 3) RGB
    X_val,   y_val,   meta_val   = load_split("val")
    X_train_n = normalize(X_train)                        # float32 z-score por canal

    meta_train["fruit"]          # fruta de cada imagen
    meta_train["path"]           # imagen ORIGINAL (para segmentar o estimar tamaño)

Etiquetas de calidad (fijas, aunque en la versión provisional no haya "regular"):
    good = 0, regular = 1, bad = 2
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SPLITS_CSV = Path("data/splits.csv")
STATS_JSON = Path("data/processed/stats.json")

QUALITY_TO_ID = {"good": 0, "regular": 1, "bad": 2}
ID_TO_QUALITY = {v: k for k, v in QUALITY_TO_ID.items()}
FRUITS = ["apple", "banana", "guava", "lime", "orange", "pomegranate"]


def load_splits() -> pd.DataFrame:
    """Tabla completa: path, fruit, quality, source, split, processed_path."""
    return pd.read_csv(SPLITS_CSV)


def load_split(split: str, fruits: list[str] | None = None):
    """
    Carga un subconjunto ('train', 'val' o 'test') en memoria.

    Devuelve (X, y, meta):
      X    → np.uint8 de forma (N, 128, 128, 3), en RGB
      y    → np.int64 de forma (N,), etiquetas de calidad (ver QUALITY_TO_ID)
      meta → DataFrame con fruit, path original, etc. (mismo orden que X)
    """
    df = load_splits()
    df = df[df["split"] == split]
    if fruits:
        df = df[df["fruit"].isin(fruits)]
    df = df.reset_index(drop=True)
    X = np.stack([cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in df["processed_path"]])
    y = df["quality"].map(QUALITY_TO_ID).to_numpy(np.int64)
    return X, y, df


def normalize(X: np.ndarray) -> np.ndarray:
    """
    Normalización z-score por canal:  x' = (x/255 − μ) / σ
    μ y σ se calcularon SOLO con train (data/processed/stats.json) para evitar fuga
    de información de val/test.
    """
    stats = json.loads(STATS_JSON.read_text())
    mean = np.array(stats["mean_rgb"], np.float32)
    std = np.array(stats["std_rgb"], np.float32)
    return (X.astype(np.float32) / 255.0 - mean) / std


def augment(img: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Aumento de datos para UNA imagen uint8 RGB (usar solo en train):
      - volteo horizontal y vertical aleatorio (una fruta volteada sigue siendo la misma)
      - rotación aleatoria de 0–360° (la fruta no tiene una orientación "correcta")
      - cambio leve de brillo (±15 %) para simular distinta iluminación
    No cambia el tono (H): el color es información clave para la calidad.
    """
    rng = rng or np.random.default_rng()
    if rng.random() < 0.5:
        img = img[:, ::-1]
    if rng.random() < 0.5:
        img = img[::-1, :]
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), float(rng.uniform(0, 360)), 1.0)
    img = cv2.warpAffine(np.ascontiguousarray(img), M, (w, h), borderValue=(0, 0, 0))
    img = np.clip(img.astype(np.float32) * rng.uniform(0.85, 1.15), 0, 255).astype(np.uint8)
    return img
