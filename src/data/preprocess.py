"""
Preprocesamiento del dataset: índice → balanceo → división train/val/test →
imágenes redimensionadas → estadísticas de normalización.

Responsable: Samuel Cifuentes — Fase 3 CRISP-DM (preparación de datos).

Uso (desde la raíz del repo):
    python -m src.data.preprocess                      # todo
    python -m src.data.preprocess --max-seconds 150    # por partes (reanudable)

Entradas
  data/raw/Good Quality_Fruits, data/raw/Bad Quality_Fruits   (Kaggle)
  data/own_images/labels.csv   (opcional) fotos propias, recortes de Mixed etiquetados
                               y re-etiquetados como "regular". Si existe, se une solo.

Salidas
  data/splits.csv              índice final: path, fruit, quality, source, split, processed_path
                               (se sube al repo para que TODOS usen la misma división)
  data/processed/images/...    imágenes 128x128 (no se suben; cada uno las genera)
  data/processed/stats.json    media y desviación por canal (calculadas SOLO con train)

Decisiones (justificadas en el EDA)
  * Balanceo: máximo MAX_PER_CLASS imágenes por (fruta, calidad). Granada buena tenía
    5.940 imágenes (≈5,5× la clase más pequeña) → submuestreo aleatorio.
  * División estratificada 70/15/15 por (fruta, calidad), estable (por hash de la ruta):
    agregar imágenes nuevas no reordena las que ya estaban.
  * Redimensionamiento con "letterbox": se escala el lado mayor a 128 px y se rellena
    con negro, para NO deformar la fruta (la forma importa para la calidad y el tamaño).
  * Normalización z-score por canal: x' = (x/255 − μ) / σ, con μ y σ de train
    (usar datos de val/test para calcularlas sería fuga de información).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

RAW = Path("data/raw")
LABELS_CSV = Path("data/own_images/labels.csv")
SPLITS_CSV = Path("data/splits.csv")
PROC_DIR = Path("data/processed")
IMG_SIZE = 128
MAX_PER_CLASS = 1200
SPLIT_FRACS = (0.70, 0.15, 0.15)
SEED = 42

KAGGLE = {"Good Quality_Fruits": "good", "Bad Quality_Fruits": "bad"}
FRUIT = {"Apple": "apple", "Banana": "banana", "Guava": "guava", "Lime": "lime",
         "Lemon": "lime", "Orange": "orange", "Pomegranate": "pomegranate"}
EXTS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# 1. Índice
# ---------------------------------------------------------------------------
def build_index() -> pd.DataFrame:
    """Una fila por imagen: path (relativo a la raíz del repo), fruit, quality, source."""
    rows = []
    for folder, quality in KAGGLE.items():
        for sub in sorted((RAW / folder).iterdir()):
            if sub.is_dir():
                fruit = FRUIT[sub.name.split("_")[0]]
                for f in sorted(sub.iterdir()):
                    if f.suffix.lower() in EXTS:
                        rows.append((f.as_posix(), fruit, quality, "kaggle"))
    df = pd.DataFrame(rows, columns=["path", "fruit", "quality", "source"])

    if LABELS_CSV.exists():
        lab = pd.read_csv(LABELS_CSV)
        lab = lab.rename(columns={"filename": "path"})
        # los re-etiquetados reemplazan la etiqueta original de Kaggle (misma ruta);
        # las marcadas como "discard" se eliminan del todo
        df = df[~df["path"].isin(lab["path"])]
        lab = lab[lab["quality"].isin(["good", "regular", "bad"])]
        df = pd.concat([df, lab[["path", "fruit", "quality", "source"]]], ignore_index=True)
        print(f"labels.csv: {len(lab)} filas unidas "
              f"({(lab.quality == 'regular').sum()} regular)")
    else:
        print("labels.csv no existe todavía → versión PROVISIONAL (solo good/bad de Kaggle)")
    return df.sort_values("path").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Balanceo y 3. división
# ---------------------------------------------------------------------------
def balance(df: pd.DataFrame) -> pd.DataFrame:
    """Submuestreo: como máximo MAX_PER_CLASS imágenes por (fruta, calidad)."""
    return (df.sample(frac=1.0, random_state=SEED)
              .groupby(["fruit", "quality"]).head(MAX_PER_CLASS)
              .sort_values("path").reset_index(drop=True))


def _hash01(path: str) -> float:
    """Número pseudoaleatorio en [0, 1) que depende SOLO de la ruta de la imagen."""
    return int(hashlib.md5(f"{SEED}:{path}".encode()).hexdigest()[:8], 16) / 16**8


def split(df: pd.DataFrame) -> pd.DataFrame:
    """
    División train/val/test estratificada por (fruta, calidad) y ESTABLE.

    Dentro de cada grupo (fruta, calidad) ordenamos las imágenes por un hash de su ruta y
    asignamos el primer 70 % a train, el siguiente 15 % a val y el resto a test.
      * Estratificada: cada grupo se reparte en las mismas proporciones.
      * Estable: agregar fotos nuevas (o etiquetas "regular") NO cambia el split de las
        imágenes que ya estaban; solo se mueven unas pocas en el borde de cada grupo.
      * Funciona con grupos pequeños (p. ej. 2 fotos propias), donde train_test_split falla.
    En grupos de 1–2 imágenes van todas a train.
    """
    df = df.copy()
    df["_h"] = df["path"].map(_hash01)
    df["split"] = "train"
    for _, g in df.groupby(["fruit", "quality"]):
        idx = g.sort_values("_h").index
        n = len(idx)
        if n < 3:
            continue
        n_train = round(n * SPLIT_FRACS[0])
        n_val = max(1, round(n * SPLIT_FRACS[1]))
        df.loc[idx[n_train:n_train + n_val], "split"] = "val"
        df.loc[idx[n_train + n_val:], "split"] = "test"
    return df.drop(columns="_h")


# ---------------------------------------------------------------------------
# 4. Redimensionamiento
# ---------------------------------------------------------------------------
def letterbox(img: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Escala el lado mayor a `size` y rellena con negro hasta size×size (sin deformar)."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
                     interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    top, left = (size - h) // 2, (size - w) // 2
    return cv2.copyMakeBorder(img, top, size - h - top, left, size - w - left,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


def processed_path(row) -> str:
    h = hashlib.sha1(row["path"].encode()).hexdigest()[:12]
    return (PROC_DIR / "images" / row["split"] / row["quality"] / f"{row['fruit']}_{h}.jpg").as_posix()


def _existing_files() -> set[str]:
    """Lista de una vez las imágenes ya procesadas (mucho más rápido que .exists() una por una)."""
    import os
    root = PROC_DIR / "images"
    return {Path(d, f).as_posix() for d, _, fs in os.walk(root) for f in fs}


def resize_all(df: pd.DataFrame, max_seconds: float | None) -> bool:
    """Genera las imágenes 128x128 que falten. Devuelve True si terminó todas."""
    t0 = time.time()
    existing = _existing_files()
    for stale in existing - set(df["processed_path"]):   # sobrantes de una corrida anterior
        Path(stale).unlink(missing_ok=True)
    for p_in, p_out in zip(df["path"], df["processed_path"]):
        out = Path(p_out)
        if p_out in existing:
            continue
        if max_seconds and time.time() - t0 > max_seconds:
            return False
        img = cv2.imread(p_in, cv2.IMREAD_REDUCED_COLOR_4 if Path(p_in).stat().st_size > 2e6
                         else cv2.IMREAD_COLOR)
        if img is None:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), letterbox(img), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return True


# ---------------------------------------------------------------------------
# 5. Estadísticas de normalización
# ---------------------------------------------------------------------------
def compute_stats(df: pd.DataFrame, n_sample: int = 1500) -> dict:
    """
    Media y desviación estándar por canal (RGB, escala [0,1]) usando SOLO train.
    Se estiman con una muestra aleatoria de `n_sample` imágenes de train: con
    ~25 millones de píxeles el error de estimación es despreciable y es mucho más rápido.
    """
    s = np.zeros(3); s2 = np.zeros(3); n = 0
    train = df.loc[df.split == "train", "processed_path"]
    for p in train.sample(min(n_sample, len(train)), random_state=SEED):
        img = cv2.imread(p)
        if img is None:
            continue
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0
        s += x.sum(0); s2 += (x ** 2).sum(0); n += len(x)
    mean = s / n
    std = np.sqrt(s2 / n - mean ** 2)          # Var(x) = E[x²] − E[x]²
    stats = {"mean_rgb": mean.round(5).tolist(), "std_rgb": std.round(5).tolist(),
             "img_size": IMG_SIZE, "n_train": int((df.split == "train").sum())}
    (PROC_DIR / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def main(max_seconds: float | None = None) -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df = split(balance(build_index()))
    df["processed_path"] = df.apply(processed_path, axis=1)
    df.to_csv(SPLITS_CSV, index=False)
    print("\nImágenes por split y calidad:")
    print(pd.crosstab(df["split"], df["quality"], margins=True))

    if not resize_all(df, max_seconds):
        done = len(_existing_files() & set(df["processed_path"]))
        print(f"\nPausa por tiempo: {done}/{len(df)} imágenes redimensionadas. Vuelve a correr el script.")
        return
    stats = compute_stats(df)
    print(f"\nListo. Normalización (train): media={stats['mean_rgb']} std={stats['std_rgb']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seconds", type=float, default=None)
    main(ap.parse_args().max_seconds)
