"""
Extrae las características de TODAS las imágenes de data/splits.csv (reanudable).

Uso (desde la raíz del repo):
    python -m src.features.build_features                    # todo (rembg; lento)
    python -m src.features.build_features --max-seconds 150  # por partes, se puede repetir
    python -m src.features.build_features --limit 600        # prueba rápida con una muestra

Entradas : data/splits.csv (+ imágenes 128x128 de data/processed, generadas por preprocess)
Salidas  : data/processed/features.csv            → una fila por imagen (características + split)
           experiments/results/size_thresholds.json → límites de tamaño por fruta (terciles de train)

Se usan las imágenes 128x128 del preprocesamiento (las mismas que ve la CNN), así todas
las fotos tienen la misma escala sin importar su resolución original.
La caché permite interrumpir y continuar: solo se procesan las imágenes que falten.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.data.segmentation import segment_fruit
from src.features.extraction import FEATURE_NAMES, extract_features
from src.features.size_estimation import fit_size_thresholds_by_fruit, save_size_thresholds

SPLITS = Path("data/splits.csv")
OUT = Path("data/processed/features.csv")
THRESHOLDS = Path("experiments/results/size_thresholds.json")


def _sample(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Muestra estratificada por (split, fruta, calidad) para pruebas rápidas."""
    frac = min(1.0, limit / len(df))
    return (df.groupby(["split", "fruit", "quality"], group_keys=False)
              .apply(lambda g: g.sample(max(1, round(len(g) * frac)), random_state=42)))


def main(max_seconds: float | None, limit: int | None, method: str) -> None:
    logging.basicConfig(level=logging.WARNING)
    df = pd.read_csv(SPLITS)
    if limit:
        df = _sample(df, limit)
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["path"])
    todo = df[~df["path"].isin(set(done["path"]))]
    print(f"{len(df)} imágenes en total, {len(todo)} pendientes (método: {method})")

    rows, t0, saved = [], time.time(), 0

    def flush():
        """Guarda en disco lo procesado hasta ahora (así no se pierde si se interrumpe)."""
        nonlocal rows, saved
        if rows:
            OUT.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
            saved += len(rows)
            rows = []

    for i, (_, r) in enumerate(todo.iterrows(), 1):
        if max_seconds and time.time() - t0 > max_seconds:
            break
        img = cv2.imread(r["processed_path"])
        row = {k: r[k] for k in ("path", "processed_path", "fruit", "quality", "source", "split", "group")}
        if img is None:
            seg_ok, feats, d_norm = False, np.full(len(FEATURE_NAMES), np.nan), np.nan
        else:
            seg = segment_fruit(img, method=method)
            seg_ok, d_norm = bool(seg["ok"]), float(seg["d_norm"])
            feats = extract_features(img, seg["mask"])
        row.update({"image_path": r["path"], "segmentation_ok": seg_ok, "d_norm": d_norm})
        row.update(dict(zip(FEATURE_NAMES, feats)))
        rows.append(row)
        if i % 100 == 0:
            flush()
            rate = i / (time.time() - t0)
            left = (len(todo) - i) / rate / 60
            print(f"  {len(done) + i}/{len(df)} imágenes ({rate:.2f} img/s, faltan ≈ {left:.0f} min)", flush=True)

    flush()
    total = len(done) + saved
    rate = saved / max(1e-9, time.time() - t0)
    print(f"Procesadas en esta corrida: {saved} ({rate:.2f} img/s). Total: {total}/{len(df)}")
    if total < len(df):
        print(f"Faltan {len(df) - total}. Vuelve a correr el comando "
              f"(≈ {(len(df) - total) / max(rate, 1e-9) / 60:.0f} min a este ritmo).")
        return

    feats = pd.read_csv(OUT).drop_duplicates("path", keep="last")
    feats.to_csv(OUT, index=False)  # por si una corrida interrumpida dejó filas repetidas
    train = feats[(feats["split"] == "train") & feats["segmentation_ok"].astype(bool)]
    save_size_thresholds(fit_size_thresholds_by_fruit(train["fruit"], train["d_norm"].clip(0, 1)), THRESHOLDS)
    print(f"Listo: {OUT} y límites de tamaño por fruta en {THRESHOLDS}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None, help="usar solo una muestra estratificada")
    ap.add_argument("--method", choices=["rembg", "classic"], default="rembg")
    a = ap.parse_args()
    main(a.max_seconds, a.limit, a.method)
