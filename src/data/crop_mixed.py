"""
Recorta cada fruta de las fotos de `Mixed Qualit_Fruits` (varias frutas por foto)
y la guarda como imagen individual.

Responsable: Samuel Cifuentes — Fase 3 CRISP-DM.

Salida:
  data/interim/mixed_crops/<fruta>/<foto>_c<k>.jpg   → un recorte por fruta
  data/interim/mixed_crops/crops.csv                 → índice de recortes (sin calidad)

Los recortes NO traen etiqueta de calidad: Samuel Lasso los revisa y los etiqueta
(good / regular / bad) en data/own_images/labels.csv, descartando los que tengan
más de una fruta o estén mal recortados (ver docs/guia_etiquetado.md).

Uso (desde la raíz del repo):
    python -m src.data.crop_mixed               # procesa todo (≈12 min)
    python -m src.data.crop_mixed --max-seconds 150   # por partes; se puede repetir

El script es reanudable: si se interrumpe, al volver a correrlo continúa donde iba.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2

from src.data.segmentation import find_fruits

RAW_MIXED = Path("data/raw/Mixed Qualit_Fruits")
OUT_DIR = Path("data/interim/mixed_crops")
FRUIT_NAMES = {"Apple": "apple", "Banana": "banana", "Guava": "guava",
               "Lemon": "lime", "Lime": "lime", "Orange": "orange",
               "Pomegranate": "pomegranate"}
MIN_SIDE = 28     # descartamos recortes muy pequeños (px). Las fotos de Mixed son de 192x256,
                  # así que cada fruta mide ~30–60 px.
PAD = 0.08        # margen alrededor de la fruta


def main(max_seconds: float | None = None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = OUT_DIR / "crops.csv"
    done_sources = set()
    if index.exists():
        with open(index, newline="", encoding="utf-8") as f:
            done_sources = {row["source_image"] for row in csv.DictReader(f)}
    new_file = not index.exists()
    # también marcamos fotos procesadas que no dieron recortes
    log = OUT_DIR / "_processed.txt"
    if log.exists():
        done_sources |= set(log.read_text(encoding="utf-8").splitlines())

    t0 = time.time()
    n_img = n_crops = 0
    with open(index, "a", newline="", encoding="utf-8") as f, open(log, "a", encoding="utf-8") as lg:
        w = csv.writer(f)
        if new_file:
            w.writerow(["filename", "fruit", "source_image", "crop_id",
                        "x", "y", "w", "h", "d_norm"])
        for sub in sorted(p for p in RAW_MIXED.iterdir() if p.is_dir()):
            fruit = FRUIT_NAMES[sub.name]
            (OUT_DIR / fruit).mkdir(exist_ok=True)
            for img_path in sorted(sub.iterdir()):
                src = img_path.as_posix()
                if src in done_sources or img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                if max_seconds and time.time() - t0 > max_seconds:
                    print(f"Pausa por tiempo: {n_img} fotos, {n_crops} recortes en esta corrida.")
                    return
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                # las fotos de 8000x6000 se reducen al leer para ahorrar memoria
                if max(img.shape[:2]) > 2000:
                    s = 2000 / max(img.shape[:2])
                    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                H, W = img.shape[:2]
                for k, fr in enumerate(find_fruits(img, method="rembg")):
                    x, y, bw, bh = fr["bbox"]
                    if min(bw, bh) < MIN_SIDE:
                        continue
                    px, py = int(bw * PAD), int(bh * PAD)
                    x0, y0 = max(0, x - px), max(0, y - py)
                    x1, y1 = min(W, x + bw + px), min(H, y + bh + py)
                    out = OUT_DIR / fruit / f"{img_path.stem}_c{k}.jpg"
                    cv2.imwrite(str(out), img[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 92])
                    w.writerow([out.as_posix(), fruit, src, k, x, y, bw, bh, round(fr["d_norm"], 4)])
                    n_crops += 1
                lg.write(src + "\n")
                n_img += 1
    print(f"Listo: {n_img} fotos, {n_crops} recortes en esta corrida.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="detener después de N segundos (se puede reanudar)")
    main(ap.parse_args().max_seconds)
