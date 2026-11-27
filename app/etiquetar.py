"""
Herramienta de etiquetado rápido de calidad (buena / regular / mala / descartar).

Ejecutar desde la raíz del repo:
    streamlit run app/etiquetar.py

Muestra una imagen a la vez y guarda cada decisión en data/own_images/labels.csv con el
formato de la guía (docs/guia_etiquetado.md). Las imágenes que ya están en labels.csv no
se vuelven a mostrar, así que se puede cerrar y continuar después.

Fuentes:
  * Recortes de Mixed       → data/interim/mixed_crops/  (generar con python -m src.data.crop_mixed)
  * Mala de Kaggle          → data/raw/Bad Quality_Fruits/ (re-etiquetar las que tengan defectos leves)
  * Fotos propias sin etiqueta → data/own_images/

Criterio (resumen de la guía):
  Buena   = sin defectos visibles, color uniforme.
  Regular = defectos LEVES en menos del ~20 % de la superficie visible.
  Mala    = podredumbre, moho, zonas negras grandes o defectos en más del ~20 %.
  Descartar = varias frutas, borrosa, mal recortada o no es una de las 6 frutas.
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LABELS = ROOT / "data/own_images/labels.csv"
HEADER = ["filename", "fruit", "quality", "size", "source", "annotator"]
FRUITS = ["apple", "banana", "guava", "lime", "orange", "pomegranate"]
FRUIT_ES = {"apple": "manzana", "banana": "banano", "guava": "guayaba",
            "lime": "limón", "orange": "naranja", "pomegranate": "granada"}
KAGGLE_FRUIT = {"Apple": "apple", "Banana": "banana", "Guava": "guava",
                "Lime": "lime", "Orange": "orange", "Pomegranate": "pomegranate"}
EXTS = {".jpg", ".jpeg", ".png"}


def rel(p: Path) -> str:
    """Ruta relativa a la raíz del repo, con '/' (formato de labels.csv)."""
    return p.resolve().relative_to(ROOT).as_posix()


def labeled() -> set[str]:
    if not LABELS.exists():
        return set()
    with open(LABELS, newline="", encoding="utf-8") as f:
        return {row["filename"] for row in csv.DictReader(f)}


def guess_fruit(name: str) -> str | None:
    n = name.lower()
    for en, es in FRUIT_ES.items():
        if en in n or es.replace("ó", "o") in n or es in n:
            return en
    return None


@st.cache_data
def candidates(source: str) -> list[tuple[str, str | None]]:
    """Lista (ruta_relativa, fruta) de la fuente elegida, en orden reproducible."""
    items: list[tuple[str, str | None]] = []
    if source == "mixed":
        for d in sorted((ROOT / "data/interim/mixed_crops").glob("*")):
            if d.is_dir():
                items += [(rel(p), d.name) for p in sorted(d.iterdir()) if p.suffix.lower() in EXTS]
    elif source == "relabel":
        for d in sorted((ROOT / "data/raw/Bad Quality_Fruits").glob("*_Bad")):
            fruit = KAGGLE_FRUIT[d.name.split("_")[0]]
            items += [(rel(p), fruit) for p in sorted(d.iterdir()) if p.suffix.lower() in EXTS]
        preds = ROOT / "experiments/results/random_forest_preds.csv"
        if preds.exists():
            # Primero las que el modelo cree más probable que sean "regular": así aparecen
            # antes las frutas con defectos leves y no cientos de frutas podridas.
            import pandas as pd
            p = pd.read_csv(preds)
            prob = dict(zip(p["image_path"].astype(str).str.replace("\\", "/", regex=False),
                            p.get("prob_regular", pd.Series(0, index=p.index))))
            items.sort(key=lambda it: -prob.get(it[0], 0.0))
        else:
            random.Random(42).shuffle(items)  # mezclar para no revisar una sola ráfaga seguida
    else:  # own
        items = [(rel(p), guess_fruit(p.name)) for p in sorted((ROOT / "data/own_images").iterdir())
                 if p.suffix.lower() in EXTS]
    return items


def save(filename: str, fruit: str, quality: str, source: str, annotator: str) -> None:
    new = not LABELS.exists()
    with open(LABELS, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HEADER)
        w.writerow([filename, fruit, quality, "", source, annotator])


def undo_last() -> str | None:
    """Borra la última fila de labels.csv (solo si la agregó esta sesión)."""
    hist = st.session_state.get("history", [])
    if not hist:
        return None
    last = hist.pop()
    with open(LABELS, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if rows and rows[-1] and rows[-1][0] == last:
        with open(LABELS, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows[:-1])
        return last
    return None


def main() -> None:
    st.set_page_config(page_title="Etiquetado de calidad", page_icon="🏷️", layout="wide")
    st.title("Etiquetado de calidad de frutas")

    with st.sidebar:
        annotator = st.text_input("Tu nombre (anotador)", value=st.session_state.get("annotator", ""))
        st.session_state["annotator"] = annotator
        source = st.radio("Fuente", ["mixed", "relabel", "own"],
                          format_func={"mixed": "Recortes de Mixed",
                                       "relabel": "Mala de Kaggle (buscar defectos leves)",
                                       "own": "Fotos propias sin etiqueta"}.get)
        fruit_filter = st.selectbox("Fruta", ["todas", *FRUITS],
                                    format_func=lambda f: f if f == "todas" else FRUIT_ES[f])
        st.markdown(
            "**Criterio**\n\n"
            "- **Buena**: sin defectos visibles.\n"
            "- **Regular**: defectos leves, < ~20 % de la superficie.\n"
            "- **Mala**: podredumbre, moho o > ~20 % dañado.\n"
            "- **Descartar**: varias frutas, borrosa o mal recortada.\n\n"
            "Si dudas entre dos clases, escoge la peor."
        )
        session = st.session_state.setdefault("count", {"good": 0, "regular": 0, "bad": 0, "discard": 0})
        st.metric("Regular etiquetadas en esta sesión", session["regular"])
        st.caption(f"Buena {session['good']} · Mala {session['bad']} · Descartadas {session['discard']}")

    if not annotator.strip():
        st.info("Escribe tu nombre en la barra lateral para empezar.")
        return

    done = labeled()
    pending = [(p, f) for p, f in candidates(source)
               if p not in done and (fruit_filter == "todas" or f == fruit_filter)]
    total_regular = 0
    if LABELS.exists():
        with open(LABELS, newline="", encoding="utf-8") as fh:
            total_regular = sum(1 for r in csv.DictReader(fh) if r["quality"] == "regular")
    st.caption(f"Pendientes en esta fuente: {len(pending)} · Regular en labels.csv: {total_regular}")
    if not pending:
        st.success("No quedan imágenes pendientes en esta fuente.")
        return

    path, fruit = pending[0]
    col_img, col_btn = st.columns([1.3, 1])
    with col_img:
        img = ImageOps.exif_transpose(Image.open(ROOT / path)).convert("RGB")
        if max(img.size) < 300:  # los recortes de Mixed son pequeños: se amplían para verlos
            s = 300 / max(img.size)
            img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
        img.thumbnail((700, 700))
        st.image(img, caption=path)
    with col_btn:
        if fruit is None or source == "own":
            fruit = st.selectbox("¿Qué fruta es?", FRUITS,
                                 index=FRUITS.index(fruit) if fruit in FRUITS else 0,
                                 format_func=FRUIT_ES.get)
        else:
            st.subheader(FRUIT_ES[fruit].capitalize())
        choice = None
        c1, c2 = st.columns(2)
        if c1.button("✅ Buena", use_container_width=True):
            choice = "good"
        if c2.button("🟡 Regular", use_container_width=True):
            choice = "regular"
        c3, c4 = st.columns(2)
        if c3.button("❌ Mala", use_container_width=True):
            choice = "bad"
        if c4.button("🗑️ Descartar", use_container_width=True):
            choice = "discard"
        if choice:
            save(path, fruit, choice, source, annotator.strip())
            session[choice] += 1
            st.session_state.setdefault("history", []).append(path)
            st.rerun()
        st.divider()
        if st.button("↩️ Deshacer la última", use_container_width=True):
            undone = undo_last()
            st.toast(f"Deshecho: {undone}" if undone else "No hay nada que deshacer en esta sesión")
            st.rerun()
        st.caption("Después de etiquetar, quien entrene debe correr de nuevo "
                   "`python -m src.data.preprocess`.")


if __name__ == "__main__":
    main()
