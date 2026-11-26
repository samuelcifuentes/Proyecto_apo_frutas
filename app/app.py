"""Interfaz del proyecto. Ejecutar desde la raíz con streamlit run app/app.py."""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
import streamlit as st

# Streamlit añade app/ a sys.path al ejecutar el archivo desde su lanzador.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.data.segmentation import segment_fruit
from src.features.size_estimation import estimate_size


def read_photo(upload) -> np.ndarray:
    """Lee la orientación de la cámara y conserva las proporciones para segmentar."""
    with Image.open(upload) as photo:
        if photo.format not in {"JPEG", "PNG"}:
            raise ValueError("Elige una imagen JPG o PNG.")
        if photo.width * photo.height > 50_000_000:
            raise ValueError("Reduce la resolución de la foto a 50 megapíxeles o menos.")
        photo.thumbnail((1200, 1200))
        rgb = np.asarray(ImageOps.exif_transpose(photo).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def demo_quality(image: np.ndarray, mask: np.ndarray) -> str:
    """Regla provisional de brillo sobre los píxeles de la fruta segmentada."""
    brightness = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2][mask > 0]
    dark_fraction = np.mean(brightness < 0.6 * np.median(brightness))
    if dark_fraction >= 0.20:
        return "Mala"
    if dark_fraction >= 0.02:
        return "Regular"
    return "Buena"


def segment_photo(image: np.ndarray) -> dict:
    try:
        return segment_fruit(image)
    except (OSError, RuntimeError):
        # Si los pesos de rembg no están disponibles, se puede seguir sin red.
        return segment_fruit(image, method="classic")


def main() -> None:
    st.set_page_config(page_title="Calidad y tamaño de frutas", page_icon="🍎")
    st.title("Calidad y tamaño de frutas")
    st.write("Centra una sola fruta sobre un fondo liso, con buena luz y a unos 30 cm de la cámara.")
    source = st.radio("Origen de la foto", ["Subir foto", "Usar cámara"], horizontal=True)
    if source == "Usar cámara":
        upload = st.camera_input("Toma una foto de la fruta")
    else:
        upload = st.file_uploader("Selecciona una foto JPG o PNG", type=["jpg", "jpeg", "png"])
    if upload is None:
        return

    try:
        image = read_photo(upload)
    except ValueError as error:
        st.error(str(error))
        return
    except (OSError, Image.DecompressionBombError):
        st.error("No se pudo leer la foto. Prueba con otro archivo JPG o PNG.")
        return

    with st.spinner("Separando la fruta del fondo…"):
        try:
            result = segment_photo(image)
        except cv2.error:
            st.error("No se pudo procesar la imagen. Prueba con otra foto de la fruta.")
            return
    if not result["ok"]:
        st.warning("No se pudo distinguir la fruta. Prueba con un fondo que contraste con la cáscara.")
        return

    mask = result["mask"]
    size = estimate_size(mask=mask)
    preview = image.copy()
    cv2.drawContours(preview, [result["contour"]], -1, (250, 225, 30), 3)
    photo_col, results_col = st.columns([1.2, 1])
    with photo_col:
        st.image(preview, channels="BGR", caption="Contorno usado para la estimación")
    with results_col:
        st.metric("Calidad · modelo de prueba", demo_quality(image, mask))
        st.caption("Una regla de brillo sin entrenar simula la clasificación mientras el equipo evalúa los modelos.")
        st.metric("Tamaño relativo · provisional", str(size["category"]).capitalize())
        st.caption(
            f"El diámetro equivalente ocupa el {size['diameter_normalized']:.0%} del lado menor "
            "de la foto y cambia con el encuadre. Se usan límites iniciales hasta calibrarlos por fruta."
        )


if __name__ == "__main__":
    main()
