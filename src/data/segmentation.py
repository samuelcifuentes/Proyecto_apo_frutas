"""
Segmentación de la fruta (separar la fruta del fondo).

Responsable: Samuel Cifuentes — Fase 3 CRISP-DM (preparación de datos).

Idea general
------------
1. Reducimos la imagen a un tamaño de trabajo (máx. 256 px) para que sea rápido.
2. Creamos DOS máscaras candidatas con el umbral de Otsu (elige automáticamente el
   umbral que mejor separa dos grupos de píxeles, minimizando la varianza intra-grupo):
     a) sobre la saturación (la fruta suele ser más colorida que el fondo), y
     b) sobre la distancia de color al fondo (estimado con el borde de la foto, en Lab).
   Nos quedamos con la candidata cuya región principal tenga forma más "de fruta"
   (compacta/convexa, centrada y que no toque los bordes).
3. Refinamos esa máscara con GrabCut, un algoritmo que modela el color del fondo y
   de la fruta con mezclas de gaussianas (GMM) y corta la imagen como un grafo
   (min-cut) para separar ambos.
4. Limpiamos con operaciones morfológicas (apertura/cierre) y nos quedamos con el
   contorno más grande y más cercano al centro (la fruta principal).

Uso
---
    from src.data.segmentation import segment_fruit
    res = segment_fruit(imagen_bgr)
    res["mask"]      # máscara binaria (255 = fruta) del tamaño de la imagen original
    res["d_norm"]    # diámetro equivalente normalizado (para estimar el tamaño)
"""
from __future__ import annotations

import cv2
import numpy as np

WORK_SIZE = 256  # lado máximo de la imagen de trabajo


def _resize_max(img: np.ndarray, max_side: int = WORK_SIZE) -> tuple[np.ndarray, float]:
    """Reduce la imagen para que su lado mayor sea `max_side`. Devuelve (imagen, escala)."""
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img, scale


_REMBG_SESSION = None


def _mask_rembg(img: np.ndarray) -> np.ndarray:
    """
    Máscara con rembg (modelo U²-Net preentrenado para quitar fondos, variante ligera
    'u2netp'). Es una red de segmentación de "objetos salientes": predice para cada píxel
    la probabilidad de pertenecer al objeto principal. Umbral 0.5 → máscara binaria.
    Se usa en las fotos de Mixed, donde los fondos (pasto, tela de color) hacen fallar
    los métodos clásicos. Requiere `pip install rembg onnxruntime`.
    """
    global _REMBG_SESSION
    from rembg import new_session, remove  # import perezoso: solo si se usa
    if _REMBG_SESSION is None:
        _REMBG_SESSION = new_session("u2netp")
    m = remove(img, session=_REMBG_SESSION, only_mask=True)
    return (np.asarray(m) > 127).astype(np.uint8) * 255


def _mask_saturation(img: np.ndarray) -> np.ndarray:
    """Candidata 1: Otsu sobre la saturación (S de HSV). La fruta suele ser más colorida que el fondo."""
    s = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1], (5, 5), 0)
    _, mask = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def _mask_background_distance(img: np.ndarray) -> np.ndarray:
    """
    Candidata 2: qué tan DISTINTO es cada píxel del color del fondo.
    El fondo se estima con la mediana del borde de la foto en el espacio Lab (donde la
    distancia euclidiana se parece a la diferencia de color que percibe el ojo):
        dist(p) = || Lab(p) - Lab_fondo ||_2
    Sirve para fondos de color (p. ej. tela azul) donde la saturación falla.
    """
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    b = max(2, int(0.05 * min(h, w)))
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    dist = np.linalg.norm(lab - np.median(border, axis=0), axis=2)
    dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def _shape_score(mask: np.ndarray) -> float:
    """
    Qué tan "de fruta" es la región principal de una máscara (más alto = mejor):
      - solidez = área / área de su envolvente convexa  (las frutas son casi convexas)
      - penaliza tocar el borde de la imagen (la fruta suele estar centrada)
      - penaliza áreas muy pequeñas (<3 %) o casi toda la imagen (>85 %)
    """
    c = _main_contour(mask)
    if c is None:
        return -1.0
    h, w = mask.shape
    area = cv2.contourArea(c)
    hull = cv2.contourArea(cv2.convexHull(c))
    solidity = area / hull if hull > 0 else 0
    pts = c.reshape(-1, 2)
    touch = np.mean((pts[:, 0] <= 1) | (pts[:, 0] >= w - 2) | (pts[:, 1] <= 1) | (pts[:, 1] >= h - 2))
    frac = area / (h * w)
    penalty = 1.0 if 0.03 <= frac <= 0.85 else 0.3
    return solidity * (1 - touch) * penalty


def _clean(mask: np.ndarray, k: int = 5) -> np.ndarray:
    """Apertura (quita ruido pequeño) + cierre (tapa huecos) + relleno de huecos."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # rellenar huecos internos (manchas oscuras de la fruta que quedaron fuera)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _main_contour(mask: np.ndarray, min_area_frac: float = 0.01):
    """Elige el contorno principal: grande y cerca del centro de la imagen."""
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area_frac * h * w]
    if not contours:
        return None
    cx, cy = w / 2, h / 2
    diag = np.hypot(w, h)

    def score(c):
        m = cv2.moments(c)
        x, y = m["m10"] / m["m00"], m["m01"] / m["m00"]
        dist = np.hypot(x - cx, y - cy) / diag          # 0 = centro, ~0.5 = esquina
        return cv2.contourArea(c) / (h * w) - 0.5 * dist  # área grande y centrada

    return max(contours, key=score)


def segment_fruit(img_bgr: np.ndarray, use_grabcut: bool = True) -> dict:
    """
    Segmenta la fruta principal de una imagen BGR (formato de OpenCV).

    Devuelve un diccionario con:
      - mask:     máscara binaria uint8 (255 = fruta), mismo tamaño que la imagen original
      - contour:  contorno de la fruta (coordenadas de la imagen original) o None
      - bbox:     (x, y, ancho, alto) del rectángulo que encierra la fruta
      - area:     área de la fruta en píxeles (imagen original)
      - d_eq:     diámetro equivalente en píxeles  d = sqrt(4·A/π)
      - d_norm:   d_eq / min(alto, ancho)  → valor entre 0 y ~1, comparable entre imágenes
      - ok:       False si no se encontró ninguna fruta
    """
    H, W = img_bgr.shape[:2]
    small, scale = _resize_max(img_bgr)
    h, w = small.shape[:2]

    # Probamos las dos máscaras iniciales y nos quedamos con la de forma más "de fruta"
    candidates = [_clean(_mask_saturation(small)), _clean(_mask_background_distance(small))]
    mask = max(candidates, key=_shape_score)

    if use_grabcut and mask.any():
        # Inicializamos GrabCut con la máscara de Otsu:
        #   fruta segura (erosionada), fruta probable, fondo probable y fondo seguro (borde).
        gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
        gc[mask > 0] = cv2.GC_PR_FGD
        sure_fg = cv2.erode(mask, np.ones((9, 9), np.uint8), iterations=2)
        gc[sure_fg > 0] = cv2.GC_FGD
        border = max(2, int(0.02 * min(h, w)))
        gc[:border, :] = gc[-border:, :] = cv2.GC_BGD
        gc[:, :border] = gc[:, -border:] = cv2.GC_BGD
        try:
            bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            cv2.grabCut(small, gc, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
            refined = _clean(np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8))
            # Solo aceptamos el refinamiento si no empeora la forma
            if _shape_score(refined) >= _shape_score(mask) - 0.05:
                mask = refined
        except cv2.error:
            pass  # si GrabCut falla, nos quedamos con la máscara de Otsu

    contour = _main_contour(mask)
    if contour is None:
        return {"mask": np.zeros((H, W), np.uint8), "contour": None, "bbox": None,
                "area": 0.0, "d_eq": 0.0, "d_norm": 0.0, "ok": False}

    # Llevamos el contorno al tamaño original
    contour_full = (contour.astype(np.float32) / scale).astype(np.int32)
    mask_full = np.zeros((H, W), np.uint8)
    cv2.drawContours(mask_full, [contour_full], -1, 255, thickness=cv2.FILLED)

    area = float(cv2.contourArea(contour_full))
    d_eq = float(np.sqrt(4.0 * area / np.pi))
    return {
        "mask": mask_full,
        "contour": contour_full,
        "bbox": cv2.boundingRect(contour_full),
        "area": area,
        "d_eq": d_eq,
        "d_norm": d_eq / min(H, W),
        "ok": True,
    }


def crop_fruit(img_bgr: np.ndarray, seg: dict, pad: float = 0.08, black_bg: bool = False) -> np.ndarray:
    """Recorta la fruta usando el bbox (con un margen `pad`). Si black_bg=True, pone el fondo en negro."""
    if not seg["ok"]:
        return img_bgr
    x, y, w, h = seg["bbox"]
    H, W = img_bgr.shape[:2]
    px, py = int(w * pad), int(h * pad)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(W, x + w + px), min(H, y + h + py)
    img = img_bgr.copy()
    if black_bg:
        img[seg["mask"] == 0] = 0
    return img[y0:y1, x0:x1]


# ---------------------------------------------------------------------------
# Varias frutas por imagen (carpeta Mixed Qualit_Fruits)
# ---------------------------------------------------------------------------
def _split_touching(mask: np.ndarray, peak_frac: float = 0.55) -> np.ndarray:
    """
    Separa frutas que se tocan con la transformada de distancia + watershed.

    La transformada de distancia asigna a cada píxel de la fruta su distancia al
    fondo más cercano: el centro de cada fruta queda como un "pico" (≈ su radio) y
    el punto de contacto entre dos frutas queda como un "valle" estrecho.
    Para cada región conectada tomamos como semillas las zonas con distancia mayor a
    `peak_frac` × (máximo de esa región); así dos círculos que se tocan dan dos
    semillas. Luego watershed "inunda" desde cada semilla y donde dos inundaciones se
    encuentran queda la frontera entre frutas.
    Devuelve una imagen de etiquetas (0 = fondo, 1..k = frutas).
    """
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    n_cc, cc = cv2.connectedComponents(mask)
    peaks = np.zeros(mask.shape, np.uint8)
    for k in range(1, n_cc):
        region = cc == k
        dmax = dist[region].max()
        peaks[region & (dist > peak_frac * dmax)] = 255
    if not peaks.any():
        return np.zeros(mask.shape, np.int32)
    _, markers = cv2.connectedComponents(peaks)
    markers = markers + 1                      # fondo seguro = 1
    markers[cv2.subtract(mask, peaks) > 0] = 0  # zona a decidir por watershed
    markers = cv2.watershed(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), markers)
    markers[markers <= 1] = 0                  # quita fondo y fronteras (-1)
    return np.where(markers > 0, markers - 1, 0).astype(np.int32)


def find_fruits(img_bgr: np.ndarray, method: str = "rembg", min_area_frac: float = 0.003,
                max_area_frac: float = 0.7) -> list[dict]:
    """
    Encuentra TODAS las frutas de una imagen (p. ej. fotos de Mixed con varias frutas).

    method = "rembg"   → máscara con U²-Net (recomendado, robusto a fondos difíciles)
    method = "classic" → máscaras de Otsu (saturación / distancia al fondo)

    Devuelve una lista de diccionarios {contour, bbox, area, d_eq, d_norm} en
    coordenadas de la imagen original, ordenada de izquierda a derecha.
    """
    H, W = img_bgr.shape[:2]
    small, scale = _resize_max(img_bgr, 512)   # más resolución: las frutas son pequeñas
    h, w = small.shape[:2]

    def components(mask):
        labels = _split_touching(mask)
        out = []
        for k in range(1, labels.max() + 1):
            comp = (labels == k).astype(np.uint8) * 255
            cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cs:
                continue
            c = max(cs, key=cv2.contourArea)
            a = cv2.contourArea(c)
            if not (min_area_frac * h * w <= a <= max_area_frac * h * w):
                continue
            out.append(c)
        return out

    if method == "rembg":
        best = components(_clean(_mask_rembg(small), k=7))
    else:
        best = []
        for m in (_mask_saturation(small), _mask_background_distance(small)):
            cs = components(_clean(m, k=7))
            if len(cs) > len(best):
                best = cs

    fruits = []
    for c in best:
        cf = (c.astype(np.float32) / scale).astype(np.int32)
        area = float(cv2.contourArea(cf))
        d_eq = float(np.sqrt(4.0 * area / np.pi))
        fruits.append({"contour": cf, "bbox": cv2.boundingRect(cf), "area": area,
                       "d_eq": d_eq, "d_norm": d_eq / min(H, W)})
    return sorted(fruits, key=lambda f: f["bbox"][0])
