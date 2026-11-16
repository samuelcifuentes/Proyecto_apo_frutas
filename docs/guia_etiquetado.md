# Guía de etiquetado — Calidad y tamaño de frutas

Todos los integrantes deben etiquetar con **estos mismos criterios**, para que las etiquetas sean coherentes.

## Frutas incluidas
Manzana, banano, guayaba, limón/lima, naranja, granada (las 6 del dataset de Kaggle).

## 1. Clases de calidad (3)

| Clase | Código | Criterio | Ejemplos |
|-------|--------|----------|----------|
| **Buena** | `good` | Sin defectos visibles. Color uniforme y propio de la fruta madura. Piel firme. | Manzana roja brillante sin manchas; banano amarillo sin pecas oscuras. |
| **Regular** | `regular` | Defectos **leves** que cubren **menos del ~20 %** de la superficie visible: manchas pequeñas, rayones, golpe leve, poco madura o algo pasada de madurez. Todavía es comestible/vendible con descuento. | Manzana con 1–2 puntos oscuros; banano con pecas marrones; naranja con una zona verde. |
| **Mala** | `bad` | Defectos **graves** o que cubren **más del ~20 %**: podredumbre, moho, piel arrugada/hundida, golpes profundos, zonas negras extensas. No apta para la venta. | Banano casi negro; guayaba con manchas marrones grandes; granada con zonas podridas. |

**Regla de duda:** si dudas entre dos clases, escoge la **peor**. Si no se ve bien la fruta (borrosa, cortada, tapada por la mano) → **descartar**.

### ¿De dónde sale cada clase?
- `good` → carpeta `Good Quality_Fruits` (se revisa una muestra).
- `bad` → carpeta `Bad Quality_Fruits`, **excepto** las que tengan defectos leves, que se re-etiquetan como `regular`.
- `regular` → re-etiquetado manual de una muestra de `Bad Quality_Fruits` + recortes de `Mixed Qualit_Fruits` + fotos propias. Meta: **300–500 imágenes**.

## 2. Tamaño (3)

El dataset no tiene referencia de escala, así que el tamaño se calcula **automáticamente** a partir de la segmentación:

1. Se segmenta la fruta y se calcula su **diámetro equivalente** en píxeles: `d = sqrt(4·A/π)`, donde `A` es el área del contorno.
2. Se normaliza por el lado menor de la imagen: `d_norm = d / min(ancho, alto)`.
3. **Por cada tipo de fruta**, se divide en terciles (percentiles 33 y 66):
   - `small` (pequeño): `d_norm` < P33
   - `medium` (mediano): P33 ≤ `d_norm` < P66
   - `large` (grande): `d_norm` ≥ P66

Se hace por tipo de fruta porque un limón grande sigue siendo más pequeño que una manzana pequeña.

## 3. Fotos propias (≈15 por integrante)

- **Una sola fruta por foto**, centrada.
- **Fondo liso y uniforme** (hoja blanca, mesa de un solo color).
- Cámara a **~30 cm** de la fruta, siempre a la misma distancia (para que el tamaño sea comparable).
- Buena luz, sin sombras fuertes ni flash directo.
- Variar: distintas frutas, madurez, tamaños y defectos. Tratar de tener las 3 clases.
- Nombre del archivo: `<fruta>_<integrante>_<número>.jpg` → ej. `manzana_samuel_01.jpg`.

## 4. Archivo de etiquetas

Las fotos propias y los recortes van en `data/own_images/` y sus etiquetas en `data/own_images/labels.csv`:

```csv
filename,fruit,quality,size,source,annotator
manzana_samuel_01.jpg,apple,good,medium,own,samuel
banano_juancamilo_03.jpg,banana,regular,large,own,juancamilo
apple_mixed_012_crop2.jpg,apple,bad,small,mixed,andres
```

- `fruit`: `apple`, `banana`, `guava`, `lime`, `orange`, `pomegranate`
- `quality`: `good`, `regular`, `bad`
- `size`: `small`, `medium`, `large` (para fotos propias se puede dejar vacío y se calcula automáticamente)
- `source`: `own` (foto propia), `mixed` (recorte de Mixed), `relabel` (re-etiquetada de Bad)
