"""
apps/ocr/segmentation.py — Visualización cacheada de la segmentación de líneas.

Genera, para cada página, una imagen JPG con cajas de colores sobre las
líneas y bloques detectados por el pipeline de preprocesado. Se usa en la
pantalla de edición para que el editor vea qué segmentó el sistema.

Salida cacheada en:
    {MEDIA_ROOT}/segmentation/{document_id}/page_{order:03d}_lines.jpg
    {MEDIA_ROOT}/segmentation/{document_id}/page_{order:03d}_lines.json

Reutiliza la misma lógica que `visualize.py` del paquete standalone:
ver `vis_lines_detected` para la convención de colores y formato.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

from django.conf import settings

logger = logging.getLogger(__name__)


# Paleta tomada de visualize.py para coherencia visual.
PALETTE = [
    (46,  204, 113), (52,  152, 219), (231,  76,  60),
    (241, 196,  15), (155,  89, 182), ( 26, 188, 156),
    (230, 126,  34),
]
BLOCK_COLORS = [(255, 144, 30), (0, 200, 180), (200, 80, 200)]


# ── Rutas ─────────────────────────────────────────────────────────────────

def _seg_dir(document_id: int) -> Path:
    return Path(settings.MEDIA_ROOT) / "segmentation" / str(document_id)


def viz_path(document_id: int, page_order: int) -> Path:
    return _seg_dir(document_id) / f"page_{page_order:03d}_lines.jpg"


def boxes_json_path(document_id: int, page_order: int) -> Path:
    return _seg_dir(document_id) / f"page_{page_order:03d}_lines.json"


# ── Generación ────────────────────────────────────────────────────────────

def _deskew_color(img_bgr, angle: float):
    """
    Aplica la misma rotación de deskew global a la imagen color original.
    Idéntico al helper homónimo en visualize.py — duplicamos en lugar
    de importar para no acoplar la app al script standalone.
    """
    import cv2
    import numpy as np

    if abs(angle) < 0.05:
        return img_bgr
    H, W = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle, 1.0)
    border_px = np.concatenate([
        img_bgr[0, :, :].reshape(-1, 3),
        img_bgr[-1, :, :].reshape(-1, 3),
        img_bgr[:, 0, :].reshape(-1, 3),
        img_bgr[:, -1, :].reshape(-1, 3),
    ])
    bg = tuple(int(v) for v in np.median(border_px, axis=0).tolist())
    return cv2.warpAffine(
        img_bgr, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=bg,
    )


def _draw_lines(img_bgr, line_boxes, block_boxes):
    """
    Dibuja cajas de colores sobre `img_bgr` (BGR uint8) y devuelve la
    imagen resultante. Mismo estilo que vis_lines_detected en visualize.py.
    """
    import cv2

    vis = img_bgr.copy()
    H, W = vis.shape[:2]

    # Bloques (más anchos)
    for bi, blk in enumerate(block_boxes):
        if len(blk) == 4:
            by_top, by_bot, bx_l, bx_r = blk
        else:
            by_top, by_bot, bx_l, bx_r = 0, H - 1, blk[0], blk[1]
        bc = BLOCK_COLORS[bi % len(BLOCK_COLORS)]
        cv2.rectangle(vis, (bx_l, by_top), (bx_r, by_bot), bc, 2)
        cv2.putText(vis, f"B{bi+1}", (bx_l + 4, by_top + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, bc, 2)

    # Líneas
    for i, lb in enumerate(line_boxes):
        if not (isinstance(lb, (list, tuple)) and len(lb) == 4):
            continue
        y_top, y_bot, x_left, x_right = lb
        color = PALETTE[i % len(PALETTE)]
        cv2.rectangle(vis, (x_left, y_top), (x_right, y_bot), color, 2)
        cv2.putText(vis, f"L{i+1}", (x_left + 6, max(20, y_top + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Resumen abajo a la izquierda
    summ = f"{len(line_boxes)} lineas  |  {len(block_boxes)} bloques"
    cv2.putText(vis, summ, (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, (0, 0, 0), 3)
    cv2.putText(vis, summ, (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, (255, 255, 255), 2)

    return vis


def generate_segmentation_image(
    image_path:   str,
    document_id:  int,
    page_order:   int,
) -> Optional[Path]:
    """
    Ejecuta el pipeline de preprocesado sobre `image_path`, dibuja las
    cajas detectadas sobre el facsimilar deskewed y guarda el resultado
    como JPG. También guarda un JSON con las cajas brutas para que el
    frontend pueda usarlas en herramientas interactivas.

    Devuelve la ruta del JPG generado, o None si falló.

    """
    import cv2
    import numpy as np
    from preprocessing.pipeline import auto_config, run

    # Cargar el facsimilar (Unicode-safe en Windows).
    buf = np.fromfile(image_path, dtype=np.uint8)
    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img_bgr is None:
        logger.error("No se pudo leer la imagen '%s'", image_path)
        return None

    cfg    = auto_config(img_bgr)
    result = run(img_bgr, cfg)

    return from_pipeline_result(img_bgr, result, document_id, page_order)


def from_pipeline_result(
    img_bgr,
    result,
    document_id: int,
    page_order:  int,
) -> Optional[Path]:
    """
    Genera y guarda la visualización a partir de un `PipelineResult`
    YA computado (evita re-correr pipeline.run). Es la vía que usa el
    thread del OCR para que el request handler no tenga que volver a
    procesar la imagen.
    """
    import cv2
    import numpy as np

    deskew_angle = float(getattr(result, "deskew_angle", 0.0))
    img_vis      = _deskew_color(img_bgr, deskew_angle)
    drawn        = _draw_lines(img_vis, result.line_boxes, result.block_boxes)

    return _write_outputs(
        drawn=drawn,
        img_bgr=img_bgr,
        img_vis=img_vis,
        line_boxes=result.line_boxes,
        block_boxes=result.block_boxes,
        deskew_angle=deskew_angle,
        n_lines=int(getattr(result, 'n_lines', len(result.line_boxes))),
        warnings=list(getattr(result, 'warnings', [])),
        document_id=document_id,
        page_order=page_order,
    )


def from_manuscript_detection(
    img_bgr,
    detection,
    document_id: int,
    page_order:  int,
) -> Optional[Path]:
    """
    Genera y guarda la visualización a partir de un `ManuscriptDetection`
    (el resultado del detector específico para manuscritos cursivos).
    Misma idea que `from_pipeline_result` pero adapta los campos.
    """
    return _write_outputs(
        drawn=_draw_lines(
            _deskew_color(img_bgr, detection.deskew_angle),
            detection.line_boxes, detection.block_boxes,
        ),
        img_bgr=img_bgr,
        img_vis=_deskew_color(img_bgr, detection.deskew_angle),
        line_boxes=detection.line_boxes,
        block_boxes=detection.block_boxes,
        deskew_angle=detection.deskew_angle,
        n_lines=detection.n_lines,
        warnings=detection.warnings,
        document_id=document_id,
        page_order=page_order,
    )


def _write_outputs(
    *,
    drawn,
    img_bgr,
    img_vis,
    line_boxes,
    block_boxes,
    deskew_angle: float,
    n_lines:      int,
    warnings:     list,
    document_id:  int,
    page_order:   int,
) -> Optional[Path]:
    """Codifica el JPG y el JSON sidecar. Devuelve la ruta del JPG."""
    import cv2

    out_jpg = viz_path(document_id, page_order)
    out_jpg.parent.mkdir(parents=True, exist_ok=True)

    ok, encoded = cv2.imencode(".jpg", drawn,
                               [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        logger.error("cv2.imencode falló para '%s'", out_jpg)
        return None
    out_jpg.write_bytes(encoded.tobytes())

    out_json = boxes_json_path(document_id, page_order)
    payload = {
        "deskew_angle": deskew_angle,
        "image_size":   {"width": int(img_bgr.shape[1]),
                         "height": int(img_bgr.shape[0])},
        "deskewed_size":{"width": int(img_vis.shape[1]),
                         "height": int(img_vis.shape[0])},
        "lines":  [
            {"order": i + 1,
             "y_top": int(yt), "y_bot": int(yb),
             "x_left": int(xl), "x_right": int(xr)}
            for i, (yt, yb, xl, xr) in enumerate(line_boxes)
        ],
        "blocks": [
            {"order": i + 1,
             "y_top": int(yt), "y_bot": int(yb),
             "x_left": int(xl), "x_right": int(xr)}
            for i, (yt, yb, xl, xr) in enumerate(block_boxes)
        ],
        "n_lines":  n_lines,
        "warnings": warnings,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    return out_jpg


def get_or_generate(
    image_path:  str,
    document_id: int,
    page_order:  int,
    *,
    force:        bool = False,
    doc_type:     str  = 'printed',
    allow_block:  bool = True,
) -> Optional[Path]:
    """
    Devuelve la ruta de la imagen de segmentación. Si no existe (o
    `force=True`), la genera. Si la generación falla, devuelve None.

    Args:
        image_path:   ruta al facsimilar original.
        document_id:  para construir el path de cache.
        page_order:   idem.
        force:        regenerar aunque haya cache válida.
        doc_type:     'printed' o 'manuscript'. Si manuscript, usa el
                      detector específico (`detect_manuscript_lines`)
                      con el mismo bbox que el OCR. Si printed, usa
                      `pipeline.run`.
        allow_block:  si False y la cache no existe, NO ejecuta el
                      pipeline pesado (devuelve None). Útil cuando el
                      caller no quiere esperar 5-30 s a que termine el
                      pipeline (e.g. handler HTTP en una página que
                      ya está siendo procesada por el thread OCR).

    Estrategia simple de invalidación: si el JPG es más viejo que el
    facsimilar, lo regeneramos (la imagen original podría haber cambiado).
    """
    out = viz_path(document_id, page_order)
    src = Path(image_path)

    if not force and out.is_file() and src.is_file():
        try:
            if out.stat().st_mtime >= src.stat().st_mtime:
                return out
        except OSError:
            pass

    if not allow_block:
        # Caller pidió NON-BLOCKING. La cache no es buena o no existe.
        return None

    # Para impreso reusamos la función vieja (que llama pipeline.run).
    # Para manuscrito, ejecutamos detect_manuscript_lines directamente.
    if doc_type == 'manuscript':
        return _generate_for_manuscript(image_path, document_id, page_order)
    return generate_segmentation_image(image_path, document_id, page_order)


def _generate_for_manuscript(
    image_path:  str,
    document_id: int,
    page_order:  int,
) -> Optional[Path]:
    """Genera la viz para manuscritos usando el detector específico."""
    import cv2
    import numpy as np

    buf = np.fromfile(image_path, dtype=np.uint8)
    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img_bgr is None:
        logger.error("No se pudo leer la imagen '%s'", image_path)
        return None

    try:
        from preprocessing.manuscript_lines import detect_manuscript_lines
        det = detect_manuscript_lines(img_bgr)
    except Exception as exc:
        logger.error("Error detectando líneas manuscritas en %s: %s",
                     image_path, exc)
        return None

    return from_manuscript_detection(img_bgr, det, document_id, page_order)


def media_url(document_id: int, page_order: int) -> str:
    """
    URL relativa al MEDIA_URL para servir el JPG cacheado.
    Útil para el frontend.
    """
    rel = f"segmentation/{document_id}/page_{page_order:03d}_lines.jpg"
    media_url = settings.MEDIA_URL.rstrip("/") + "/"
    return media_url + rel
