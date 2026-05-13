"""
Visualizador de preprocesado.

  - Si `SINGLE_LINE_MODE = True`: cada imagen de entrada se trata como una
    linea ya recortada y se procesa con `preprocessing.line_preprocess`. La
    salida conserva el ancho original (sin recortes laterales).

  - Si `SINGLE_LINE_MODE = False`: se usa el pipeline multi-linea original
    (`preprocessing.pipeline.run`): deteccion de bloques, columnas y lineas.

El pipeline a usar se elige solo a partir del modo; ambas rutas siguen
disponibles y aisladas.
"""

import sys
import time
import traceback
import contextlib
import io
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from preprocessing.pipeline        import run, analyze, auto_config
from preprocessing.line_preprocess import preprocess_line, LineConfig


INPUT_PATH       = Path("images")
OUTPUT_DIR       = Path("results")
SINGLE_LINE_MODE = True

PALETTE = [
    (46,  204, 113), (52,  152, 219), (231,  76,  60),
    (241, 196,  15), (155,  89, 182), ( 26, 188, 156),
    (230, 126,  34),
]
BLOCK_COLORS = [(255, 144, 30), (0, 200, 180), (200, 80, 200)]


def _save(path: str, img: np.ndarray) -> None:
    """Guarda con imencode para soportar rutas no-ASCII en Windows."""
    ext = Path(path).suffix.lower() or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"cv2.imencode fallo para '{path}'")
    Path(path).write_bytes(buf.tobytes())
    print(f"  [IMG]  {path}")


def _sep(char: str = "-", w: int = 60) -> str:
    # Construye una linea separadora para logs por consola.
    return f"  {char * w}"


# ----------------------------------------------------------------------------
# RUTA SINGLE-LINE
# ----------------------------------------------------------------------------

def run_single_line(
    image_path: str,
    debug_dir:  str = str(OUTPUT_DIR),
) -> str:
    """
    Procesa una imagen como UNA SOLA LINEA manuscrita.

    No detecta bloques ni segmenta lineas: la imagen completa es la linea.
    Conserva el ancho original; solo recorta arriba/abajo.

    Salida:
        {debug_dir}/{stem}.jpg
    """
    stem = Path(image_path).stem
    print()
    print(_sep("="))
    print(f"  Procesando: {image_path}")
    print(_sep("="))

    out_dir = Path(debug_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"  ERROR: no se pudo leer '{image_path}'")
        return "error"
    h, w = img_bgr.shape[:2]
    print(f"  Imagen: {w}x{h} px")

    cfg    = LineConfig(debug=True)
    result = preprocess_line(img_bgr, cfg=cfg)

    if result.warnings:
        print(_sep())
        for wm in result.warnings:
            print(f"  [!]  {wm}")

    print(_sep())
    print("  RESULTADO")
    print(_sep())
    print(f"  Altura texto est.    {result.text_height_est} px")
    print(f"  Angulo deskew        {result.angle_deg:+.2f} deg")
    print(f"  Binaria full         {result.binary_full.shape[1]}x{result.binary_full.shape[0]} px")
    print(f"  Salida (OCR-ready)   {result.image.shape[1]}x{result.image.shape[0]} px")
    print(f"  Tinta en salida      {100.0 * (result.image < 128).mean():.1f}%")

    out_path = out_dir / f"{stem}.jpg"
    ok, buf  = cv2.imencode(".jpg", result.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        print(f"  ERROR codificando {stem}.jpg")
        return "error"
    out_path.write_bytes(buf.tobytes())
    print(_sep())
    print(f"  [OK]   -> {out_path}")
    print(_sep("="))
    print()
    return "ok"


# ----------------------------------------------------------------------------
# RUTA MULTI-LINEA
# ----------------------------------------------------------------------------

def vis_lines_detected(
    img:         np.ndarray,
    line_boxes:  list,
    block_boxes: list,
    out:         Path,
    prefix:      str = "",
) -> None:
    # Dibuja sobre la imagen los bounding boxes de bloques y lineas detectados
    # y guarda el resultado como JPG.
    vis  = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    H, W = vis.shape[:2]

    for bi, blk in enumerate(block_boxes):
        by_top, by_bot, bx_l, bx_r = blk if len(blk) == 4 else (0, H - 1, blk[0], blk[1])
        bc = BLOCK_COLORS[bi % len(BLOCK_COLORS)]
        cv2.rectangle(vis, (bx_l, by_top), (bx_r, by_bot), bc, 2)
        cv2.putText(vis, f"B{bi+1}", (bx_l + 4, by_top + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, bc, 2)

    for i, lb in enumerate(line_boxes):
        color = PALETTE[i % len(PALETTE)]
        if isinstance(lb, (list, tuple)) and len(lb) == 4:
            y_top, y_bot, x_left, x_right = lb
            cv2.rectangle(vis, (x_left, y_top), (x_right, y_bot), color, 2)
            cv2.putText(vis, f"L{i+1}", (x_left + 6, max(20, y_top + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    summ = f"{len(line_boxes)} lineas  |  {len(block_boxes)} bloques"
    cv2.putText(vis, summ, (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0),       3)
    cv2.putText(vis, summ, (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    _save(str(out / f"{prefix}lines_detected.jpg"), vis)


def save_line_crops(
    line_crops: list,
    out:        Path,
    fixed_name: str = "",
) -> int:
    # Persiste cada line crop como JPG numerado, devolviendo cuantas se guardaron.
    out.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, crop in enumerate(line_crops, 1):
        if crop is None or crop.size == 0:
            continue
        ok, buf = cv2.imencode(".jpg", crop)
        if not ok:
            print(f"  [WARN] no se pudo codificar linea {i}")
            continue
        filename = fixed_name if fixed_name else f"line_{i:03d}.jpg"
        (out / filename).write_bytes(buf.tobytes())
        saved += 1
        if fixed_name:
            break
    return saved


def run_multiline(
    image_path: str,
    debug_dir:  str = str(OUTPUT_DIR),
) -> str:
    """Pipeline multi-linea."""
    stem = Path(image_path).stem
    print()
    print(_sep("="))
    print(f"  Procesando: {image_path}")
    print(_sep("="))

    out_dir = Path(debug_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"  ERROR: no se pudo leer '{image_path}'")
        return "error"
    h, w = img_bgr.shape[:2]
    print(f"  Imagen: {w}x{h} px")

    m   = analyze(img_bgr)
    cfg = auto_config(img_bgr)

    print(_sep())
    print("  METRICAS")
    print(_sep())
    print(f"  Contraste p95-p5  {m.contrast:>8.1f}  {'<- CLAHE' if m.needs_clahe else ''}")
    print(f"  Luminancia media  {m.mean_luminance:>8.1f}")
    print(f"  Altura texto est. {m.estimated_text_h:>8.1f} px")

    result = run(img_bgr, cfg=cfg)

    if result.warnings:
        print(_sep())
        for wm in result.warnings:
            print(f"  [!]  {wm}")

    print(_sep())
    print(f"  LINEAS: {result.n_lines}  BLOQUES: {len(result.block_boxes)}")

    lines_dir = out_dir / stem
    vis_lines_detected(result.binary, result.line_boxes, result.block_boxes, out_dir, prefix=stem + "_")
    n_saved = save_line_crops(result.line_crops, lines_dir)
    print(f"  [OK]   {n_saved} lineas -> {stem}/line_NNN.jpg")
    print(_sep("="))
    return "ok"


# ----------------------------------------------------------------------------
# BATCH
# ----------------------------------------------------------------------------

def _process_one(img_path: Path, output_dir: Path, quiet: bool = False) -> tuple[str, str, str]:
    # Procesa una imagen individual capturando stdout si quiet=True.
    runner = run_single_line if SINGLE_LINE_MODE else run_multiline
    try:
        if quiet:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                status = runner(str(img_path), str(output_dir))
        else:
            status = runner(str(img_path), str(output_dir))
        return (img_path.name, status or "ok", "")
    except Exception:
        return (img_path.name, "error", traceback.format_exc())


def run_batch(input_dir: Path, output_dir: Path) -> None:
    # Procesa todas las imagenes .jpg/.jpeg/.png de una carpeta y resume resultados.
    if not input_dir.is_dir():
        print(f"ERROR: '{input_dir}' no es una carpeta valida.", file=sys.stderr)
        sys.exit(1)

    _IMG_EXTS = {".jpg", ".jpeg", ".png"}
    images = sorted(p for p in input_dir.iterdir()
                    if p.suffix.lower() in _IMG_EXTS and p.is_file())
    if not images:
        print(f"No se encontraron .jpg/.jpeg/.png en '{input_dir}'.", file=sys.stderr)
        sys.exit(0)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    mode = "SINGLE-LINE" if SINGLE_LINE_MODE else "MULTI-LINE"
    print(f"  BATCH  |  {len(images)} imagenes  |  modo: {mode}")
    print(f"  Entrada : {input_dir}")
    print(f"  Salida  : {output_dir}")
    print(f"{'='*60}\n")

    t0      = time.time()
    results = []
    for i, img in enumerate(images, 1):
        name, status, err = _process_one(img, output_dir)
        results.append((name, status, err))
        label = {"ok": "OK", "skipped": "SKIP", "error": "ERROR"}.get(status, status.upper())
        print(f"  [{i:>3}/{len(images)}] {label:<5}  {name}")
        if status == "error":
            print(f"\nERROR en {name}:\n{err}", file=sys.stderr)

    ok_count  = sum(1 for _, s, _ in results if s == "ok")
    err_count = sum(1 for _, s, _ in results if s == "error")

    print(f"\n{'='*60}")
    print(f"  Completado en {time.time()-t0:.1f}s")
    print(f"  OK: {ok_count}  ERROR: {err_count}")
    print(f"{'='*60}\n")


def main() -> None:
    # Punto de entrada: dispatcha entre modo batch (carpeta) y modo archivo unico.
    inp = INPUT_PATH.resolve()
    out = OUTPUT_DIR.resolve()
    if inp.is_dir():
        run_batch(inp, out)
    elif inp.is_file() and inp.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        out.mkdir(parents=True, exist_ok=True)
        runner = run_single_line if SINGLE_LINE_MODE else run_multiline
        runner(str(inp), str(out))
    elif inp.is_file():
        print(f"ERROR: INPUT_PATH='{inp}' no soportado (.jpg, .jpeg, .png).", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"ERROR: INPUT_PATH='{inp}' no existe.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
