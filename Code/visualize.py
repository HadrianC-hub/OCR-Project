import sys
import time
import traceback
import contextlib
import io
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).parent))
from preprocessing.pipeline     import run, analyze, auto_config, to_gray, deskew_image
from preprocessing.binarization import binarize, clean_binary, filter_small_components


# CONFIGURACIÓN GLOBAL
INPUT_PATH       = Path("images")    # archivo .jpg/.png individual O carpeta con imágenes
OUTPUT_DIR       = Path("results")  # carpeta de salida para todas las imágenes
SINGLE_LINE_MODE = True               # si True, advierte cuando una imagen no tenga exactamente 1 línea

PALETTE = [
    (46,  204, 113),  (52,  152, 219),  (231,  76,  60),
    (241, 196,  15),  (155,  89, 182),  ( 26, 188, 156),
    (230, 126,  34),
]
BLOCK_COLORS = [(255, 144, 30), (0, 200, 180), (200, 80, 200)]


# Helpers

def _save(path: str, img: np.ndarray) -> None:
    """Guarda con imencode para soportar rutas no-ASCII en Windows."""
    ext = Path(path).suffix.lower() or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"cv2.imencode falló para '{path}'")
    Path(path).write_bytes(buf.tobytes())
    print(f"  [IMG]  {path}")

def _sep(char: str = "─", w: int = 60) -> str:
    return f"  {char * w}"


# Generación de imágenes

def vis_lines_detected(
    binary:      np.ndarray,
    line_boxes:  list,
    block_boxes: list,
    out:         Path,
    prefix:      str = "",
) -> None:
    """Dibuja bounding-boxes de bloques y líneas sobre la imagen binarizada."""
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    H, W = vis.shape[:2]

    for bi, blk in enumerate(block_boxes):
        by_top, by_bot, bx_l, bx_r = blk if len(blk) == 4 else (0, H - 1, blk[0], blk[1])
        bc = BLOCK_COLORS[bi % len(BLOCK_COLORS)]
        cv2.rectangle(vis, (bx_l, by_top), (bx_r, by_bot), bc, 2)
        cv2.putText(vis, f"B{bi+1}", (bx_l + 4, by_top + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, bc, 2)

    for i, (y_top, y_bot, x_left, x_right) in enumerate(line_boxes):
        color = PALETTE[i % len(PALETTE)]
        cv2.rectangle(vis, (x_left, y_top), (x_right, y_bot), color, 2)
        cv2.putText(vis, f"L{i+1}", (x_left + 6, max(20, y_top + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    summ = f"{len(line_boxes)} lineas  |  {len(block_boxes)} bloques"
    cv2.putText(vis, summ, (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0),       3)
    cv2.putText(vis, summ, (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    _save(str(out / f"{prefix}lines_detected.jpg"), vis)

def save_line_images(
    lines:       list,
    out:         Path,
    min_ink_pct: float = 2.0,
    prefix:      str   = "",
) -> int:
    """Guarda cada línea normalizada (float32 [0,1]) como JPEG en `out`. Omite líneas vacías."""
    out.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, line in enumerate(lines, 1):
        if 100.0 * float((line < 0.5).mean()) < min_ink_pct:
            continue
        img_u8 = (line * 255.0).clip(0, 255).astype("uint8")
        ok, buf = cv2.imencode(".jpg", img_u8)
        if not ok:
            print(f"  [WARN] no se pudo codificar línea {i}")
            continue
        filename = f"{prefix}line_{i:03d}.jpg" if prefix else f"line_{i:03d}.jpg"
        (out / filename).write_bytes(buf.tobytes())
        saved += 1
    return saved


# Diagnóstico por consola

def _print_stats(binary: np.ndarray, line_boxes: list, block_boxes: list) -> None:
    H, W   = binary.shape
    h_proj = (binary < 128).sum(axis=1).astype(float)
    h_sm   = uniform_filter(h_proj, size=5)
    h_max  = float(h_sm.max())

    print(_sep())
    print("  PROYECCIÓN HORIZONTAL")
    print(_sep())
    print(f"  pico={h_max:.0f}px  filas_activas={int((h_sm > h_max * 0.10).sum())}/{H}")
    print(f"  {'#':>3}  {'y_top':>6}  {'y_bot':>6}  {'alto':>5}  {'pico':>7}")
    print(f"  {'-'*3}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*7}")
    for i, (yt, yb, xl, xr) in enumerate(line_boxes):
        seg = h_sm[yt:yb]
        print(f"  {i+1:>3}  {yt:>6}  {yb:>6}  {yb-yt:>5}  {float(seg.max()) if len(seg) else 0:>7.1f}")

    v_proj = (binary < 128).sum(axis=0).astype(float)
    v_sm   = uniform_filter(v_proj, size=max(15, W // 80))
    v_max  = float(v_sm.max())
    print(f"\n  V-proj  pico={v_max:.0f}px  cols_activas={int((v_sm > v_max * 0.10).sum())}/{W}")
    for bi, blk in enumerate(block_boxes):
        by_top, by_bot, bx_l, bx_r = blk if len(blk) == 4 else (0, H - 1, blk[0], blk[1])
        bseg = v_sm[bx_l:bx_r]
        print(f"    B{bi+1}  x=[{bx_l}..{bx_r}]  y=[{by_top}..{by_bot}]"
              f"  densidad={float(bseg.mean()) if len(bseg) else 0:.1f}px")


# Punto de entrada por imagen

def run_and_visualize(
    image_path: str,
    debug_dir:  str = str(OUTPUT_DIR),
) -> None:
    """Ejecuta el pipeline y emite diagnóstico + archivos de salida.

    Estructura de salida:
        debug/
            {stem}_lines_detected.jpg   ← imagen global en la raíz
            {stem}/
                line_001.jpg            ← líneas individuales en subcarpeta
                line_002.jpg
                ...
    """
    stem = Path(image_path).stem
    print()
    print(_sep("═"))
    print(f"  Procesando: {image_path}")
    print(_sep("═"))

    out = Path(debug_dir)
    out.mkdir(parents=True, exist_ok=True)

    img_bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"  ERROR: no se pudo leer '{image_path}'")
        return
    h, w = img_bgr.shape[:2]
    print(f"  Imagen: {w}×{h} px")

    m   = analyze(img_bgr)
    cfg = auto_config(img_bgr)

    print(_sep())
    print("  MÉTRICAS")
    print(_sep())
    print(f"  Contraste p95-p5  {m.contrast:>8.1f}  {'← CLAHE' if m.needs_clahe else ''}")
    print(f"  Luminancia media  {m.mean_luminance:>8.1f}")
    print(f"  Fondo oscuro      {'sí' if m.dark_background else 'no':>8}")
    print(f"  Altura texto est. {m.estimated_text_h:>8.1f} px")
    print(f"  Canal óptimo      {m.best_channel:>8}")

    result = run(img_bgr, cfg=cfg)

    if result.warnings:
        print(_sep())
        for wm in result.warnings:
            print(f"  [!]  {wm}")

    if SINGLE_LINE_MODE and result.n_lines != 1:
        print(_sep())
        if result.n_lines == 0:
            print(f"  [!]  SINGLE_LINE_MODE: no se detecto ninguna linea en '{Path(image_path).name}'")
        else:
            print(f"  [!]  SINGLE_LINE_MODE: se esperaba 1 linea pero se detectaron "
                  f"{result.n_lines} en '{Path(image_path).name}'")

    # Reconstruir binary para la visualización
    gray = to_gray(img_bgr, cfg)
    if cfg.deskew:
        gray, angle = deskew_image(gray)
        if abs(angle) > 0.1:
            print(f"\n  Deskew global: {angle:.2f}°")

    binary = binarize(
        img=gray, window=cfg.sauvola_window, k=cfg.sauvola_k,
        use_clahe=cfg.use_clahe, clahe_clip=cfg.clahe_clip, clahe_tile=cfg.clahe_tile,
        invert=cfg.invert_binary,
        use_bilateral=cfg.use_bilateral, bilateral_d=cfg.bilateral_d,
        bilateral_sc=cfg.bilateral_sc, bilateral_ss=cfg.bilateral_ss,
        global_floor_pct=cfg.global_floor_pct,
    )
    if cfg.morph_open > 0 or cfg.morph_close > 0:
        binary = clean_binary(binary, cfg.morph_open, cfg.morph_close)
    if cfg.min_component_area > 0:
        binary = filter_small_components(binary, cfg.min_component_area)

    _print_stats(binary, result.line_boxes, result.block_boxes)

    print(_sep())
    print(f"  LÍNEAS: {result.n_lines}")
    print(_sep())
    print(f"  {'#':>3}  {'y_top':>6}  {'y_bot':>6}  {'x_left':>7}  {'x_right':>8}"
          f"  {'alto':>5}  {'ancho_norm':>11}  {'texto%':>7}")
    print(f"  {'-'*3}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*8}"
          f"  {'-'*5}  {'-'*11}  {'-'*7}")
    for i, ((yt, yb, xl, xr), line) in enumerate(zip(result.line_boxes, result.lines)):
        print(f"  {i+1:>3}  {yt:>6}  {yb:>6}  {xl:>7}  {xr:>8}"
              f"  {yb-yt:>5}  {line.shape[1]:>11}  {100.0*(line<0.5).mean():>6.1f}%")

    print(_sep())
    print("  SALIDA")
    print(_sep())
    if SINGLE_LINE_MODE:
        # Sin imagen de diagnóstico; líneas directamente en el directorio raíz
        n_saved = save_line_images(result.lines, out, prefix=stem + "_")
        print(f"  [OK]   {n_saved} líneas → {stem}_line_NNN.jpg")
    else:
        lines_dir = out / stem
        vis_lines_detected(binary, result.line_boxes, result.block_boxes, out, prefix=stem + "_")
        n_saved = save_line_images(result.lines, lines_dir)
        print(f"  [OK]   {n_saved} líneas → {stem}/line_NNN.jpg")
    print()
    print(_sep("═"))
    print()


# Batch

def _process_one(img_path: Path, output_dir: Path, quiet: bool = False) -> tuple[str, bool, str]:
    """Procesa una sola imagen."""
    try:
        if quiet:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                run_and_visualize(str(img_path), str(output_dir))
        else:
            run_and_visualize(str(img_path), str(output_dir))
        return (img_path.name, True, "")
    except Exception:
        return (img_path.name, False, traceback.format_exc())

def run_batch(input_dir: Path, output_dir: Path) -> None:
    """Procesa todos los .jpg de input_dir (1 imagen a la vez)."""
    if not input_dir.is_dir():
        print(f"ERROR: '{input_dir}' no es una carpeta válida.", file=sys.stderr)
        sys.exit(1)

    _IMG_EXTS = {".jpg", ".jpeg", ".png"}
    images = sorted(p for p in input_dir.iterdir()
                    if p.suffix.lower() in _IMG_EXTS and p.is_file())
    if not images:
        print(f"No se encontraron archivos .jpg/.jpeg/.png en '{input_dir}'.", file=sys.stderr)
        sys.exit(0)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  BATCH  |  {len(images)} imágenes")
    print(f"  Entrada : {input_dir}")
    print(f"  Salida  : {output_dir}")
    print(f"{'='*60}\n")

    t0      = time.time()
    results = []
    for i, img in enumerate(images, 1):
        name, ok, err = _process_one(img, output_dir)
        results.append((name, ok, err))
        print(f"  [{i:>3}/{len(images)}] {'OK' if ok else 'ERROR':5}  {name}")
        if not ok:
            print(f"\nERROR en {name}:\n{err}", file=sys.stderr)

    ok_count  = sum(1 for _, ok, _ in results if ok)
    err_count = len(results) - ok_count
    print(f"\n{'='*60}")
    print(f"  Completado en {time.time()-t0:.1f}s  —  OK: {ok_count}  ERROR: {err_count}")
    print(f"{'='*60}\n")

def main() -> None:
    inp = INPUT_PATH.resolve()
    out = OUTPUT_DIR.resolve()
    if inp.is_dir():
        run_batch(inp, out)
    elif inp.is_file() and inp.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        out.mkdir(parents=True, exist_ok=True)
        run_and_visualize(str(inp), str(out))
    elif inp.is_file():
        print(f"ERROR: INPUT_PATH='{inp}' no es una imagen soportada (.jpg, .jpeg, .png).", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"ERROR: INPUT_PATH='{inp}' no existe.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()