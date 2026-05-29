#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparación cualitativa de decodificadores OCR (greedy vs beam+LM).

Usa exactamente el mismo módulo `ocr_predict.py` que la aplicación web,
incluyendo el lector `ArpaLM` en Python puro (sin necesidad de instalar
el paquete `kenlm` ni compilar nada).

Toma todas las imágenes de la carpeta indicada, las preprocesa con el
mismo pipeline que la web (`autocrop` + resize a 64 px + normalización),
ejecuta el modelo CRNN+CTC una vez por imagen y produce dos
transcripciones: greedy y beam+LM, para inspección visual.

Uso:
    python compare.py
    python compare.py --images-dir mis_lineas/
    python compare.py --output resultado.txt
    python compare.py --no-lm                  # solo greedy
    python compare.py --lm-alpha 0.5
    python compare.py --lm-max-order 3         # trigramas (más RAM, más calidad)
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--images-dir",   default="images",
                   help="Carpeta con imágenes de líneas (.png/.jpg). Default: ./images")
    p.add_argument("--output",       default=None,
                   help="Fichero de salida (texto plano). Default: stdout")
    p.add_argument("--predict-file", default="ocr_predict.py",
                   help="Ruta al ocr_predict.py de tu app web. Default: ./ocr_predict.py")
    p.add_argument("--checkpoint",   default="models/best_model.pt",
                   help="Ruta del checkpoint .pt. Default: ./models/best_model.pt")
    p.add_argument("--vocab",        default="models/vocab.txt",
                   help="Ruta del vocab.txt. Default: ./models/vocab.txt")
    p.add_argument("--lm",           default="models/kenLM.arpa",
                   help="Ruta del kenLM.arpa. Default: ./models/kenLM.arpa")
    p.add_argument("--lm-alpha",     type=float, default=0.4,
                   help="Peso del LM. Default: 0.4 (idéntico al de la web)")
    p.add_argument("--lm-max-order", type=int, default=2,
                   help="Orden máximo del LM (2=bigramas, 3=trigramas). "
                        "Default: 2 (idéntico al de la web)")
    p.add_argument("--beam-width",   type=int, default=10,
                   help="Anchura del haz. Default: 10")
    p.add_argument("--no-lm",        action="store_true",
                   help="Saltar beam+LM (solo greedy)")
    return p.parse_args()


def check_paths(args: argparse.Namespace) -> None:
    """Verifica todas las rutas antes de cargar nada."""
    paths = [
        ("PREDICT_FILE", Path(args.predict_file)),
        ("CHECKPOINT",   Path(args.checkpoint)),
        ("VOCAB",        Path(args.vocab)),
        ("IMAGES_DIR",   Path(args.images_dir)),
    ]
    if not args.no_lm:
        paths.append(("LM", Path(args.lm)))

    missing = [(name, p) for name, p in paths if not p.exists()]
    if missing:
        print("ERROR: las siguientes rutas no existen:", file=sys.stderr)
        for name, p in missing:
            print(f"  {name:15} {p}", file=sys.stderr)
        print("\nRevisa las rutas (--help para ver las opciones).", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    args = parse_args()
    check_paths(args)

    # Añadir la carpeta de ocr_predict.py al path antes de importar
    predict_dir = Path(args.predict_file).resolve().parent
    sys.path.insert(0, str(predict_dir))

    # Importar la misma maquinaria que la app web
    import torch
    from ocr_predict import (
        OCRPredictor,
        preprocess,
        decode_greedy,
        decode_beam,
        _CNN_STRIDE,
    )

    # ── Cargar todo una sola vez ────────────────────────────────────
    print("Cargando modelo y LM (la primera carga del LM tarda 1-2 min)...\n")
    predictor = OCRPredictor(
        checkpoint_path = args.checkpoint,
        vocab_path      = args.vocab,
        beam_width      = args.beam_width,
        beam_bonus      = 2.0,
        length_norm     = 0.65,
        lm_path         = None if args.no_lm else args.lm,
        lm_alpha        = args.lm_alpha,
        lm_max_order    = args.lm_max_order,
        verbose         = True,
    )

    # ── Recolectar imágenes ─────────────────────────────────────────
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    img_dir  = Path(args.images_dir)
    images   = sorted([p for p in img_dir.iterdir()
                       if p.suffix.lower() in img_exts])

    if not images:
        print(f"\nERROR: no hay imágenes en {img_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\nProcesando {len(images)} imágenes...\n")

    # ── Inferencia: una pasada del modelo, dos decodificadores ──────
    out_lines: list[str] = []
    n_diff = 0
    n_ok   = 0

    for img_path in images:
        try:
            tensor = preprocess(img_path, predictor.img_height).to(predictor.device)
            with torch.no_grad():
                log_probs = predictor.model(tensor)         # [T, 1, V]
            valid_t = tensor.shape[3] // _CNN_STRIDE
            lp = log_probs[:valid_t, 0]                     # [T, V]

            # Greedy
            indices    = lp.argmax(dim=1).cpu().tolist()
            hyp_greedy = decode_greedy(indices, predictor.idx2char)

            # Beam + LM (si el LM se cargó correctamente)
            hyp_beam = None
            if predictor.lm is not None or not args.no_lm:
                lp_np = lp.cpu().float().numpy()
                seq   = [lp_np[t].tolist() for t in range(len(lp_np))]
                hyp_beam = decode_beam(
                    seq, predictor.idx2char,
                    beam_width  = args.beam_width,
                    blank_bonus = 2.0,
                    length_norm = 0.65,
                    lm          = predictor.lm,
                    lm_alpha    = args.lm_alpha,
                )
        except Exception as e:
            out_lines.append(f"## {img_path.name}: error ({e})\n")
            continue

        n_ok += 1
        sep = "─" * 72
        out_lines.append(sep)
        out_lines.append(f"Archivo: {img_path.name}")
        out_lines.append(sep)
        out_lines.append(f"  greedy : {hyp_greedy}")
        if hyp_beam is not None:
            out_lines.append(f"  beam+LM: {hyp_beam}")
            if hyp_greedy != hyp_beam:
                out_lines.append(f"  → los decodificadores difieren")
                n_diff += 1
            else:
                out_lines.append(f"  → los decodificadores coinciden")
        out_lines.append("")

    # ── Resumen ─────────────────────────────────────────────────────
    out_lines.append("=" * 72)
    out_lines.append(f"TOTAL: {n_ok} imágenes procesadas con éxito de {len(images)}")
    if predictor.lm is not None:
        pct = 100 * n_diff / max(1, n_ok)
        out_lines.append(f"       {n_diff} con diferencias entre greedy y beam+LM "
                         f"({pct:.1f}%)")
    out_lines.append("=" * 72)

    output = "\n".join(out_lines)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"\nResultado guardado en {args.output}")
        print()
        print("\n".join(out_lines[-3:]))
    else:
        print(output)


if __name__ == "__main__":
    main()
