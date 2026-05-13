"""
convert_beto_to_onnx.py — Convierte BETO de HuggingFace a un modelo ONNX
cuantizado, listo para usar con onnxruntime sin necesidad de torch en
producción.

EJECUTAR UNA SOLA VEZ, en cualquier máquina con acceso a internet:

    pip install transformers torch onnxruntime onnxscript
    python scripts/convert_beto_to_onnx.py --output models/beto --quantize

Lo que hace:
  1. Descarga BETO de HuggingFace (~440 MB)
  2. Lo exporta a ONNX (~440 MB)
  3. Lo cuantiza a int8 (~110 MB final, calidad casi idéntica, 3× más
     rápido en CPU que el float32)
  4. Verifica que onnxruntime puede cargarlo y hacer inferencia
  5. Guarda modelo + tokenizador + configs en --output

El directorio resultante (~111 MB cuantizado) se copia tal cual al
servidor de producción dentro del proyecto, en models/beto/.

En producción ya solo necesitas dos librerías:

    pip install transformers onnxruntime    # sin torch
"""

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convierte BETO de HuggingFace a ONNX cuantizado.",
    )
    parser.add_argument(
        "--model", default="dccuchile/bert-base-spanish-wwm-cased",
        help="ID HuggingFace o ruta local del modelo (default: %(default)s)",
    )
    parser.add_argument(
        "--output", default="models/beto",
        help="Directorio de salida (default: %(default)s)",
    )
    parser.add_argument(
        "--quantize", action="store_true",
        help="Cuantiza a int8: ~4× más pequeño, ~3× más rápido en CPU, "
             "calidad casi idéntica. Recomendado.",
    )
    parser.add_argument(
        "--opset", type=int, default=18,
        help="Versión del opset ONNX (default: 18, requerido por torch ≥2.9)",
    )
    parser.add_argument(
        "--seq-len", type=int, default=32,
        help="Longitud de la secuencia dummy usada para el trace (default: 32). "
             "El modelo exportado acepta cualquier longitud porque definimos "
             "ejes dinámicos; este parámetro es solo para el trace.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Modelo origen:  {args.model}")
    print(f"Destino:        {out_dir}")
    print(f"Cuantizar:      {args.quantize}")
    print()

    # ── Imports diferidos con mensajes claros si falta algo ────────────
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForMaskedLM
    except ImportError as exc:
        print("ERROR: faltan dependencias.", file=sys.stderr)
        print(file=sys.stderr)
        print("  pip install transformers torch onnxruntime onnxscript",
              file=sys.stderr)
        print(file=sys.stderr)
        print(f"(detalle: {exc})", file=sys.stderr)
        return 1

    # ── 1. Descargar modelo y tokenizador ──────────────────────────────
    print("[1/4] Descargando modelo y tokenizador desde HuggingFace...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForMaskedLM.from_pretrained(args.model)
    except Exception as exc:
        print(f"ERROR al descargar el modelo: {exc}", file=sys.stderr)
        print("Comprueba que tienes acceso a HuggingFace o que el ID/ruta es válido.",
              file=sys.stderr)
        return 1
    model.eval()
    tokenizer.save_pretrained(out_dir)
    print(f"      OK. Tokenizador guardado en {out_dir}")

    # ── 2. Export a ONNX ────────────────────────────────────────────────
    fp32_path = out_dir / ("model_fp32.onnx" if args.quantize else "model.onnx")
    print(f"[2/4] Exportando a ONNX (puede tardar 1-3 min)...")
    print(f"      torch={torch.__version__}, opset={args.opset}")

    dummy = tokenizer(
        "Esto es un texto de prueba para exportar a ONNX",
        return_tensors="pt", padding="max_length", max_length=args.seq_len,
    )

    try:
        torch.onnx.export(
            model,
            (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
            str(fp32_path),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids":      {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "token_type_ids": {0: "batch", 1: "seq"},
                "logits":         {0: "batch", 1: "seq"},
            },
            opset_version=args.opset,
            do_constant_folding=True,
        )
    except ImportError as exc:
        if "onnxscript" in str(exc):
            print("ERROR: torch ≥2.9 necesita onnxscript para export.",
                  file=sys.stderr)
            print("  pip install onnxscript", file=sys.stderr)
            return 1
        raise

    fp32_size = fp32_path.stat().st_size / (1024 * 1024)
    print(f"      OK. {fp32_path.name} = {fp32_size:.1f} MB")

    # ── 2.5 Consolidar external data (torch ≥2.9 escribe pesos aparte) ──
    # Cuando el modelo es algo grande, torch.onnx.export deja los pesos
    # en ficheros .onnx.data o .onnx_data separados, y el .onnx queda
    # con solo el grafo (unos pocos MB). onnxruntime.quantize_dynamic
    # no maneja bien ese formato, así que metemos los pesos dentro del
    # .onnx para que sea un fichero único.
    try:
        import onnx
    except ImportError:
        print("ERROR: onnx no está instalado.", file=sys.stderr)
        print("  pip install onnx", file=sys.stderr)
        return 1

    print(f"      Consolidando pesos externos en un fichero único...")
    model_proto = onnx.load(str(fp32_path), load_external_data=True)
    # Borramos el .onnx y todos los .data residuales
    fp32_path.unlink()
    for pattern in ("*.onnx_data", "*.onnx.data"):
        for f in out_dir.glob(pattern):
            f.unlink(missing_ok=True)
    onnx.save_model(model_proto, str(fp32_path), save_as_external_data=False)
    fp32_size = fp32_path.stat().st_size / (1024 * 1024)
    print(f"      OK. {fp32_path.name} (consolidado) = {fp32_size:.1f} MB")

    # ── 3. Cuantización ────────────────────────────────────────────────
    if args.quantize:
        print("[3/4] Cuantizando a int8 (~1-2 min)...")
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
        except ImportError:
            print("ERROR: onnxruntime no está instalado.", file=sys.stderr)
            print("  pip install onnxruntime", file=sys.stderr)
            return 1

        quant_path = out_dir / "model.onnx"
        # Pasamos el ModelProto ya en memoria en lugar de la ruta al
        # fichero. Así evitamos load_model_with_shape_infer(), que en
        # algunos setups (Windows + Python 3.13 + ciertas versiones de
        # onnx) intenta correr inferencia de formas en disco y falla
        # silenciosamente con FileNotFoundError sobre *-inferred.onnx.
        quantize_dynamic(
            model_proto, str(quant_path),
            weight_type=QuantType.QUInt8,
        )

        # Limpiamos el fp32 intermedio y cualquier .data residual
        for residual in out_dir.glob("model_fp32.onnx*"):
            residual.unlink(missing_ok=True)

        quant_size = quant_path.stat().st_size / (1024 * 1024)
        print(f"      OK. model.onnx (int8) = {quant_size:.1f} MB "
              f"({quant_size/fp32_size*100:.0f}% del fp32)")
    else:
        print("[3/4] Cuantización omitida (sin --quantize).")
        # En modo no-quantize, fp32_path ya se llama model.onnx

    # ── 4. Verificar que ONNX Runtime puede cargarlo ───────────────────
    print("[4/4] Verificando carga e inferencia con onnxruntime...")
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError as exc:
        print(f"WARNING: no se pudo verificar — falta {exc.name}.")
        print("Instala onnxruntime para validar antes de copiar a producción.")
    else:
        session = ort.InferenceSession(
            str(out_dir / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        sess_inputs = {i.name for i in session.get_inputs()}
        test = tokenizer(
            f"Esto es {tokenizer.mask_token} de prueba",
            return_tensors="np",
        )
        feed = {
            "input_ids":      test["input_ids"].astype(np.int64),
            "attention_mask": test["attention_mask"].astype(np.int64),
        }
        if "token_type_ids" in sess_inputs:
            feed["token_type_ids"] = np.zeros_like(test["input_ids"], dtype=np.int64)
        out = session.run(None, feed)
        mask_idx = int((test["input_ids"] == tokenizer.mask_token_id).nonzero()[1][0])
        top_id = int(np.argmax(out[0][0, mask_idx]))
        top_word = tokenizer.decode([top_id])
        print(f"      OK. Inferencia funciona. Predicción top en MASK: {top_word!r}")

    # ── Resumen ────────────────────────────────────────────────────────
    print()
    print("Ficheros generados (cópialos al servidor):")
    total = 0.0
    for f in sorted(out_dir.iterdir()):
        if f.is_file():
            size = f.stat().st_size / (1024 * 1024)
            total += size
            print(f"  {f.name:36s}  {size:>9.2f} MB")
    print(f"  {'TOTAL':36s}  {total:>9.2f} MB")

    print()
    print("Estructura esperada en el servidor de producción:")
    print()
    print("  proyecto/")
    print("  ├── apps/")
    print("  ├── models/")
    print(f"  │   └── beto/      ← contenido de {out_dir}")
    print("  ├── manage.py")
    print("  └── ...")
    print()
    print("En producción, instala solo:")
    print("  pip install transformers onnxruntime")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
