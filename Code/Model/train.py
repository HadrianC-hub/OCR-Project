"""
train.py  —  Entrenamiento CRNN+CTC para Kaggle (GPU + reanudación)

Características:
  ✓ GPU automático con Mixed Precision (AMP) — 3-5× más rápido que FP32
  ✓ Reanudación automática desde checkpoint (optimizer + scheduler incluidos)
  ✓ Métricas exhaustivas al final de cada época y reporte final completo
  ✓ Guardado del mejor modelo por CER + checkpoints periódicos
  ✓ Logging de curvas en JSON para visualización posterior
  ✓ torch.compile activado en Linux (Kaggle) con backend inductor

Rutas Kaggle por defecto:
  Imágenes : /kaggle/input/<dataset>/images/
  Vocab    : /kaggle/input/<dataset>/vocab/vocab.txt
  Salidas  : /kaggle/working/checkpoints/
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from model   import CRNN
from dataset import OCRDataset, collate_fn, decode_ctc, BLANK_IDX, IMG_HEIGHT
from metrics import compute_all_metrics, print_metrics


# ================================================================== #
#  CONFIGURACIÓN  —  ajusta DATASET_NAME antes de lanzar en Kaggle   #
# ================================================================== #
DATASET_NAME = "tu-dataset"   # ← nombre exacto del dataset en Kaggle

CONFIG = {
    # ---- Rutas ----
    "data_dir":        f"/kaggle/input/{DATASET_NAME}/images",
    "vocab_path":      f"/kaggle/input/{DATASET_NAME}/vocab/vocab.txt",
    "checkpoint_dir":  "/kaggle/working/checkpoints",
    "history_path":    "/kaggle/working/training_history.json",

    # ---- Imagen ----
    "img_height":  IMG_HEIGHT,   # 32 px

    # ---- Modelo (mayor capacidad) ----
    "hidden_size": 256,          # 256 por dirección → 512 bidireccional
    "num_layers":  2,            # 2 capas BiLSTM
    "vocab_size":  101,          # 99 símbolos + blank en idx 100
    "dropout":     0.2,          # más dropout por modelo más grande

    # ---- Entrenamiento ----
    "epochs":       50,          # más épocas para modelo más profundo
    "batch_size":   48,          # reducido ligeramente por mayor VRAM usage
    "lr":           3e-4,        # lr más bajo — modelo más grande necesita pasos más finos
    "weight_decay": 1e-4,
    "val_split":    0.1,
    "grad_clip":    5.0,

    # ---- GPU / Mixed Precision ----
    "use_amp":      True,

    # ---- DataLoader ----
    "num_workers":  4,

    # ---- Checkpoints ----
    "save_every":   5,
    "resume":       True,
}
# ================================================================== #


# ------------------------------------------------------------------ #
#  Utilidades                                                         #
# ------------------------------------------------------------------ #

def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        vram  = props.total_memory / 1024**3
        print(f"GPU: {props.name}  |  VRAM: {vram:.1f} GB")
    else:
        dev = torch.device("cpu")
        print("ADVERTENCIA: No se detectó GPU — usando CPU (lento)")
    return dev


def save_checkpoint(path: Path, epoch: int, model, optimizer, scheduler,
                    best_cer: float, metrics: dict, cfg: dict) -> None:
    torch.save({
        "epoch":            epoch,
        "model_state":      model.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
        "scheduler_state":  scheduler.state_dict(),
        "best_cer":         best_cer,
        "metrics":          metrics,
        "config":           cfg,
    }, path)


def load_checkpoint(path: Path, model, optimizer, scheduler, device):
    """Carga estado completo. Retorna (start_epoch, best_cer)."""
    ckpt = torch.load(path, map_location=device)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state"].items()}
    model.load_state_dict(state)
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch = ckpt["epoch"] + 1
    best_cer    = ckpt["best_cer"]
    print(f"Reanudando desde época {start_epoch}  |  Mejor CER previo: {best_cer:.4f}")
    return start_epoch, best_cer


# ------------------------------------------------------------------ #
#  Evaluación con métricas completas                                  #
# ------------------------------------------------------------------ #

def evaluate(model, loader, device, scaler=None) -> dict:
    """Decodifica todas las predicciones y calcula métricas completas."""
    model.eval()
    all_hyps, all_refs = [], []

    with torch.no_grad():
        for images, _, input_lengths, target_lengths, texts in loader:
            images = images.to(device)

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    log_probs = model(images)
            else:
                log_probs = model(images)

            _, best = log_probs.max(2)   # [T, B]
            best = best.permute(1, 0)    # [B, T]

            for i, seq in enumerate(best):
                all_hyps.append(decode_ctc(seq.tolist()))
                all_refs.append(texts[i])

    return compute_all_metrics(all_hyps, all_refs)


# ------------------------------------------------------------------ #
#  Bucle de entrenamiento                                             #
# ------------------------------------------------------------------ #

def train(cfg: dict) -> None:
    device = get_device()

    # Fijar variables de entorno para que dataset.py use las rutas de cfg
    os.environ["OCR_DATA_DIR"]   = cfg["data_dir"]
    os.environ["OCR_VOCAB_PATH"] = cfg["vocab_path"]

    # Importar aquí para que cojan las env vars actualizadas
    import importlib
    import dataset as ds_module
    importlib.reload(ds_module)
    from dataset import OCRDataset, collate_fn, decode_ctc, BLANK_IDX

    print(f"\nDispositivo: {device}  |  PyTorch {torch.__version__}")
    print(f"AMP (Mixed Precision): {'ON' if cfg['use_amp'] and device.type == 'cuda' else 'OFF'}")

    # ---- Dataset -------------------------------------------------- #
    full_ds = OCRDataset(data_dir=cfg["data_dir"], img_height=cfg["img_height"], augment=True)
    n_val   = max(1, int(len(full_ds) * cfg["val_split"]))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    val_ds.dataset.augment = False

    loader_kw = dict(
        collate_fn=collate_fn,
        num_workers=cfg["num_workers"],
        persistent_workers=(cfg["num_workers"] > 0),
        prefetch_factor=2 if cfg["num_workers"] > 0 else None,
        pin_memory=(device.type == "cuda"),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, **loader_kw)
    print(f"Train: {n_train} | Val: {n_val} | Batches/época: {len(train_loader)}")

    # ---- Modelo --------------------------------------------------- #
    model = CRNN(
        vocab_size=cfg["vocab_size"],
        img_height=cfg["img_height"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg.get("num_layers", 2),
        dropout=cfg["dropout"],
    ).to(device)

    # torch.compile desactivado — incompatible con anchos dinámicos en inductor
    print("torch.compile desactivado (anchos de imagen variables)")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros: {n_params:,}")

    # ---- Loss, Optimizer, Scheduler, Scaler ----------------------- #
    ctc_loss  = nn.CTCLoss(blank=BLANK_IDX, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg["lr"],
        steps_per_epoch=len(train_loader),
        epochs=cfg["epochs"],
        pct_start=0.1,
        anneal_strategy="cos",
    )
    use_amp = cfg["use_amp"] and device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    # ---- Checkpoints ---------------------------------------------- #
    ckpt_dir  = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_model.pt"
    start_epoch = 1
    best_cer    = float("inf")

    if cfg["resume"] and best_path.exists():
        start_epoch, best_cer = load_checkpoint(best_path, model, optimizer, scheduler, device)

    # ---- Historial de entrenamiento ------------------------------- #
    history_path = Path(cfg["history_path"])
    history = {"train_loss": [], "val_metrics": []}
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

    # ---- Bucle de entrenamiento ----------------------------------- #
    print(f"\nIniciando entrenamiento desde época {start_epoch}/{cfg['epochs']}\n")
    total_t0 = time.time()

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (images, labels, input_lengths, target_lengths, _) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    log_probs = model(images)
                    loss = ctc_loss(log_probs, labels, input_lengths, target_lengths)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                log_probs = model(images)
                loss = ctc_loss(log_probs, labels, input_lengths, target_lengths)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimizer.step()

            scheduler.step()
            epoch_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                elapsed = time.time() - t0
                eta_min = elapsed / (batch_idx + 1) * (len(train_loader) - batch_idx - 1) / 60
                print(
                    f"  Epoch {epoch:02d} [{batch_idx+1}/{len(train_loader)}] "
                    f"loss={loss.item():.4f}  "
                    f"lr={scheduler.get_last_lr()[0]:.2e}  "
                    f"ETA época: {eta_min:.1f} min"
                )

        # ---- Validación ------------------------------------------- #
        avg_loss  = epoch_loss / len(train_loader)
        epoch_sec = time.time() - t0
        eta_total = (cfg["epochs"] - epoch) * epoch_sec / 60

        metrics = evaluate(model, val_loader, device, scaler)

        # Log corto en consola
        print(
            f"\n[Epoch {epoch:02d}/{cfg['epochs']}] "
            f"loss={avg_loss:.4f}  "
            f"CER={metrics['CER']:.4f}  "
            f"WER={metrics['WER']:.4f}  "
            f"1-NED={metrics['1-NED']:.4f}  "
            f"LineAcc={metrics['LineAcc']:.4f}  "
            f"CharF1={metrics['Char_F1']:.4f}  "
            f"BLEU={metrics['BLEU4_char']:.4f}  "
            f"({epoch_sec:.0f}s  ~{eta_total:.0f} min rest.)"
        )

        # Historial
        history["train_loss"].append(avg_loss)
        history["val_metrics"].append({
            "epoch":    epoch,
            "CER":      metrics["CER"],
            "WER":      metrics["WER"],
            "1-NED":    metrics["1-NED"],
            "LineAcc":  metrics["LineAcc"],
            "Char_F1":  metrics["Char_F1"],
            "BLEU4":    metrics["BLEU4_char"],
        })
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        # Mejor modelo
        if metrics["CER"] < best_cer:
            best_cer = metrics["CER"]
            save_checkpoint(best_path, epoch, model, optimizer, scheduler,
                            best_cer, metrics, cfg)
            print(f"  ✓ Mejor modelo guardado  CER={best_cer:.4f}")

        # Checkpoint periódico (con estado completo para reanudar)
        if epoch % cfg["save_every"] == 0:
            periodic_path = ckpt_dir / f"ckpt_epoch{epoch:03d}.pt"
            save_checkpoint(periodic_path, epoch, model, optimizer, scheduler,
                            best_cer, metrics, cfg)
            print(f"  ✓ Checkpoint guardado: {periodic_path.name}")

    # ---- Reporte final completo ----------------------------------- #
    total_min = (time.time() - total_t0) / 60
    print(f"\nEntrenamiento completado en {total_min:.1f} min")
    print(f"Mejor CER alcanzado: {best_cer:.4f}\n")

    print("Cargando mejor modelo para evaluación final...")
    ckpt = torch.load(best_path, map_location=device)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state"].items()}
    model.load_state_dict(state)

    final_metrics = evaluate(model, val_loader, device, scaler)
    print_metrics(final_metrics, title=f"EVALUACIÓN FINAL — Mejor modelo (época {ckpt['epoch']})")

    # Guardar métricas finales en JSON
    final_path = Path(cfg["checkpoint_dir"]) / "final_metrics.json"
    # CER_by_length tiene claves int -> convertir a str para JSON
    final_serializable = {
        k: ({str(kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v)
        for k, v in final_metrics.items()
    }
    with open(final_path, "w") as f:
        json.dump(final_serializable, f, indent=2)
    print(f"Métricas finales guardadas en: {final_path}")


# ------------------------------------------------------------------ #
#  Entry point                                                        #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento CRNN+CTC (Kaggle / GPU)")
    parser.add_argument("--data_dir",       default=CONFIG["data_dir"])
    parser.add_argument("--vocab_path",     default=CONFIG["vocab_path"])
    parser.add_argument("--epochs",         type=int,   default=CONFIG["epochs"])
    parser.add_argument("--batch_size",     type=int,   default=CONFIG["batch_size"])
    parser.add_argument("--lr",             type=float, default=CONFIG["lr"])
    parser.add_argument("--hidden_size",    type=int,   default=CONFIG["hidden_size"])
    parser.add_argument("--checkpoint_dir", default=CONFIG["checkpoint_dir"])
    parser.add_argument("--no_amp",         action="store_true", help="Desactiva Mixed Precision")
    parser.add_argument("--no_resume",      action="store_true", help="Empieza desde cero")
    args = parser.parse_args()

    cfg = CONFIG.copy()
    cfg.update({k: v for k, v in vars(args).items() if v is not None})
    if args.no_amp:     cfg["use_amp"] = False
    if args.no_resume:  cfg["resume"]  = False

    train(cfg)