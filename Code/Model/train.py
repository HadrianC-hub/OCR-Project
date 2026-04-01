"""
train.py  —  Entrenamiento CRNN+CTC optimizado para CPU (Ryzen 5700H)

Optimizaciones aplicadas:
  1. torch.set_num_threads / set_num_interop_threads → usa todos los núcleos
  2. torch.compile (PyTorch ≥ 2.0) → JIT del grafo computacional
  3. OneCycleLR → converge en menos épocas que LR fijo
  4. zero_grad(set_to_none=True) → evita allocation de tensores cero
  5. DataLoader con num_workers > 0 y prefetch_factor → pipeline de carga
  6. No mixed precision (AMP solo ayuda en CUDA; en CPU empeora)
"""

import os
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from model import CRNN
from dataset import OCRDataset, collate_fn, decode_ctc, BLANK_IDX, DATA_DIR, IMG_HEIGHT
from metrics import cer, wer


# ================================================================== #
#  CONFIGURACIÓN — ajusta aquí antes de lanzar                        #
# ================================================================== #
CONFIG = {
    # Datos
    "data_dir":        DATA_DIR,    # Heredado de dataset.py
    "img_height":      IMG_HEIGHT,  # 32 px — optimizado para CPU

    # Modelo
    "hidden_size":     128,   # Por dirección BiLSTM; 256 si quieres más potencia
    "vocab_size":      101,   # 99 símbolos + blank en índice 100

    # Entrenamiento
    "epochs":          30,    # Suficiente con OneCycleLR para texto impreso
    "batch_size":      16,    # Óptimo para CPU (balance cómputo/overhead)
    "lr":              5e-4,  # Pico del OneCycleLR
    "weight_decay":    1e-4,
    "val_split":       0.1,   # 10 % para validación

    # CPU
    "num_threads":     14,    # Hilos para operaciones intra-op (≤ hilos lógicos)
    "num_workers":     4,     # Workers del DataLoader (prefork de imágenes)

    # Checkpoints
    "checkpoint_dir":  "checkpoints/",
    "save_every":      5,     # Guardar checkpoint cada N épocas
}
# ================================================================== #


def setup_cpu(num_threads: int) -> None:
    """Configura PyTorch para usar todos los núcleos disponibles."""
    torch.set_num_threads(num_threads)
    # Inter-op: paraleliza operaciones independientes en el grafo
    torch.set_num_interop_threads(max(1, num_threads // 2))
    # Deshabilita overhead de CUDA (no existe, pero evita checks internos)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    print(f"CPU threads: intra={torch.get_num_threads()}, inter={torch.get_num_interop_threads()}")


def evaluate(model, loader, device):
    model.eval()
    total_cer, total_wer, exact, n = 0.0, 0.0, 0, 0
    with torch.no_grad():
        for images, _, input_lengths, target_lengths, texts in loader:
            images = images.to(device)
            log_probs = model(images)         # [T, B, vocab]
            _, best = log_probs.max(2)        # [T, B]
            best = best.permute(1, 0)         # [B, T]
            for i, seq in enumerate(best):
                pred = decode_ctc(seq.tolist())
                ref  = texts[i]
                total_cer += cer(pred, ref)
                total_wer += wer(pred, ref)
                exact     += int(pred == ref)
                n         += 1
    return {"CER": total_cer / n, "WER": total_wer / n, "line_acc": exact / n}


def train(cfg: dict) -> None:
    device = torch.device("cpu")
    setup_cpu(cfg["num_threads"])
    print(f"Dispositivo: CPU — PyTorch {torch.__version__}")

    # ---- Dataset -------------------------------------------------- #
    full_ds = OCRDataset(
        data_dir=cfg["data_dir"],
        img_height=cfg["img_height"],
        augment=True,
    )
    n_val   = max(1, int(len(full_ds) * cfg["val_split"]))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    # Desactivar augmentación en el split de validación
    val_ds.dataset.augment = False

    loader_kw = dict(
        collate_fn=collate_fn,
        num_workers=cfg["num_workers"],
        persistent_workers=(cfg["num_workers"] > 0),  # Reutiliza workers entre épocas
        prefetch_factor=2 if cfg["num_workers"] > 0 else None,
        pin_memory=False,  # pin_memory solo ayuda con CUDA
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, **loader_kw)
    print(f"Train: {n_train} | Val: {n_val} | Batches/epoch: {len(train_loader)}")

    # ---- Modelo --------------------------------------------------- #
    model = CRNN(
        vocab_size=cfg["vocab_size"],
        img_height=cfg["img_height"],
        hidden_size=cfg["hidden_size"],
    ).to(device)

    # torch.compile en Linux/Mac: acelera el grafo con el backend inductor.
    # En Windows requiere cl.exe (MSVC), que normalmente no está instalado,
    # por lo que lo desactivamos directamente en lugar de fallar en runtime.
    import sys
    if sys.platform != "win32":
        try:
            model = torch.compile(model, backend="inductor", mode="reduce-overhead")
            print("torch.compile activado (inductor backend)")
        except Exception as e:
            print(f"torch.compile no disponible: {e}")
    else:
        print("torch.compile desactivado (Windows — entrena igual sin él)")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros: {n_params:,}")

    # ---- Optimizador y scheduler ---------------------------------- #
    ctc_loss  = nn.CTCLoss(blank=BLANK_IDX, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    # OneCycleLR: calentamiento rápido + decaimiento suave
    # pct_start=0.1 → el 10 % inicial sube el lr; el 90 % restante baja
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg["lr"],
        steps_per_epoch=len(train_loader),
        epochs=cfg["epochs"],
        pct_start=0.1,
        anneal_strategy="cos",
    )

    # ---- Checkpoints ---------------------------------------------- #
    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_cer = float("inf")

    # ---- Bucle de entrenamiento ----------------------------------- #
    total_t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (images, labels, input_lengths, target_lengths, _) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)

            log_probs = model(images)
            loss = ctc_loss(log_probs, labels, input_lengths, target_lengths)

            loss.backward()
            # Clip de gradientes — importante con LSTM
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            # Log cada 100 batches
            if (batch_idx + 1) % 100 == 0:
                elapsed = time.time() - t0
                batches_left = len(train_loader) - batch_idx - 1
                eta = elapsed / (batch_idx + 1) * batches_left
                print(
                    f"  Epoch {epoch:02d} [{batch_idx+1}/{len(train_loader)}] "
                    f"loss={loss.item():.4f}  "
                    f"lr={scheduler.get_last_lr()[0]:.2e}  "
                    f"ETA épocas: {eta/60:.1f} min"
                )

        # ---- Validación ------------------------------------------- #
        avg_loss = epoch_loss / len(train_loader)
        epoch_t  = time.time() - t0
        total_t  = time.time() - total_t0
        remaining_epochs = cfg["epochs"] - epoch
        eta_total = remaining_epochs * epoch_t / 60

        metrics = evaluate(model, val_loader, device)

        print(
            f"[Epoch {epoch:02d}/{cfg['epochs']}] "
            f"loss={avg_loss:.4f}  "
            f"CER={metrics['CER']:.4f}  "
            f"WER={metrics['WER']:.4f}  "
            f"LineAcc={metrics['line_acc']:.4f}  "
            f"({epoch_t:.0f}s/época  ~{eta_total:.0f} min restantes)"
        )

        # Guardar mejor modelo
        if metrics["CER"] < best_cer:
            best_cer = metrics["CER"]
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "cer":         best_cer,
                    "config":      cfg,
                },
                ckpt_dir / "best_model.pt",
            )
            print(f"  ✓ Mejor modelo guardado  CER={best_cer:.4f}")

        # Checkpoint periódico
        if epoch % cfg["save_every"] == 0:
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "metrics": metrics},
                ckpt_dir / f"ckpt_epoch{epoch:03d}.pt",
            )

    total_min = (time.time() - total_t0) / 60
    print(f"\nEntrenamiento completado en {total_min:.1f} min  |  Mejor CER: {best_cer:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento CRNN+CTC (CPU)")
    parser.add_argument("--data_dir",      default=CONFIG["data_dir"])
    parser.add_argument("--epochs",        type=int,   default=CONFIG["epochs"])
    parser.add_argument("--batch_size",    type=int,   default=CONFIG["batch_size"])
    parser.add_argument("--lr",            type=float, default=CONFIG["lr"])
    parser.add_argument("--hidden_size",   type=int,   default=CONFIG["hidden_size"])
    parser.add_argument("--num_threads",   type=int,   default=CONFIG["num_threads"])
    parser.add_argument("--checkpoint_dir", default=CONFIG["checkpoint_dir"])
    args = parser.parse_args()

    cfg = CONFIG.copy()
    cfg.update({k: v for k, v in vars(args).items() if v is not None})
    train(cfg)