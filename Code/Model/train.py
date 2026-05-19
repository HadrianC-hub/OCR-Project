import os
import sys
import json
import time
import argparse
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split

from model   import CRNN
from dataset import (OCRDataset, NoAugSubset, BucketBatchSampler, collate_fn,
                     decode_ctc, decode_ctc_beam,
                     BLANK_IDX, IMG_HEIGHT, get_font_label)
from metrics import (compute_all_metrics, print_metrics,
                     compute_statistical_report, print_statistical_report)


DATASET_NAME = "tu-dataset"

CONFIG = {
    # Rutas
    "data_dir":       f"/kaggle/input/{DATASET_NAME}/images",
    "vocab_path":     f"/kaggle/input/{DATASET_NAME}/vocab/vocab.txt",
    "checkpoint_dir": "/kaggle/working/checkpoints",
    "history_path":   "/kaggle/working/training_history.json",

    # Imagen
    "img_height": IMG_HEIGHT,

    # Modelo
    "hidden_size": 256,
    "num_layers":  2,
    "vocab_size":  101,
    "dropout":     0.2,

    #  MODO DE EJECUCIÓN
    # "train"    → entrenamiento desde cero
    # "finetune" → fine-tuning desde un checkpoint existente
    "mode": "train",

    # Qué etapas ejecutar
    "run_training": True,   # bucle de entrenamiento / fine-tuning principal
    "run_cv":       False,  # validación cruzada k-fold
    "run_lofo":     False,  # Leave-One-Font-Out

    #  PARÁMETROS DE ENTRENAMIENTO NORMAL (mode="train")
    "epochs":       35,
    "batch_size":   32,
    "lr":           3e-4,
    "weight_decay": 1e-4,
    "val_split":    0.1,
    "grad_clip":    5.0,

    #  PARÁMETROS DE FINE-TUNING (mode="finetune")
    # Checkpoint del modelo pre-entrenado a cargar
    "finetune_checkpoint": "/kaggle/input/mi-modelo/checkpoints/best_model.pt",

    # Hiperparámetros específicos de fine-tuning
    "finetune_epochs":       20,     # menos épocas: el modelo ya sabe leer
    "finetune_lr":           5e-5,   # LR ≈ 1/6 del LR original (3e-4)
    "finetune_weight_decay": 1e-4,   # mantener regularización
    "finetune_grad_clip":    3.0,    # gradiente más conservador

    # Congelar CNN las primeras N épocas (0 = no congelar nunca)
    # Útil si el nuevo dataset es muy diferente: el RNN se adapta primero
    # Para tu caso (mismas fuentes + nuevas) recomiendo 0 o 2
    "freeze_cnn_epochs":     0,

    #  GPU / CARGA DE DATOS
    "use_amp":         True,
    "num_workers":     2,
    "prefetch_factor": 2,

    # Decodificación
    "beam_width":  10,
    "beam_bonus":  2.0,
    "beam_alpha":  0.65,

    # KenLM
    "corpus_dir":  f"/kaggle/input/{DATASET_NAME}/Corpus",
    "lm_path":     "/kaggle/working/lm_5gram.arpa",
    "lm_alpha_lm": 0.4,

    # Estadística
    "n_bootstrap": 10_000,

    # CV
    "cv_folds":    5,
    "cv_epochs":   10,

    # LOFO
    "lofo_epochs": 10,

    # Optimizaciones
    "skip_beam_final": True,  # beam empeora en este modelo, ahorrar ~13 min

    # Checkpoints
    "save_every": 5,
    "resume":     True,  # reanudar desde best_model.pt si existe (solo mode="train")
}

# Helpers de LM

def build_lm(corpus_dir: str, lm_path: str, order: int = 5) -> bool:
    import subprocess, tempfile
    corpus = Path(corpus_dir)
    if not corpus.exists():
        print(f"[LM] corpus_dir no encontrado: {corpus_dir} — omitiendo build_lm")
        return False
    txt_files = sorted(corpus.glob("**/*.txt"))
    if not txt_files:
        print(f"[LM] No se encontraron .txt en {corpus_dir} — omitiendo build_lm")
        return False
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        combined = fh.name
        for f in txt_files:
            fh.write(f.read_text(encoding="utf-8", errors="replace"))
            fh.write("\n")
    print(f"[LM] Entrenando modelo {order}-gram con {len(txt_files)} archivos → {lm_path}")
    cmd = f"lmplz -o {order} --discount_fallback < {combined} > {lm_path}"
    ret = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if ret.returncode != 0:
        print(f"[LM] lmplz falló:\n{ret.stderr}")
        return False
    print(f"[LM] Modelo guardado en {lm_path}")
    return True


def load_lm(lm_path: str, lm_alpha: float):
    try:
        import kenlm
        p = Path(lm_path)
        if not p.exists():
            print(f"[LM] Archivo no encontrado: {lm_path} — beam sin LM")
            return None, 0.0
        model = kenlm.Model(str(p))
        print(f"[LM] Modelo cargado: {lm_path}  |  peso α={lm_alpha}")
        return model, lm_alpha
    except ImportError:
        print("[LM] kenlm no instalado — beam sin LM")
        return None, 0.0


# Device

def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev   = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        vram  = props.total_memory / 1024**3
        print(f"GPU: {props.name}  |  VRAM: {vram:.1f} GB")
    else:
        dev = torch.device("cpu")
        print("ADVERTENCIA: No se detectó GPU — usando CPU (lento)")
    return dev


# Checkpoints

def save_checkpoint(path, epoch, model, optimizer, scheduler, best_cer, metrics, cfg):
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_cer":        best_cer,
        "metrics":         metrics,
        "config":          cfg,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt  = torch.load(path, map_location=device)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state"].items()}
    model.load_state_dict(state)
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        except Exception:
            print("  [WARN] No se pudo restaurar el estado del scheduler.")
    start_epoch = ckpt["epoch"] + 1
    best_cer    = ckpt["best_cer"]
    print(f"Reanudando desde época {start_epoch}  |  Mejor CER previo: {best_cer:.4f}")
    return start_epoch, best_cer


def load_pretrained_weights(path: str | Path, model: nn.Module, device: torch.device) -> dict:
    """
    Carga solo los pesos del modelo desde un checkpoint.
    Devuelve el dict de config guardado (útil para verificar arquitectura).
    """
    ckpt  = torch.load(path, map_location=device)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [WARN] Parámetros faltantes en checkpoint: {missing}")
    if unexpected:
        print(f"  [WARN] Parámetros inesperados en checkpoint: {unexpected}")
    print(f"  Pesos cargados desde '{Path(path).name}'  "
          f"(entrenado en época {ckpt.get('epoch', '?')}  "
          f"CER={ckpt.get('best_cer', '?'):.4f})")
    return ckpt.get("config", {})


# Freeze / Unfreeze CNN

def _set_cnn_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    for param in model.cnn.parameters():
        param.requires_grad = requires_grad
    status = "DESCONGELADA" if requires_grad else "CONGELADA"
    n = sum(p.numel() for p in model.cnn.parameters())
    print(f"  CNN {status} ({n:,} parámetros)")


# Evaluación

def evaluate_greedy(model, loader, device, use_amp: bool = False):
    model.eval()
    hyps, refs = [], []
    with torch.no_grad():
        for images, _, input_lengths, _, texts in loader:
            images = images.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast('cuda'):
                    log_probs = model(images)
            else:
                log_probs = model(images)
            indices = log_probs.argmax(dim=2)
            B = indices.shape[1]
            for i in range(B):
                vt = input_lengths[i].item()
                hyps.append(decode_ctc(indices[:vt, i].cpu().tolist()))
                refs.append(texts[i])
    return compute_all_metrics(hyps, refs), hyps, refs


def evaluate_beam(model, loader, device, beam_width: int = 10,
                  beam_bonus: float = 2.0, beam_alpha: float = 0.65,
                  use_amp: bool = False, lm=None, lm_alpha: float = 0.4):
    model.eval()
    hyps, refs = [], []
    with torch.no_grad():
        for images, _, input_lengths, _, texts in loader:
            images = images.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast('cuda'):
                    log_probs = model(images)
            else:
                log_probs = model(images)
            lp_np = log_probs.cpu().float().numpy()
            B = lp_np.shape[1]
            for i in range(B):
                vt  = input_lengths[i].item()
                seq = [lp_np[t, i].tolist() for t in range(vt)]
                hyps.append(decode_ctc_beam(
                    seq, beam_width=beam_width,
                    blank_bonus=beam_bonus,
                    length_norm_alpha=beam_alpha,
                    lm=lm, lm_alpha=lm_alpha,
                ))
                refs.append(texts[i])
    return compute_all_metrics(hyps, refs), hyps, refs


# Construcción de modelo nuevo

def _build_model(cfg: dict, device: torch.device) -> nn.Module:
    model = CRNN(
        vocab_size  = cfg["vocab_size"],
        img_height  = cfg["img_height"],
        hidden_size = cfg["hidden_size"],
        num_layers  = cfg.get("num_layers", 2),
        dropout     = cfg["dropout"],
    ).to(device)
    return model


# Función de entrenamiento interno (usada por CV y LOFO)

def _train_model(
    train_loader, val_loader, cfg: dict, device: torch.device,
    ckpt_path: Path, desc: str = "", skip_beam: bool = False,
    lm=None, lm_alpha: float = 0.4,
    pretrained_path: str | Path | None = None,
) -> tuple:
    """
    Entrena o hace fine-tuning de un modelo CRNN+CTC.
    Si pretrained_path != None, carga pesos antes de entrenar (warm start).
    Devuelve (best_metrics, hyps_greedy, hyps_beam, refs).
    """
    use_amp  = cfg.get("use_amp", True) and device.type == "cuda"
    n_epochs = cfg["epochs"]
    prefix   = f"[{desc}] " if desc else ""

    model = _build_model(cfg, device)

    if pretrained_path is not None and Path(pretrained_path).exists():
        try:
            load_pretrained_weights(pretrained_path, model, device)
        except Exception as e:
            print(f"  {prefix}→ Warm start falló ({e}), inicio desde cero")

    ctc_loss  = nn.CTCLoss(blank=BLANK_IDX, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg["lr"],
        steps_per_epoch=len(train_loader),
        epochs=n_epochs,
        pct_start=0.1, anneal_strategy="cos",
    )
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    best_cer     = float("inf")
    best_metrics = {}
    best_hyps_g  = []
    best_refs    = []
    t_global     = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for images, labels, input_lengths, target_lengths, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    lp   = model(images)
                    loss = ctc_loss(lp, labels, input_lengths, target_lengths)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scale_before = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                if scaler.get_scale() == scale_before:
                    scheduler.step()
            else:
                lp   = model(images)
                loss = ctc_loss(lp, labels, input_lengths, target_lengths)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimizer.step()
                scheduler.step()
            epoch_loss += loss.item()

        avg_loss  = epoch_loss / len(train_loader)
        elapsed   = time.time() - t_global
        eta_min   = elapsed / epoch * (n_epochs - epoch) / 60

        metrics_e, hyps_e, refs_e = evaluate_greedy(model, val_loader, device, use_amp)
        print(
            f"{prefix}Epoch {epoch:02d}/{n_epochs}  "
            f"loss={avg_loss:.4f}  CER={metrics_e['CER']:.4f}  "
            f"LineAcc={metrics_e['LineAcc']:.4f}  "
            f"(ETA: {eta_min:.1f} min)"
        )

        if metrics_e["CER"] < best_cer:
            best_cer     = metrics_e["CER"]
            best_metrics = metrics_e
            best_hyps_g  = hyps_e
            best_refs    = refs_e
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "best_cer": best_cer, "metrics": metrics_e, "config": cfg,
            }, ckpt_path)
            print(f"  ✓ Mejor CER: {best_cer:.6f}")

    ckpt  = torch.load(ckpt_path, map_location=device)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state"].items()}
    model.load_state_dict(state)

    if skip_beam:
        best_hyps_b = best_hyps_g
    else:
        _, best_hyps_b, _ = evaluate_beam(
            model, val_loader, device,
            beam_width=cfg["beam_width"],
            beam_bonus=cfg["beam_bonus"],
            beam_alpha=cfg["beam_alpha"],
            use_amp=use_amp, lm=lm, lm_alpha=lm_alpha,
        )

    model.cpu()
    del model, optimizer, scheduler
    if scaler: del scaler
    torch.cuda.empty_cache()

    return best_metrics, best_hyps_g, best_hyps_b, best_refs


# Bucle principal: entrenamiento o fine-tuning

def _run_main_training(cfg: dict, full_ds, train_loader, val_loader,
                       val_ds_indices: list, device: torch.device,
                       ckpt_dir: Path, lm_model, lm_alpha: float,
                       n_train: int, n_val: int) -> Path:
    """
    Ejecuta el bucle de entrenamiento principal (train o finetune).
    Devuelve la ruta al mejor checkpoint guardado.
    """
    use_amp          = cfg["use_amp"] and device.type == "cuda"
    is_finetune      = cfg.get("mode", "train") == "finetune"
    skip_beam_final  = cfg.get("skip_beam_final", True)
    best_path        = ckpt_dir / "best_model.pt"

    # Hiperparámetros según modo
    if is_finetune:
        n_epochs     = cfg["finetune_epochs"]
        lr           = cfg["finetune_lr"]
        weight_decay = cfg.get("finetune_weight_decay", cfg["weight_decay"])
        grad_clip    = cfg.get("finetune_grad_clip", cfg["grad_clip"])
        freeze_n     = cfg.get("freeze_cnn_epochs", 0)
    else:
        n_epochs     = cfg["epochs"]
        lr           = cfg["lr"]
        weight_decay = cfg["weight_decay"]
        grad_clip    = cfg["grad_clip"]
        freeze_n     = 0

    # Crear modelo
    model = _build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros totales: {n_params:,}")

    # Cargar pesos pre-entrenados (fine-tuning)
    if is_finetune:
        ft_ckpt = cfg.get("finetune_checkpoint", "")
        if not ft_ckpt or not Path(ft_ckpt).exists():
            raise FileNotFoundError(
                f"Fine-tuning requiere un checkpoint válido.\n"
                f"  CONFIG['finetune_checkpoint'] = '{ft_ckpt}' no existe."
            )
        pretrained_cfg = load_pretrained_weights(ft_ckpt, model, device)
        print(f"  Arquitectura pre-entrenada: {pretrained_cfg}")

        if freeze_n > 0:
            _set_cnn_requires_grad(model, requires_grad=False)
            print(f"  CNN congelada las primeras {freeze_n} épocas")

    ctc_loss  = nn.CTCLoss(blank=BLANK_IDX, reduction="mean", zero_infinity=True)

    # Optimizer
    # En fine-tuning: parámetros de CNN con LR aún más baja si están activos
    if is_finetune and freeze_n == 0:
        # Layer-wise LR: CNN aprende más despacio que RNN/FC
        optimizer = torch.optim.AdamW([
            {"params": model.cnn.parameters(),           "lr": lr * 0.3},
            {"params": model.rnn.parameters(),           "lr": lr},
            {"params": model.adaptive_pool.parameters(), "lr": lr},
            {"params": model.fc.parameters(),            "lr": lr},
        ], weight_decay=weight_decay)
        print(f"  Layer-wise LR → CNN: {lr*0.3:.2e}  |  RNN/FC: {lr:.2e}")
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=weight_decay
        )

    # Scheduler
    if is_finetune:
        # CosineAnnealingWarmRestarts: suave, adecuado para fine-tuning
        # T_0 = la mitad de las épocas, T_mult=1 → dos cosenos completos
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(1, n_epochs // 2), T_mult=1, eta_min=lr * 0.01
        )
        scheduler_step_batch = False   # este scheduler se llama por época
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr,
            steps_per_epoch=len(train_loader),
            epochs=n_epochs,
            pct_start=0.1, anneal_strategy="cos",
        )
        scheduler_step_batch = True    # OneCycleLR se llama por batch

    scaler    = torch.amp.GradScaler('cuda') if use_amp else None
    best_cer  = float("inf")
    start_epoch = 1

    # Reanudar entrenamiento normal (no fine-tuning)
    if not is_finetune and cfg["resume"] and best_path.exists():
        start_epoch, best_cer = load_checkpoint(
            best_path, model, optimizer, None, device
        )
        # Reconstruir scheduler para las épocas restantes
        remaining = n_epochs - start_epoch + 1
        if remaining > 0 and scheduler_step_batch:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=lr,
                steps_per_epoch=len(train_loader),
                epochs=remaining,
                pct_start=0.05, anneal_strategy="cos",
            )
            print(f"  Scheduler reconstruido para {remaining} épocas restantes")

    history_path = Path(cfg["history_path"])
    history = {"train_loss": [], "val_metrics": [], "mode": cfg.get("mode", "train")}
    if history_path.exists():
        try:
            with open(history_path) as f:
                history = json.load(f)
        except Exception:
            pass

    mode_label = "FINE-TUNING" if is_finetune else "ENTRENAMIENTO"
    print(f"\n{'═'*66}")
    print(f"  {mode_label} — {n_epochs} épocas  |  LR={lr:.2e}  |  "
          f"AMP={'ON' if use_amp else 'OFF'}")
    if is_finetune and freeze_n > 0:
        print(f"  CNN congelada épocas 1-{freeze_n}, luego descongelada")
    print(f"{'═'*66}\n")

    total_t0 = time.time()
    best_epoch   = start_epoch
    best_hyps_g  = []
    best_refs    = []
    best_metrics = {}

    for epoch in range(start_epoch, n_epochs + 1):

        # Descongelar CNN tras freeze_n épocas (fine-tuning)
        if is_finetune and freeze_n > 0 and epoch == freeze_n + 1:
            _set_cnn_requires_grad(model, requires_grad=True)
            # Añadir parámetros de CNN al optimizer con LR reducida
            for g in optimizer.param_groups:
                g["lr"] = g.get("lr", lr) * 0.5   # CNN empieza con LR/2
            print(f"  Época {epoch}: CNN descongelada, LR ajustada")

        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (images, labels, input_lengths, target_lengths, _) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    log_probs = model(images)
                    loss = ctc_loss(log_probs, labels, input_lengths, target_lengths)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), grad_clip
                )
                scale_before = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                if scheduler_step_batch and scaler.get_scale() == scale_before:
                    scheduler.step()
            else:
                log_probs = model(images)
                loss = ctc_loss(log_probs, labels, input_lengths, target_lengths)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), grad_clip
                )
                optimizer.step()
                if scheduler_step_batch:
                    scheduler.step()
            epoch_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                elapsed  = time.time() - t0
                eta_min  = elapsed / (batch_idx + 1) * (len(train_loader) - batch_idx - 1) / 60
                cur_lr   = optimizer.param_groups[-1]["lr"]
                print(
                    f"  Epoch {epoch:02d} [{batch_idx+1}/{len(train_loader)}] "
                    f"loss={loss.item():.4f}  lr={cur_lr:.2e}  ETA época: {eta_min:.1f} min"
                )

        # Scheduler por época (CosineAnnealingWarmRestarts)
        if not scheduler_step_batch:
            scheduler.step()

        avg_loss  = epoch_loss / len(train_loader)
        train_sec = time.time() - t0
        cur_lr    = optimizer.param_groups[-1]["lr"]

        metrics, hyps_ep, refs_ep = evaluate_greedy(model, val_loader, device, use_amp)
        eta_total = (n_epochs - epoch) * train_sec / 60

        print(
            f"\n[Epoch {epoch:02d}/{n_epochs}]  "
            f"loss={avg_loss:.4f}  CER={metrics['CER']:.4f}  "
            f"WER={metrics['WER']:.4f}  LineAcc={metrics['LineAcc']:.4f}  "
            f"CharF1={metrics['Char_F1']:.4f}  BLEU={metrics['BLEU4_char']:.4f}  "
            f"lr={cur_lr:.2e}  (~{eta_total:.0f} min rest.)"
        )

        history["train_loss"].append(avg_loss)
        history["val_metrics"].append({
            "epoch": epoch, "mode": cfg.get("mode", "train"),
            "CER": metrics["CER"], "WER": metrics["WER"],
            "LineAcc": metrics["LineAcc"], "Char_F1": metrics["Char_F1"],
            "BLEU4": metrics["BLEU4_char"],
        })
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        if metrics["CER"] < best_cer:
            best_cer     = metrics["CER"]
            best_epoch   = epoch
            best_metrics = metrics
            best_hyps_g  = hyps_ep
            best_refs    = refs_ep
            save_checkpoint(best_path, epoch, model, optimizer, scheduler,
                            best_cer, metrics, cfg)
            print(f"  ✓ Mejor modelo guardado  CER={best_cer:.4f}")

        if epoch % cfg["save_every"] == 0:
            periodic = ckpt_dir / f"ckpt_epoch{epoch:03d}.pt"
            save_checkpoint(periodic, epoch, model, optimizer, scheduler,
                            best_cer, metrics, cfg)
            print(f"  ✓ Checkpoint: {periodic.name}")

    total_min = (time.time() - total_t0) / 60
    print(f"\n{mode_label} completado en {total_min:.1f} min  |  "
          f"Mejor CER: {best_cer:.4f}  (época {best_epoch})\n")

    # Evaluación final con mejor modelo
    print("Cargando mejor modelo para evaluación final...")
    ckpt  = torch.load(best_path, map_location=device)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state"].items()}
    model.load_state_dict(state)

    greedy_metrics, hyps_g, refs = evaluate_greedy(model, val_loader, device, use_amp)
    print_metrics(greedy_metrics, title=f"EVALUACIÓN FINAL [greedy] — época {best_epoch}")

    if skip_beam_final:
        print("\n[INFO] skip_beam_final=True — beam omitido.")
        hyps_b      = hyps_g
        beam_metrics = greedy_metrics
    else:
        print(f"Beam search final (w={cfg['beam_width']})...")
        beam_metrics, hyps_b, _ = evaluate_beam(
            model, val_loader, device,
            beam_width=cfg["beam_width"],
            beam_bonus=cfg["beam_bonus"],
            beam_alpha=cfg["beam_alpha"],
            use_amp=use_amp, lm=lm_model, lm_alpha=lm_alpha,
        )
        print_metrics(beam_metrics,
                      title=f"EVALUACIÓN FINAL [beam={cfg['beam_width']}] — época {best_epoch}")

    font_lbls = [get_font_label(full_ds.samples[i][0]) for i in val_ds_indices]

    print(f"\n{'═'*66}")
    print(f"  ANÁLISIS ESTADÍSTICO (n_bootstrap={cfg['n_bootstrap']:,})")
    print(f"{'═'*66}\n")
    stat_report = compute_statistical_report(
        hyps_g, hyps_b, refs,
        font_labels=font_lbls,
        n_bootstrap=cfg["n_bootstrap"],
    )
    print_statistical_report(stat_report)

    def _ser(m):
        return {k: ({str(kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v)
                for k, v in m.items()}

    final_report = {
        "training": {
            "mode":           cfg.get("mode", "train"),
            "best_epoch":     best_epoch,
            "best_cer_greedy": best_cer,
            "total_time_min": total_min,
            "n_train":        n_train,
            "n_val":          n_val,
            "skip_beam_final": skip_beam_final,
            "finetune_checkpoint": cfg.get("finetune_checkpoint", None),
        },
        "greedy_metrics": _ser(greedy_metrics),
        "beam_metrics":   _ser(beam_metrics),
        "statistical":    {k: v for k, v in stat_report.items() if k != "per_sample"},
        "history":        history,
    }
    final_path = ckpt_dir / "final_metrics.json"
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    print(f"  Métricas guardadas en: {final_path}")

    model.cpu()
    del model, optimizer, scheduler
    if scaler: del scaler
    torch.cuda.empty_cache()

    return best_path


# Validación cruzada k-fold

def run_cross_validation(cfg: dict, full_ds: OCRDataset,
                         ckpt_dir: Path, device: torch.device,
                         pretrained_path: str | Path | None = None) -> dict:
    import numpy as np
    k    = cfg["cv_folds"]
    seed = cfg.get("val_seed", 42)

    print(f"\n{'═'*66}")
    print(f"  VALIDACIÓN CRUZADA {k}-FOLD ESTRATIFICADA POR FUENTE")
    if pretrained_path:
        print(f"  Warm start: {Path(pretrained_path).name}")
    print(f"{'═'*66}")

    font_to_idx = defaultdict(list)
    for idx, (img_path, _) in enumerate(full_ds.samples):
        font_to_idx[get_font_label(img_path)].append(idx)

    fonts = sorted(font_to_idx.keys())
    print(f"\n  Fuentes ({len(fonts)}): {', '.join(fonts)}")

    cv_epochs = cfg.get("cv_epochs", cfg.get("finetune_epochs", cfg["epochs"]))
    print(f"  Épocas por fold: {cv_epochs}")

    rng = np.random.default_rng(seed=seed)
    fold_indices = [[] for _ in range(k)]
    for font, idxs in font_to_idx.items():
        arr = list(idxs); rng.shuffle(arr)
        for i, idx in enumerate(arr):
            fold_indices[i % k].append(idx)

    loader_kw = dict(
        collate_fn=collate_fn,
        num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    fold_results = []
    cv_ckpt_dir  = ckpt_dir / "cv_folds"
    cv_ckpt_dir.mkdir(exist_ok=True)
    partial_path = ckpt_dir / "cv_partial_results.json"
    wc = full_ds._width_cache if full_ds._width_cache else None

    for fold_idx in range(k):
        print(f"\n{'─'*66}  FOLD {fold_idx+1}/{k}")

        val_idx   = fold_indices[fold_idx]
        train_idx = [i for j, fold in enumerate(fold_indices)
                     if j != fold_idx for i in fold]

        train_subset = Subset(full_ds, train_idx)
        val_subset   = NoAugSubset(full_ds, val_idx)

        train_sampler = BucketBatchSampler(train_subset, batch_size=cfg["batch_size"],
                                           shuffle=True, width_cache=wc)
        val_sampler   = BucketBatchSampler(val_subset,   batch_size=cfg["batch_size"],
                                           shuffle=False, width_cache=wc)
        train_loader  = DataLoader(train_subset, batch_sampler=train_sampler, **loader_kw)
        val_loader    = DataLoader(val_subset,   batch_sampler=val_sampler,   **loader_kw)

        fold_ckpt = cv_ckpt_dir / f"fold_{fold_idx+1}_best.pt"
        fold_cfg  = {
            **cfg,
            "epochs":      cv_epochs,
            "lr":          cfg.get("finetune_lr" if cfg.get("mode")=="finetune" else "lr", cfg["lr"]),
            "weight_decay": cfg.get("finetune_weight_decay", cfg["weight_decay"]),
            "grad_clip":   cfg.get("finetune_grad_clip", cfg["grad_clip"]),
        }
        t0 = time.time()
        best_m, hyps_g, _, refs = _train_model(
            train_loader, val_loader, fold_cfg, device,
            fold_ckpt, desc=f"Fold {fold_idx+1}/{k}", skip_beam=True,
            pretrained_path=pretrained_path,
        )
        elapsed = (time.time() - t0) / 60

        fold_result = {
            "fold":        fold_idx + 1,
            "n_val":       len(val_idx),
            "CER":         best_m["CER"],
            "WER":         best_m["WER"],
            "LineAcc":     best_m["LineAcc"],
            "Char_F1":     best_m["Char_F1"],
            "BLEU4_char":  best_m["BLEU4_char"],
            "elapsed_min": elapsed,
        }
        fold_results.append(fold_result)
        print(f"  CER={best_m['CER']:.6f}  LineAcc={best_m['LineAcc']:.6f}  ({elapsed:.1f} min)")

        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump({"completed_folds": fold_idx + 1, "fold_results": fold_results}, f, indent=2)

    import numpy as np
    from scipy import stats as _stats

    print(f"\n{'═'*66}\n  RESUMEN {k}-FOLD\n{'═'*66}")
    print(f"\n  {'Métrica':<20} {'Media':>9} {'Std':>9} {'Min':>9} {'Max':>9} {'IC95-':>9} {'IC95+':>9}")
    print("─" * 66)

    summary = {}
    for key in ["CER", "WER", "LineAcc", "Char_F1", "BLEU4_char"]:
        vals   = np.array([r[key] for r in fold_results])
        mean, std = float(vals.mean()), float(vals.std())
        vmin, vmax = float(vals.min()), float(vals.max())
        t_crit = _stats.t.ppf(0.975, df=k - 1)
        margin = t_crit * std / (k ** 0.5)
        cv_pct = (std / mean * 100) if mean > 0 else float("inf")
        summary[key] = {
            "mean": mean, "std": std, "min": vmin, "max": vmax,
            "ci95_low": mean - margin, "ci95_high": mean + margin,
            "cv_pct": cv_pct, "values": vals.tolist(),
        }
        print(f"  {key:<20} {mean:>9.6f} {std:>9.6f} {vmin:>9.6f} {vmax:>9.6f} "
              f"{mean-margin:>9.6f} {mean+margin:>9.6f}")

    print(f"\n  Resultado para tesis:")
    for key in ["CER", "LineAcc", "Char_F1"]:
        s = summary[key]
        print(f"  {key}: {s['mean']:.4f} ± {s['std']:.4f}  "
              f"IC95%=[{s['ci95_low']:.4f}, {s['ci95_high']:.4f}]  "
              f"CV={s['cv_pct']:.1f}%")

    return {"k": k, "summary": summary, "fold_results": fold_results}


# Leave-One-Font-Out

def run_lofo(cfg: dict, full_ds: OCRDataset,
             ckpt_dir: Path, device: torch.device,
             pretrained_path: str | Path | None = None) -> dict:
    import numpy as np
    from scipy import stats as _stats

    seed = cfg.get("val_seed", 42)
    rng  = np.random.default_rng(seed=seed)

    print(f"\n{'═'*66}")
    print(f"  LEAVE-ONE-FONT-OUT (LOFO)")
    if pretrained_path:
        print(f"  Warm start: {Path(pretrained_path).name}")
    print(f"{'═'*66}")

    font_to_idx = defaultdict(list)
    for idx, (img_path, _) in enumerate(full_ds.samples):
        font_to_idx[get_font_label(img_path)].append(idx)

    all_fonts   = sorted(font_to_idx.keys())
    lofo_epochs = cfg.get("lofo_epochs", cfg.get("finetune_epochs", cfg["epochs"]))
    print(f"  Fuentes ({len(all_fonts)}): {', '.join(all_fonts)}")
    print(f"  Épocas por iteración: {lofo_epochs}")

    loader_kw = dict(
        collate_fn=collate_fn,
        num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    wc           = full_ds._width_cache if full_ds._width_cache else None
    lofo_results = []
    lofo_ckpt_dir = ckpt_dir / "lofo"
    lofo_ckpt_dir.mkdir(exist_ok=True)
    partial_path = ckpt_dir / "lofo_partial_results.json"

    for held_font in all_fonts:
        print(f"\n{'─'*66}\n  Fuente retenida: {held_font}")

        test_idx  = font_to_idx[held_font]
        other_idx = [i for f, idxs in font_to_idx.items()
                     if f != held_font for i in idxs]

        arr = np.array(other_idx); rng.shuffle(arr)
        n_val_int   = max(1, int(len(arr) * cfg["val_split"]))
        val_int_idx = arr[:n_val_int].tolist()
        train_idx   = arr[n_val_int:].tolist()

        print(f"  Train: {len(train_idx):,}  Val-int: {len(val_int_idx):,}"
              f"  Test ({held_font}): {len(test_idx):,}")

        train_subset = Subset(full_ds, train_idx)
        val_subset   = NoAugSubset(full_ds, val_int_idx)
        test_subset  = NoAugSubset(full_ds, test_idx)

        train_sampler = BucketBatchSampler(train_subset, batch_size=cfg["batch_size"],
                                           shuffle=True,  width_cache=wc)
        val_sampler   = BucketBatchSampler(val_subset,   batch_size=cfg["batch_size"],
                                           shuffle=False, width_cache=wc)
        test_sampler  = BucketBatchSampler(test_subset,  batch_size=cfg["batch_size"],
                                           shuffle=False, width_cache=wc)

        train_loader = DataLoader(train_subset, batch_sampler=train_sampler, **loader_kw)
        val_loader   = DataLoader(val_subset,   batch_sampler=val_sampler,   **loader_kw)
        test_loader  = DataLoader(test_subset,  batch_sampler=test_sampler,  **loader_kw)

        lofo_ckpt    = lofo_ckpt_dir / f"lofo_{held_font[:30]}_best.pt"
        lofo_run_cfg = {
            **cfg,
            "epochs":       lofo_epochs,
            "lr":           cfg.get("finetune_lr" if cfg.get("mode")=="finetune" else "lr", cfg["lr"]),
            "weight_decay": cfg.get("finetune_weight_decay", cfg["weight_decay"]),
            "grad_clip":    cfg.get("finetune_grad_clip", cfg["grad_clip"]),
        }

        t0 = time.time()
        _train_model(
            train_loader, val_loader, lofo_run_cfg, device,
            lofo_ckpt, desc=f"LOFO:{held_font[:20]}", skip_beam=True,
            pretrained_path=pretrained_path,
        )
        elapsed = (time.time() - t0) / 60

        ckpt = torch.load(lofo_ckpt, map_location=device)
        eval_model = _build_model(cfg, device)
        state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state"].items()}
        eval_model.load_state_dict(state)

        use_amp_lofo = cfg.get("use_amp", True) and device.type == "cuda"
        test_m, test_hyps_g, test_refs = evaluate_greedy(
            eval_model, test_loader, device, use_amp_lofo
        )
        eval_model.cpu(); del eval_model; torch.cuda.empty_cache()

        from metrics import bootstrap_ci as _bci, cer as _cer_fn
        cer_arr = np.array([_cer_fn(h, r) for h, r in zip(test_hyps_g, test_refs)])
        bci = _bci(cer_arr, n_bootstrap=min(cfg["n_bootstrap"], 2000))

        print(f"\n  ── Test '{held_font}' (nunca visto) ──")
        print(f"  CER={test_m['CER']:.6f}  WER={test_m['WER']:.6f}  "
              f"LineAcc={test_m['LineAcc']:.6f}  "
              f"CER IC95%=[{bci['ci_low']:.6f}, {bci['ci_high']:.6f}]  "
              f"({elapsed:.1f} min)")

        lofo_result = {
            "held_font":          held_font,
            "n_test":             len(test_idx),
            "n_train":            len(train_idx),
            "test_CER":           test_m["CER"],
            "test_WER":           test_m["WER"],
            "test_LineAcc":       test_m["LineAcc"],
            "test_Char_F1":       test_m["Char_F1"],
            "test_CER_ci95_low":  bci["ci_low"],
            "test_CER_ci95_high": bci["ci_high"],
            "elapsed_min":        elapsed,
        }
        lofo_results.append(lofo_result)

        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump({"completed": len(lofo_results), "lofo_results": lofo_results}, f, indent=2)

    print(f"\n{'═'*66}\n  RESUMEN LOFO\n{'═'*66}")
    print(f"\n  {'Fuente excluida':<45} {'CER':>8} {'IC95-':>9} {'IC95+':>9} {'LineAcc':>9}")
    print("─" * 66)
    for r in lofo_results:
        lbl = (r["held_font"][:43] + "…") if len(r["held_font"]) > 44 else r["held_font"]
        print(f"  {lbl:<45} {r['test_CER']:>8.4f} "
              f"{r['test_CER_ci95_low']:>9.4f} {r['test_CER_ci95_high']:>9.4f} "
              f"{r['test_LineAcc']:>9.4f}")
    print("─" * 66)

    summary_lofo = {}
    for key in ["test_CER", "test_WER", "test_LineAcc", "test_Char_F1"]:
        vals   = np.array([r[key] for r in lofo_results])
        k_lofo = len(vals)
        t_crit = _stats.t.ppf(0.975, df=max(k_lofo - 1, 1))
        margin = t_crit * vals.std() / (k_lofo ** 0.5)
        label  = key.replace("test_", "")
        summary_lofo[label] = {
            "mean": float(vals.mean()), "std": float(vals.std()),
            "min":  float(vals.min()),  "max": float(vals.max()),
            "ci95_low": float(vals.mean() - margin),
            "ci95_high": float(vals.mean() + margin),
        }
        print(f"  {label:<10} media={vals.mean():.4f} ± {vals.std():.4f}  "
              f"IC95%=[{vals.mean()-margin:.4f}, {vals.mean()+margin:.4f}]")

    return {"lofo_results": lofo_results, "summary": summary_lofo}


# Entry point principal 

def train(cfg: dict) -> None:
    device = get_device()

    os.environ["OCR_DATA_DIR"]   = cfg["data_dir"]
    os.environ["OCR_VOCAB_PATH"] = cfg["vocab_path"]

    import importlib
    import dataset as ds_module
    importlib.reload(ds_module)
    from dataset import OCRDataset

    print(f"\nDispositivo: {device}  |  PyTorch {torch.__version__}")
    use_amp = cfg["use_amp"] and device.type == "cuda"
    mode    = cfg.get("mode", "train")
    print(f"AMP: {'ON' if use_amp else 'OFF'}  |  Modo: {mode.upper()}")

    # Dataset 
    full_ds = OCRDataset(data_dir=cfg["data_dir"],
                         img_height=cfg["img_height"], augment=True)

    print("Precalculando anchos de imagen (reutilizados en CV y LOFO)...")
    full_ds.precompute_widths()
    wc = full_ds._width_cache

    n_val   = max(1, int(len(full_ds) * cfg["val_split"]))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    val_ds_noaug = NoAugSubset(full_ds, list(val_ds.indices))

    loader_kw = dict(
        collate_fn=collate_fn,
        num_workers=cfg["num_workers"],
        persistent_workers=(cfg["num_workers"] > 0),
        prefetch_factor=cfg["prefetch_factor"] if cfg["num_workers"] > 0 else None,
        pin_memory=(device.type == "cuda"),
    )

    train_sampler = BucketBatchSampler(train_ds, batch_size=cfg["batch_size"],
                                       shuffle=True,  width_cache=wc)
    val_sampler   = BucketBatchSampler(val_ds_noaug, batch_size=cfg["batch_size"],
                                       shuffle=False, width_cache=wc)
    train_loader  = DataLoader(train_ds,      batch_sampler=train_sampler, **loader_kw)
    val_loader    = DataLoader(val_ds_noaug,  batch_sampler=val_sampler,   **loader_kw)
    print(f"Train: {n_train} | Val: {n_val} | Batches/época: {len(train_loader)}")

    # KenLM 
    lm_model, lm_alpha_val = None, 0.0
    if cfg.get("corpus_dir") and cfg.get("lm_path"):
        if not Path(cfg["lm_path"]).exists():
            build_lm(cfg["corpus_dir"], cfg["lm_path"])
        lm_model, lm_alpha_val = load_lm(cfg["lm_path"], cfg.get("lm_alpha_lm", 0.4))

    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_model.pt"

    #  ENTRENAMIENTO / FINE-TUNING PRINCIPAL
    if cfg.get("run_training", True):
        best_path = _run_main_training(
            cfg, full_ds, train_loader, val_loader,
            list(val_ds.indices), device, ckpt_dir,
            lm_model, lm_alpha_val, n_train, n_val,
        )
    else:
        print("\n[INFO] run_training=False — se omite el bucle de entrenamiento principal.")
        if not best_path.exists():
            raise FileNotFoundError(
                f"run_training=False pero no existe '{best_path}'.\n"
                "Asegúrate de que el checkpoint esté en checkpoint_dir."
            )
        print(f"  Usando checkpoint existente: {best_path}")

    #  VALIDACIÓN CRUZADA
    if cfg.get("run_cv", False):
        if cfg.get("cv_folds", 0) < 2:
            print("[WARN] run_cv=True pero cv_folds < 2 — CV omitida.")
        else:
            cv_cfg = {**cfg, "use_amp": use_amp}
            cv_result = run_cross_validation(
                cv_cfg, full_ds, ckpt_dir, device,
                pretrained_path=best_path,
            )
            cv_path = ckpt_dir / "cross_validation_results.json"
            with open(cv_path, "w", encoding="utf-8") as f:
                json.dump(cv_result, f, indent=2, ensure_ascii=False)
            print(f"  CV guardada en: {cv_path}")
    else:
        print("\n[INFO] run_cv=False — CV omitida.")

    #  LEAVE-ONE-FONT-OUT
    if cfg.get("run_lofo", False):
        lofo_cfg = {**cfg, "use_amp": use_amp, "val_seed": 42}
        lofo_result = run_lofo(
            lofo_cfg, full_ds, ckpt_dir, device,
            pretrained_path=best_path,
        )
        lofo_path = ckpt_dir / "lofo_results.json"
        with open(lofo_path, "w", encoding="utf-8") as f:
            json.dump(lofo_result, f, indent=2, ensure_ascii=False)
        print(f"  LOFO guardado en: {lofo_path}")
    else:
        print("\n[INFO] run_lofo=False — LOFO omitido.")


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrenamiento / Fine-tuning CRNN+CTC para OCR"
    )
    parser.add_argument("--data_dir",             default=CONFIG["data_dir"])
    parser.add_argument("--vocab_path",           default=CONFIG["vocab_path"])
    parser.add_argument("--checkpoint_dir",       default=CONFIG["checkpoint_dir"])
    parser.add_argument("--mode",                 default=CONFIG["mode"],
                        choices=["train", "finetune"])
    parser.add_argument("--finetune_checkpoint",  default=CONFIG["finetune_checkpoint"])
    parser.add_argument("--epochs",               type=int,   default=CONFIG["epochs"])
    parser.add_argument("--finetune_epochs",      type=int,   default=CONFIG["finetune_epochs"])
    parser.add_argument("--batch_size",           type=int,   default=CONFIG["batch_size"])
    parser.add_argument("--lr",                   type=float, default=CONFIG["lr"])
    parser.add_argument("--finetune_lr",          type=float, default=CONFIG["finetune_lr"])
    parser.add_argument("--freeze_cnn_epochs",    type=int,   default=CONFIG["freeze_cnn_epochs"])
    parser.add_argument("--hidden_size",          type=int,   default=CONFIG["hidden_size"])
    parser.add_argument("--beam_width",           type=int,   default=CONFIG["beam_width"])
    parser.add_argument("--n_bootstrap",          type=int,   default=CONFIG["n_bootstrap"])
    parser.add_argument("--cv_folds",             type=int,   default=CONFIG["cv_folds"])
    parser.add_argument("--cv_epochs",            type=int,   default=CONFIG["cv_epochs"])
    parser.add_argument("--lofo_epochs",          type=int,   default=CONFIG["lofo_epochs"])
    parser.add_argument("--run_training",         action="store_true", default=True)
    parser.add_argument("--no_training",          action="store_true")
    parser.add_argument("--run_cv",               action="store_true")
    parser.add_argument("--run_lofo",             action="store_true")
    parser.add_argument("--no_amp",               action="store_true")
    parser.add_argument("--no_resume",            action="store_true")
    parser.add_argument("--run_beam",             action="store_true")
    args = parser.parse_args()

    cfg = CONFIG.copy()
    cfg.update({
        "data_dir":            args.data_dir,
        "vocab_path":          args.vocab_path,
        "checkpoint_dir":      args.checkpoint_dir,
        "mode":                args.mode,
        "finetune_checkpoint": args.finetune_checkpoint,
        "epochs":              args.epochs,
        "finetune_epochs":     args.finetune_epochs,
        "batch_size":          args.batch_size,
        "lr":                  args.lr,
        "finetune_lr":         args.finetune_lr,
        "freeze_cnn_epochs":   args.freeze_cnn_epochs,
        "hidden_size":         args.hidden_size,
        "beam_width":          args.beam_width,
        "n_bootstrap":         args.n_bootstrap,
        "cv_folds":            args.cv_folds,
        "cv_epochs":           args.cv_epochs,
        "lofo_epochs":         args.lofo_epochs,
        "run_training":        not args.no_training,
        "run_cv":              args.run_cv,
        "run_lofo":            args.run_lofo,
        "skip_beam_final":     not args.run_beam,
    })
    if args.no_amp:    cfg["use_amp"] = False
    if args.no_resume: cfg["resume"]  = False

    train(cfg)