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


# ======================================================================
#  CONFIGURACIÓN
# ======================================================================
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

    # Entrenamiento
    "epochs":       35,
    "batch_size":   32,
    "lr":           3e-4,
    "weight_decay": 1e-4,
    "val_split":    0.1,
    "grad_clip":    5.0,

    # GPU
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
    "cv_folds":    0,
    "cv_epochs":   10,
    "lofo":        False,
    "lofo_epochs": 10,

    # Si True, salta el beam search en la evaluación final del entrenamiento
    # principal. El beam empeora en este modelo (CER 0.0102 → 0.0172)
    # y tarda ~13 min extra.
    "skip_beam_final": True,

    # Checkpoints
    "save_every": 5,
    "resume":     True,
}
# ======================================================================


# --- KenLM ---

def build_lm(corpus_dir: str, lm_path: str, order: int = 5) -> bool:
    import subprocess
    import tempfile

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
        print("[LM] kenlm no instalado — beam sin LM (pip install kenlm para activar)")
        return None, 0.0


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
    scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch = ckpt["epoch"] + 1
    best_cer    = ckpt["best_cer"]
    print(f"Reanudando desde época {start_epoch}  |  Mejor CER previo: {best_cer:.4f}")
    return start_epoch, best_cer


# --- Evaluación ---

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


# --- Entrenamiento de un modelo (usado por CV y LOFO) ---

def _train_model(
    train_loader, val_loader, cfg: dict, device: torch.device,
    ckpt_path: Path, desc: str = "", skip_beam: bool = False,
    lm=None, lm_alpha: float = 0.4,
    pretrained_path: str | Path | None = None,
) -> tuple:
    use_amp  = cfg.get("use_amp", True) and device.type == "cuda"
    n_epochs = cfg["epochs"]
    prefix   = f"[{desc}] " if desc else ""

    model = CRNN(
        vocab_size  = cfg["vocab_size"],
        img_height  = cfg["img_height"],
        hidden_size = cfg["hidden_size"],
        num_layers  = cfg.get("num_layers", 2),
        dropout     = cfg["dropout"],
    ).to(device)

    # Warm start: carga pesos del modelo principal para acelerar convergencia en CV/LOFO.
    if pretrained_path is not None and Path(pretrained_path).exists():
        try:
            ckpt_pre = torch.load(pretrained_path, map_location=device)
            state    = {k.replace("_orig_mod.", ""): v
                        for k, v in ckpt_pre["model_state"].items()}
            model.load_state_dict(state)
            print(f"  {prefix}→ Warm start desde '{Path(pretrained_path).name}'")
        except Exception as e:
            print(f"  {prefix}→ Warm start falló ({e}), inicio desde cero")

    ctc_loss  = nn.CTCLoss(blank=BLANK_IDX, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["lr"], weight_decay=cfg["weight_decay"])
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
        train_sec = time.time() - t0
        elapsed   = time.time() - t_global
        eta_min   = elapsed / epoch * (n_epochs - epoch) / 60

        metrics_e, hyps_e, refs_e = evaluate_greedy(model, val_loader, device, use_amp)
        print(
            f"{prefix}Epoch {epoch:02d}/{n_epochs}  "
            f"loss={avg_loss:.4f}  CER={metrics_e['CER']:.4f}  "
            f"LineAcc={metrics_e['LineAcc']:.4f}  "
            f"({train_sec:.0f}s | ETA: {eta_min:.1f} min)"
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
            print(f"  Mejor CER: {best_cer:.6f}  (guardado)")

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
            use_amp=use_amp,
            lm=lm, lm_alpha=lm_alpha,
        )

    model.cpu()
    del model, optimizer, scheduler
    if scaler is not None:
        del scaler
    torch.cuda.empty_cache()

    return best_metrics, best_hyps_g, best_hyps_b, best_refs


# --- Validación cruzada k-fold estratificada por fuente ---

def run_cross_validation(cfg: dict, full_ds: OCRDataset,
                         ckpt_dir: Path, device: torch.device,
                         pretrained_path: str | Path | None = None) -> dict:
    import numpy as np
    k    = cfg["cv_folds"]
    seed = cfg.get("val_seed", 42)

    print(f"\n{'═'*66}")
    print(f"  VALIDACIÓN CRUZADA {k}-FOLD ESTRATIFICADA POR FUENTE")
    if pretrained_path:
        print(f"  Warm start: {pretrained_path}")
    print(f"{'═'*66}")

    font_to_idx = defaultdict(list)
    for idx, (img_path, _) in enumerate(full_ds.samples):
        font_to_idx[get_font_label(img_path)].append(idx)

    fonts = sorted(font_to_idx.keys())
    print(f"\n  Fuentes ({len(fonts)}): {', '.join(fonts)}")

    cv_epochs = cfg.get("cv_epochs", cfg["epochs"])
    print(f"  Épocas por fold: {cv_epochs}")

    rng = np.random.default_rng(seed=seed)
    fold_indices = [[] for _ in range(k)]
    for font, idxs in font_to_idx.items():
        arr = list(idxs); rng.shuffle(arr)
        for i, idx in enumerate(arr):
            fold_indices[i % k].append(idx)

    for i, fold in enumerate(fold_indices):
        print(f"  Fold {i+1}: {len(fold):,} muestras")

    loader_kw = dict(
        collate_fn=collate_fn,
        num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    fold_results = []
    cv_ckpt_dir  = ckpt_dir / "cv_folds"
    cv_ckpt_dir.mkdir(exist_ok=True)
    partial_path = ckpt_dir / "cv_partial_results.json"

    for fold_idx in range(k):
        print(f"\n{'─'*66}")
        print(f"  FOLD {fold_idx+1} / {k}")
        print(f"{'─'*66}")

        val_idx   = fold_indices[fold_idx]
        train_idx = [i for j, fold in enumerate(fold_indices)
                     if j != fold_idx for i in fold]

        train_subset = Subset(full_ds, train_idx)
        val_subset   = NoAugSubset(full_ds, val_idx)

        wc = full_ds._width_cache if full_ds._width_cache else None
        train_sampler = BucketBatchSampler(train_subset, batch_size=cfg["batch_size"],
                                           shuffle=True, width_cache=wc)
        val_sampler   = BucketBatchSampler(val_subset,   batch_size=cfg["batch_size"],
                                           shuffle=False, width_cache=wc)

        train_loader = DataLoader(train_subset, batch_sampler=train_sampler, **loader_kw)
        val_loader   = DataLoader(val_subset,   batch_sampler=val_sampler,   **loader_kw)

        print(f"  Train: {len(train_idx):,}  |  Val: {len(val_idx):,}")

        fold_ckpt = cv_ckpt_dir / f"fold_{fold_idx+1}_best.pt"
        t0 = time.time()
        fold_cfg  = {**cfg, "epochs": cv_epochs}
        best_m, hyps_g, hyps_b, refs = _train_model(
            train_loader, val_loader, fold_cfg, device,
            fold_ckpt, desc=f"Fold {fold_idx+1}/{k}", skip_beam=True,
            pretrained_path=pretrained_path,
        )
        elapsed = (time.time() - t0) / 60

        font_lbls = [get_font_label(full_ds.samples[i][0]) for i in val_idx]
        stat_r = compute_statistical_report(
            hyps_g, hyps_g, refs,
            font_labels=font_lbls,
            n_bootstrap=min(cfg["n_bootstrap"], 1000),
        )
        fold_result = {
            "fold":       fold_idx + 1,
            "n_val":      len(val_idx),
            "CER":        best_m["CER"],
            "WER":        best_m["WER"],
            "LineAcc":    best_m["LineAcc"],
            "Char_F1":    best_m["Char_F1"],
            "BLEU4_char": best_m["BLEU4_char"],
            "elapsed_min": elapsed,
        }
        fold_results.append(fold_result)
        print(f"  CER={best_m['CER']:.6f}  LineAcc={best_m['LineAcc']:.6f}  ({elapsed:.1f} min)")

        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump({"completed_folds": fold_idx + 1, "fold_results": fold_results}, f, indent=2)

    import numpy as np
    from scipy import stats as _stats

    print(f"\n{'═'*66}")
    print(f"  RESUMEN {k}-FOLD")
    print(f"{'═'*66}")
    print(f"\n  {'Métrica':<20} {'Media':>9} {'Std':>9} {'Min':>9} {'Max':>9} {'IC95-':>9} {'IC95+':>9}")
    print("─" * 66)

    summary = {}
    for key in ["CER", "WER", "LineAcc", "Char_F1", "BLEU4_char"]:
        vals    = np.array([r[key] for r in fold_results])
        mean, std = float(vals.mean()), float(vals.std())
        vmin, vmax = float(vals.min()), float(vals.max())
        t_crit  = _stats.t.ppf(0.975, df=k - 1)
        margin  = t_crit * std / (k ** 0.5)
        cv_pct  = (std / mean * 100) if mean > 0 else float("inf")
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


# --- Leave-One-Font-Out ---

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
        print(f"  Warm start: {pretrained_path}")
    print(f"{'═'*66}")

    font_to_idx = defaultdict(list)
    for idx, (img_path, _) in enumerate(full_ds.samples):
        font_to_idx[get_font_label(img_path)].append(idx)

    all_fonts = sorted(font_to_idx.keys())
    print(f"\n  Fuentes ({len(all_fonts)}): {', '.join(all_fonts)}")

    lofo_epochs = cfg.get("lofo_epochs", cfg["epochs"])
    print(f"  Épocas por iteración: {lofo_epochs}")

    loader_kw = dict(
        collate_fn=collate_fn,
        num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    lofo_results = []
    lofo_ckpt_dir = ckpt_dir / "lofo"
    lofo_ckpt_dir.mkdir(exist_ok=True)
    partial_path = ckpt_dir / "lofo_partial_results.json"

    for held_font in all_fonts:
        print(f"\n{'─'*66}")
        print(f"  Fuente retenida: {held_font}")
        print(f"{'─'*66}")

        test_idx  = font_to_idx[held_font]
        other_idx = [i for f, idxs in font_to_idx.items()
                     if f != held_font for i in idxs]

        arr = np.array(other_idx); rng.shuffle(arr)
        n_val_int   = max(1, int(len(arr) * cfg["val_split"]))
        val_int_idx = arr[:n_val_int].tolist()
        train_idx   = arr[n_val_int:].tolist()

        print(f"  Train: {len(train_idx):,}  |  Val-internal: {len(val_int_idx):,}"
              f"  |  Test ({held_font}): {len(test_idx):,}")

        train_subset = Subset(full_ds, train_idx)
        val_subset   = NoAugSubset(full_ds, val_int_idx)
        test_subset  = NoAugSubset(full_ds, test_idx)

        wc = full_ds._width_cache if full_ds._width_cache else None
        train_sampler = BucketBatchSampler(train_subset, batch_size=cfg["batch_size"],
                                           shuffle=True,  width_cache=wc)
        val_sampler   = BucketBatchSampler(val_subset,   batch_size=cfg["batch_size"],
                                           shuffle=False, width_cache=wc)
        test_sampler  = BucketBatchSampler(test_subset,  batch_size=cfg["batch_size"],
                                           shuffle=False, width_cache=wc)

        train_loader = DataLoader(train_subset, batch_sampler=train_sampler, **loader_kw)
        val_loader   = DataLoader(val_subset,   batch_sampler=val_sampler,   **loader_kw)
        test_loader  = DataLoader(test_subset,  batch_sampler=test_sampler,  **loader_kw)

        lofo_ckpt = lofo_ckpt_dir / f"lofo_{held_font[:30]}_best.pt"
        t0 = time.time()
        lofo_run_cfg = {**cfg, "epochs": lofo_epochs}
        _, _, _, _ = _train_model(
            train_loader, val_loader, lofo_run_cfg, device,
            lofo_ckpt, desc=f"LOFO:{held_font[:20]}", skip_beam=True,
            pretrained_path=pretrained_path,
        )
        elapsed = (time.time() - t0) / 60

        ckpt  = torch.load(lofo_ckpt, map_location=device)
        eval_model = CRNN(
            vocab_size=cfg["vocab_size"], img_height=cfg["img_height"],
            hidden_size=cfg["hidden_size"], num_layers=cfg.get("num_layers", 2),
        ).to(device)
        state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state"].items()}
        eval_model.load_state_dict(state)

        use_amp_lofo  = cfg.get("use_amp", True) and device.type == "cuda"
        test_m, test_hyps_g, test_refs = evaluate_greedy(
            eval_model, test_loader, device, use_amp_lofo
        )

        eval_model.cpu()
        del eval_model
        torch.cuda.empty_cache()

        from metrics import bootstrap_ci as _bci, cer as _cer_fn
        cer_arr = np.array([_cer_fn(h, r) for h, r in zip(test_hyps_g, test_refs)])
        bci = _bci(cer_arr, n_bootstrap=min(cfg["n_bootstrap"], 2000))

        print(f"\n  Test sobre '{held_font}' (nunca visto en train):")
        print(f"  CER={test_m['CER']:.6f}  WER={test_m['WER']:.6f}  "
              f"LineAcc={test_m['LineAcc']:.6f}  "
              f"CER IC95%=[{bci['ci_low']:.6f}, {bci['ci_high']:.6f}]  "
              f"({elapsed:.1f} min)")

        lofo_result = {
            "held_font":   held_font,
            "n_test":      len(test_idx),
            "n_train":     len(train_idx),
            "test_CER":    test_m["CER"],
            "test_WER":    test_m["WER"],
            "test_LineAcc":test_m["LineAcc"],
            "test_Char_F1":test_m["Char_F1"],
            "test_CER_ci95_low":  bci["ci_low"],
            "test_CER_ci95_high": bci["ci_high"],
            "elapsed_min": elapsed,
        }
        lofo_results.append(lofo_result)

        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump({"completed": len(lofo_results), "lofo_results": lofo_results}, f, indent=2)

    print(f"\n{'═'*66}")
    print(f"  RESUMEN LOFO")
    print(f"{'═'*66}")
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
        t_crit = _stats.t.ppf(0.975, df=k_lofo - 1)
        margin = t_crit * vals.std() / (k_lofo ** 0.5)
        label  = key.replace("test_", "")
        summary_lofo[label] = {
            "mean": float(vals.mean()), "std": float(vals.std()),
            "min":  float(vals.min()),  "max": float(vals.max()),
            "ci95_low": float(vals.mean() - margin),
            "ci95_high": float(vals.mean() + margin),
        }
        print(f"  {label:<10} media={vals.mean():.4f} ± {vals.std():.4f}  "
              f"IC95%=[{vals.mean()-margin:.4f}, {vals.mean()+margin:.4f}]  "
              f"[{vals.min():.4f}–{vals.max():.4f}]")

    final_json = ckpt_dir / "final_metrics.json"
    in_sample_cer = None
    if final_json.exists():
        try:
            with open(final_json) as _f:
                _d = json.load(_f)
            in_sample_cer = _d.get("greedy_metrics", {}).get("CER")
        except Exception:
            pass

    if in_sample_cer is not None:
        lofo_cer_mean = summary_lofo["CER"]["mean"]
        degradation   = lofo_cer_mean - in_sample_cer
        print(f"\n  CER in-sample (best_model.pt greedy): {in_sample_cer:.4f}")
        print(f"  CER LOFO (fuentes nuevas):             {lofo_cer_mean:.4f}")
        print(f"  Degradación por fuente nueva:          {degradation:+.4f}  "
              f"({'pequeña <0.05' if abs(degradation) < 0.05 else 'notable ≥0.05'})")

    return {"lofo_results": lofo_results, "summary": summary_lofo}


# --- Bucle de entrenamiento principal ---

def train(cfg: dict) -> None:
    device = get_device()

    os.environ["OCR_DATA_DIR"]   = cfg["data_dir"]
    os.environ["OCR_VOCAB_PATH"] = cfg["vocab_path"]

    import importlib
    import dataset as ds_module
    importlib.reload(ds_module)
    from dataset import OCRDataset, collate_fn, NoAugSubset, BucketBatchSampler

    print(f"\nDispositivo: {device}  |  PyTorch {torch.__version__}")
    use_amp = cfg["use_amp"] and device.type == "cuda"
    print(f"AMP: {'ON' if use_amp else 'OFF'}")

    full_ds = OCRDataset(data_dir=cfg["data_dir"],
                         img_height=cfg["img_height"], augment=True)

    print("Precalculando anchos de imagen (se reutilizarán en CV y LOFO)...")
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

    print("Construyendo BucketBatchSampler para train...")
    train_sampler = BucketBatchSampler(train_ds, batch_size=cfg["batch_size"],
                                       shuffle=True, width_cache=wc)
    print("Construyendo BucketBatchSampler para val...")
    val_sampler   = BucketBatchSampler(val_ds_noaug, batch_size=cfg["batch_size"],
                                       shuffle=False, width_cache=wc)

    train_loader = DataLoader(train_ds,      batch_sampler=train_sampler, **loader_kw)
    val_loader   = DataLoader(val_ds_noaug,  batch_sampler=val_sampler,   **loader_kw)
    print(f"Train: {n_train} | Val: {n_val} | Batches/época: {len(train_loader)}")

    lm_model, lm_alpha = None, 0.0
    if cfg.get("corpus_dir") and cfg.get("lm_path"):
        lm_path = cfg["lm_path"]
        if not Path(lm_path).exists():
            build_lm(cfg["corpus_dir"], lm_path)
        lm_model, lm_alpha = load_lm(lm_path, cfg.get("lm_alpha_lm", 0.4))

    model = CRNN(
        vocab_size  = cfg["vocab_size"],
        img_height  = cfg["img_height"],
        hidden_size = cfg["hidden_size"],
        num_layers  = cfg.get("num_layers", 2),
        dropout     = cfg["dropout"],
    ).to(device)

    # torch.compile no se usa: las imágenes tienen anchos variables por batch.
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros: {n_params:,}")

    ctc_loss  = nn.CTCLoss(blank=BLANK_IDX, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg["lr"],
        steps_per_epoch=len(train_loader),
        epochs=cfg["epochs"],
        pct_start=0.1, anneal_strategy="cos",
    )
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    ckpt_dir  = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path   = ckpt_dir / "best_model.pt"
    start_epoch = 1
    best_cer    = float("inf")

    if cfg["resume"] and best_path.exists():
        start_epoch, best_cer = load_checkpoint(best_path, model, optimizer, scheduler, device)

    history_path = Path(cfg["history_path"])
    history = {"train_loss": [], "val_metrics": []}
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

    print(f"\nIniciando desde época {start_epoch}/{cfg['epochs']}")
    skip_beam_final = cfg.get("skip_beam_final", True)
    print(f"Beam search final: {'DESACTIVADO' if skip_beam_final else 'ACTIVADO'}\n")
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
                with torch.amp.autocast('cuda'):
                    log_probs = model(images)
                    loss = ctc_loss(log_probs, labels, input_lengths, target_lengths)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scale_before = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                if scaler.get_scale() == scale_before:
                    scheduler.step()
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
                    f"loss={loss.item():.4f}  lr={scheduler.get_last_lr()[0]:.2e}  "
                    f"ETA época: {eta_min:.1f} min"
                )

        avg_loss  = epoch_loss / len(train_loader)
        train_sec = time.time() - t0
        eval_t0   = time.time()

        metrics, hyps_ep, refs_ep = evaluate_greedy(model, val_loader, device, use_amp)
        eval_sec  = time.time() - eval_t0
        eta_total = (cfg["epochs"] - epoch) * (train_sec + eval_sec) / 60

        print(
            f"\n[Epoch {epoch:02d}/{cfg['epochs']}] [greedy]  "
            f"loss={avg_loss:.4f}  CER={metrics['CER']:.4f}  "
            f"WER={metrics['WER']:.4f}  1-NED={metrics['1-NED']:.4f}  "
            f"LineAcc={metrics['LineAcc']:.4f}  CharF1={metrics['Char_F1']:.4f}  "
            f"BLEU={metrics['BLEU4_char']:.4f}  "
            f"(train {train_sec:.0f}s | eval {eval_sec:.0f}s | ~{eta_total:.0f} min rest.)"
        )

        history["train_loss"].append(avg_loss)
        history["val_metrics"].append({
            "epoch": epoch, "eval_mode": "greedy",
            "CER": metrics["CER"], "WER": metrics["WER"],
            "1-NED": metrics["1-NED"], "LineAcc": metrics["LineAcc"],
            "Char_F1": metrics["Char_F1"], "BLEU4": metrics["BLEU4_char"],
        })
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        if metrics["CER"] < best_cer:
            best_cer = metrics["CER"]
            save_checkpoint(best_path, epoch, model, optimizer, scheduler,
                            best_cer, metrics, cfg)
            print(f"  Mejor modelo guardado  CER={best_cer:.4f}  [greedy]")

        if epoch % cfg["save_every"] == 0:
            periodic = ckpt_dir / f"ckpt_epoch{epoch:03d}.pt"
            save_checkpoint(periodic, epoch, model, optimizer, scheduler,
                            best_cer, metrics, cfg)
            print(f"  Checkpoint guardado: {periodic.name}")

    total_min = (time.time() - total_t0) / 60
    print(f"\nEntrenamiento completado en {total_min:.1f} min  |  Mejor CER: {best_cer:.4f}\n")

    print("Cargando mejor modelo para evaluación final...")
    ckpt  = torch.load(best_path, map_location=device)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state"].items()}
    model.load_state_dict(state)
    best_epoch = ckpt["epoch"]

    greedy_metrics, hyps_g, refs = evaluate_greedy(model, val_loader, device, use_amp)
    print_metrics(greedy_metrics, title=f"EVALUACIÓN FINAL [greedy] — época {best_epoch}")

    if skip_beam_final:
        print("\n[INFO] skip_beam_final=True — se omite beam search en val final.")
        hyps_b = hyps_g
        beam_metrics = greedy_metrics
    else:
        print(f"Ejecutando beam search final (w={cfg['beam_width']}, "
              f"blank_bonus={cfg['beam_bonus']}, alpha={cfg['beam_alpha']})...")
        beam_metrics, hyps_b, _ = evaluate_beam(
            model, val_loader, device,
            beam_width=cfg["beam_width"],
            beam_bonus=cfg["beam_bonus"],
            beam_alpha=cfg["beam_alpha"],
            use_amp=use_amp,
            lm=lm_model, lm_alpha=lm_alpha,
        )
        print_metrics(beam_metrics, title=f"EVALUACIÓN FINAL [beam={cfg['beam_width']}] — época {best_epoch}")

    font_lbls = [get_font_label(full_ds.samples[i][0]) for i in val_ds.indices]

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
            "best_epoch": best_epoch,
            "best_cer_greedy": best_cer,
            "total_time_min": total_min,
            "n_train": n_train,
            "n_val": n_val,
            "skip_beam_final": skip_beam_final,
        },
        "greedy_metrics": _ser(greedy_metrics),
        "beam_metrics":   _ser(beam_metrics),
        "statistical":    {
            k: v for k, v in stat_report.items() if k != "per_sample"
        },
        "history": history,
    }

    final_path = ckpt_dir / "final_metrics.json"
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    print(f"  Métricas finales guardadas en: {final_path}")

    model.cpu()
    del model, optimizer, scheduler
    if scaler is not None:
        del scaler
    torch.cuda.empty_cache()

    if cfg.get("cv_folds", 0) > 1:
        cv_cfg = {**cfg, "use_amp": use_amp}
        cv_result = run_cross_validation(
            cv_cfg, full_ds, ckpt_dir, device,
        )
        cv_path = ckpt_dir / "cross_validation_results.json"
        with open(cv_path, "w", encoding="utf-8") as f:
            json.dump(cv_result, f, indent=2, ensure_ascii=False)
        print(f"  CV guardada en: {cv_path}")

    if cfg.get("lofo", False):
        lofo_cfg = {**cfg, "use_amp": use_amp, "val_seed": 42}
        lofo_result = run_lofo(
            lofo_cfg, full_ds, ckpt_dir, device,
        )
        lofo_path = ckpt_dir / "lofo_results.json"
        with open(lofo_path, "w", encoding="utf-8") as f:
            json.dump(lofo_result, f, indent=2, ensure_ascii=False)
        print(f"  LOFO guardado en: {lofo_path}")


# --- Entry point ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",        default=CONFIG["data_dir"])
    parser.add_argument("--vocab_path",      default=CONFIG["vocab_path"])
    parser.add_argument("--epochs",          type=int,   default=CONFIG["epochs"])
    parser.add_argument("--batch_size",      type=int,   default=CONFIG["batch_size"])
    parser.add_argument("--lr",              type=float, default=CONFIG["lr"])
    parser.add_argument("--hidden_size",     type=int,   default=CONFIG["hidden_size"])
    parser.add_argument("--checkpoint_dir",  default=CONFIG["checkpoint_dir"])
    parser.add_argument("--beam_width",      type=int,   default=CONFIG["beam_width"])
    parser.add_argument("--beam_bonus",      type=float, default=CONFIG["beam_bonus"])
    parser.add_argument("--beam_alpha",      type=float, default=CONFIG["beam_alpha"])
    parser.add_argument("--n_bootstrap",     type=int,   default=CONFIG["n_bootstrap"])
    parser.add_argument("--cv_folds",        type=int,   default=CONFIG["cv_folds"])
    parser.add_argument("--cv_epochs",       type=int,   default=CONFIG["cv_epochs"])
    parser.add_argument("--lofo",            action="store_true")
    parser.add_argument("--lofo_epochs",     type=int,   default=CONFIG["lofo_epochs"])
    parser.add_argument("--no_amp",          action="store_true")
    parser.add_argument("--no_resume",       action="store_true")
    parser.add_argument("--run_beam",        action="store_true",
                        help="Activar beam search en evaluación final")
    args = parser.parse_args()

    cfg = CONFIG.copy()
    cfg.update({
        "data_dir":    args.data_dir,
        "vocab_path":  args.vocab_path,
        "epochs":      args.epochs,
        "batch_size":  args.batch_size,
        "lr":          args.lr,
        "hidden_size": args.hidden_size,
        "checkpoint_dir": args.checkpoint_dir,
        "beam_width":  args.beam_width,
        "beam_bonus":  args.beam_bonus,
        "beam_alpha":  args.beam_alpha,
        "n_bootstrap": args.n_bootstrap,
        "cv_folds":    args.cv_folds,
        "cv_epochs":   args.cv_epochs,
        "lofo":        args.lofo,
        "lofo_epochs": args.lofo_epochs,
        "skip_beam_final": not args.run_beam,
    })
    if args.no_amp:    cfg["use_amp"] = False
    if args.no_resume: cfg["resume"]  = False

    train(cfg)