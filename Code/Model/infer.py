import argparse
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

from model   import CRNN
from dataset import decode_ctc_beam, autocrop_whitespace, IMG_HEIGHT, CNN_STRIDE


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: str, device: torch.device = None):
    if device is None:
        device = get_device()
    ckpt  = torch.load(checkpoint_path, map_location=device)
    cfg   = ckpt.get("config", {})
    model = CRNN(
        vocab_size=cfg.get("vocab_size",   101),
        img_height=cfg.get("img_height",   IMG_HEIGHT),
        hidden_size=cfg.get("hidden_size", 256),
        num_layers=cfg.get("num_layers",   2),
    )
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state"].items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"Modelo cargado desde '{checkpoint_path}'  |  dispositivo: {device}")
    return model, cfg, device


def transcribe_image(
    model,
    img_path: str,
    device: torch.device,
    img_height: int  = IMG_HEIGHT,
    beam_width: int  = 10,
    blank_bonus: float = 2.0,
    length_norm_alpha: float = 0.65,
    lm=None,
    lm_alpha: float = 0.4,
) -> str:
    img = Image.open(img_path).convert("L")
    img = autocrop_whitespace(img, threshold=200, padding=2)
    w, h = img.size
    new_w = max(4, int(round(w * img_height / h)))
    img = img.resize((new_w, img_height), Image.BICUBIC)

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    x = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        log_probs = model(x)   # [T, 1, vocab]

    valid_t = x.shape[3] // CNN_STRIDE
    lp_np = log_probs[:valid_t].squeeze(1).cpu().float().numpy()   # [T, vocab]
    seq   = [lp_np[t].tolist() for t in range(len(lp_np))]
    return decode_ctc_beam(
        seq,
        beam_width=beam_width,
        blank_bonus=blank_bonus,
        length_norm_alpha=length_norm_alpha,
        lm=lm,
        lm_alpha=lm_alpha,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint",          help="Ruta al checkpoint .pt")
    parser.add_argument("input",               help="Imagen .png/.jpg o carpeta")
    parser.add_argument("--beam_width",        type=int,   default=10)
    parser.add_argument("--blank_bonus",       type=float, default=2.0)
    parser.add_argument("--length_norm_alpha", type=float, default=0.65)
    parser.add_argument("--lm_path",           type=str,   default=None,
                        help="Ruta al modelo KenLM .arpa (opcional)")
    parser.add_argument("--lm_alpha",          type=float, default=0.4)
    args = parser.parse_args()

    lm_model, lm_alpha = None, 0.0
    if args.lm_path:
        try:
            import kenlm
            from pathlib import Path as _Path
            if _Path(args.lm_path).exists():
                lm_model = kenlm.Model(args.lm_path)
                lm_alpha = args.lm_alpha
                print(f"[LM] Modelo cargado: {args.lm_path}  |  α={lm_alpha}")
            else:
                print(f"[LM] Archivo no encontrado: {args.lm_path} — beam sin LM")
        except ImportError:
            print("[LM] kenlm no instalado — beam sin LM")

    model, cfg, device = load_model(args.checkpoint)
    h = cfg.get("img_height", IMG_HEIGHT)

    input_path = Path(args.input)
    paths = [input_path] if input_path.is_file() else sorted(
        list(input_path.glob("*.png")) + list(input_path.glob("*.jpg"))
    )

    print(f"\n{'Archivo':<40} Transcripción")
    print("─" * 80)
    for p in paths:
        text = transcribe_image(
            model, str(p), device,
            img_height=h,
            beam_width=args.beam_width,
            blank_bonus=args.blank_bonus,
            length_norm_alpha=args.length_norm_alpha,
            lm=lm_model,
            lm_alpha=lm_alpha,
        )
        print(f"{p.name:<40} {text}")