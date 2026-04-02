"""
model.py  —  CRNN+CTC calidad mejorada, optimizado para GPU

Cambios respecto al modelo ligero:
  - CNN: 64/128/256/256/256 canales + BN en todos los bloques
  - Bloque 5 extra sin reducción espacial → más capacidad representativa
  - BiLSTM: 2 capas, hidden_size=256 → ~4M parámetros vs 400k anterior
  - Dropout 0.2 entre capas LSTM
  - Proyección con capa intermedia (FC1 → ReLU → FC2)

Flujo dimensional (img_height=32, W=ancho de la línea):
    [B, 1,   32, W]
    → Bloque 1 (MaxPool 2×2)        → [B,  64, 16, W/2]
    → Bloque 2 (MaxPool 2×2)        → [B, 128,  8, W/4]
    → Bloque 3 (MaxPool alto 2×1)   → [B, 256,  4, W/4]
    → Bloque 4 (MaxPool alto 2×1)   → [B, 256,  2, W/4]
    → Bloque 5 (sin pool)           → [B, 256,  2, W/4]
    → AdaptiveAvgPool(1, None)      → [B, 256,  1,   T]
    → squeeze + permute             → [B, T, 256]
    → BiLSTM ×2                     → [B, T, 512]
    → FC1 → ReLU → FC2              → [T, B, vocab_size]

Stride horizontal acumulado = 4 → restricción CTC: W ≥ L×4
"""

import torch
import torch.nn as nn


class CRNN(nn.Module):

    def __init__(
        self,
        vocab_size:  int   = 101,
        img_height:  int   = 32,
        hidden_size: int   = 256,
        num_layers:  int   = 2,
        dropout:     float = 0.2,
    ):
        super().__init__()
        assert img_height % 8 == 0, "img_height debe ser divisible entre 8"

        self.cnn = nn.Sequential(
            # Bloque 1: → [B, 64, H/2, W/2]
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloque 2: → [B, 128, H/4, W/4]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloque 3: MaxPool solo en alto → [B, 256, H/8, W/4]
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Bloque 4: MaxPool solo en alto → [B, 256, H/16, W/4]
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Bloque 5: sin reducción — más capacidad representativa
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))

        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        mid = hidden_size
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size * 2, mid),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(mid, vocab_size),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
                n = p.size(0)
                p.data[n // 4 : n // 2].fill_(1.0)
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, 1, H, W]
        → log_probs : [T, B, vocab_size]
        """
        feat = self.cnn(x)
        feat = self.adaptive_pool(feat)          # [B, 256, 1, T]
        B, C, _, T = feat.shape
        feat = feat.squeeze(2).permute(0, 2, 1)  # [B, T, 256]
        out, _ = self.rnn(feat)                  # [B, T, 512]
        out = self.fc(out)                       # [B, T, vocab]
        out = out.permute(1, 0, 2)               # [T, B, vocab]
        return torch.log_softmax(out, dim=2)