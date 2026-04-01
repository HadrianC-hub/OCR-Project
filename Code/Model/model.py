"""
model.py  —  CRNN+CTC ligero, optimizado para CPU (Ryzen 5700H)

Cambios respecto al modelo original:
  - CNN con la mitad de canales → 4× menos operaciones
  - Una sola capa BiLSTM → evita el overhead de capas apiladas en CPU
  - hidden_size=128 por defecto → suficiente para texto impreso español
  - Mismo stride=4 → mantiene la restricción CTC (W ≥ L×4)
  - ~400k parámetros vs ~2M del modelo completo
"""

import torch
import torch.nn as nn


class CRNN(nn.Module):
    """
    CNN ligera + BiLSTM + proyección lineal.

    Flujo dimensional (img_height=32, W=ancho de la línea):
        [B, 1, 32, W]
        → CNN bloques 1-2 (MaxPool 2×2 × 2)  → [B, 64,  8,  W/4]
        → CNN bloque 3 (MaxPool solo en alto) → [B, 128, 4,  W/4]
        → CNN bloque 4 (MaxPool solo en alto) → [B, 128, 1,  W/4]  (aprox.)
        → AdaptiveAvgPool(1, None)            → [B, 128, 1,  T]
        → squeeze + permute                   → [B, T,  128]
        → BiLSTM                              → [B, T,  256]
        → FC                                  → [T, B,  vocab_size]

    El stride horizontal acumulado es 4 → restricción CTC: W ≥ L × 4.
    """

    def __init__(
        self,
        vocab_size: int = 101,   # 99 símbolos + blank (índice 100)
        img_height: int = 32,
        hidden_size: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert img_height % 8 == 0, "img_height debe ser divisible entre 8"

        # ------------------------------------------------------------------ #
        # CNN ligera                                                           #
        # stride horizontal acumulado = 4                                     #
        # ------------------------------------------------------------------ #
        self.cnn = nn.Sequential(
            # Bloque 1: [B, 1, H, W] → [B, 32, H/2, W/2]
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloque 2: [B, 32, H/2, W/2] → [B, 64, H/4, W/4]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloque 3: [B, 64, H/4, W/4] → [B, 128, H/8, W/4]
            # MaxPool solo en alto → stride horizontal sigue siendo 4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Bloque 4: ajuste fino sin reducir más
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Aplasta el alto residual a 1 → [B, 128, 1, T]
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))

        # ------------------------------------------------------------------ #
        # BiLSTM — una sola capa, bidireccional                               #
        # ------------------------------------------------------------------ #
        self.rnn = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,  # Sin dropout con una sola capa
        )

        # Proyección a vocabulario
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

        # Dropout ligero antes de la proyección
        self.drop = nn.Dropout(p=dropout)

        self._init_weights()

    def _init_weights(self):
        """Inicialización ortogonal para la LSTM — converge más rápido."""
        for name, p in self.rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
                # Activar forget gate al inicio → evita vanishing gradient
                n = p.size(0)
                p.data[n // 4 : n // 2].fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parámetros
        ----------
        x : Tensor  [B, 1, H, W]

        Retorna
        -------
        log_probs : Tensor  [T, B, vocab_size]
        """
        # CNN: [B, 1, H, W] → [B, 128, ≈1, T]
        feat = self.cnn(x)
        feat = self.adaptive_pool(feat)     # [B, 128, 1, T]
        B, C, _, T = feat.shape
        feat = feat.squeeze(2).permute(0, 2, 1)  # [B, T, 128]

        # BiLSTM: [B, T, 128] → [B, T, 256]
        out, _ = self.rnn(feat)

        # Proyección: [B, T, vocab_size] → [T, B, vocab_size]
        out = self.fc(self.drop(out))
        out = out.permute(1, 0, 2)

        return torch.log_softmax(out, dim=2)