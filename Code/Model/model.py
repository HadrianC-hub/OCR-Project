"""
model.py  —  CRNN+CTC para reconocimiento de texto manuscrito/impreso.

Arquitectura: CNN (5 bloques) → AdaptiveAvgPool → BiLSTM × num_layers → FC → log_softmax
Stride horizontal acumulado: 4 (CNN_STRIDE=4 en dataset.py)
Restricción CTC: ancho_imagen ≥ longitud_etiqueta × 4
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
        # La altura se reduce ×16 por los 4 MaxPool: mínimo 16 px.
        assert img_height % 16 == 0, "img_height debe ser divisible entre 16"

        self.cnn = nn.Sequential(
            # Bloque 1: [B, 1, H, W] → [B, 64, H/2, W/2]
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloque 2: MaxPool 2×2 — reduce ancho también → [B, 128, H/4, W/4]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloques 3-4: MaxPool solo en altura → el ancho queda fijo en W/4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Bloque 5: sin reducción espacial
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Colapsa la dimensión de altura → [B, 256, 1, T]  con T = W/4
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
                # Sesgo de puerta de olvido a 1 para estabilizar el entrenamiento inicial.
                n = p.size(0)
                p.data[n // 4 : n // 2].fill_(1.0)
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x)                          # [B, 256, H/16, T]
        feat = self.adaptive_pool(feat)             # [B, 256, 1, T]
        B, C, _, T = feat.shape
        feat = feat.squeeze(2).permute(0, 2, 1)    # [B, T, 256]
        out, _ = self.rnn(feat)                     # [B, T, 2·hidden]
        out = self.fc(out)                          # [B, T, vocab]
        out = out.permute(1, 0, 2)                  # [T, B, vocab]  — formato requerido por CTCLoss
        return torch.log_softmax(out, dim=2)