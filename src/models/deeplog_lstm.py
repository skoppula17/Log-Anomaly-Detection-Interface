from __future__ import annotations
import torch
from torch import nn

class DeepLogLSTM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 64, hidden_dim: int = 128, num_layers: int = 1, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T]
        emb = self.embedding(x)           # [B, T, D]
        out, _ = self.lstm(emb)           # [B, T, H]
        last = out[:, -1, :]              # [B, H]
        logits = self.fc(last)            # [B, V]
        return logits
