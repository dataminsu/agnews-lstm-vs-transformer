

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


def count_parameters(model: nn.Module) -> int:
    """Trainable parameter count -- required in the report for both models."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageEmbeddingClassifier(nn.Module):
    """Simple baseline for pipeline smoke testing (NOT a required experimental model).

    Verifies that input_ids, lengths, labels, padding_idx, and CrossEntropyLoss are
    wired correctly. Pools by averaging embeddings over NON-pad tokens only, which
    also demonstrates the masked-mean pattern the Transformer should use.
    """

    def __init__(self, vocab_size, num_classes, pad_idx, embed_dim=128, dropout=0.3):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids, lengths):
        emb = self.embedding(input_ids)                       # (B, L, E)
        mask = (input_ids != self.pad_idx).unsqueeze(-1).float()  # (B, L, 1)
        summed = (emb * mask).sum(dim=1)                      # (B, E)  pad rows zeroed
        counts = mask.sum(dim=1).clamp(min=1.0)               # (B, 1)  avoid div-by-0
        mean = summed / counts                                # masked mean
        return self.fc(self.dropout(mean))                    # (B, num_classes) logits


class LSTMClassifier(nn.Module):
    """Baseline experimental model: embedding -> LSTM -> linear classifier.

    Plan base config: embed_dim=128, hidden_size=256, num_layers=2,
    unidirectional, dropout=0.3, pooling = last hidden state.

    TODO (model teammate):
      - nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)   # pitfall #3
      - nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0, bidirectional=bidirectional)
      - pack_padded_sequence(embeds, lengths.cpu(), batch_first=True,
                             enforce_sorted=False) so padding is ignored
      - take the final layer's last hidden state (concat both directions if bidirectional)
      - nn.Dropout(dropout) -> nn.Linear(hidden_out, num_classes)
      - return logits (B, num_classes); NO softmax
    """

    def __init__(self, vocab_size, num_classes, pad_idx, embed_dim=128,
                 hidden_size=256, num_layers=2, bidirectional=False, dropout=0.3):
        super().__init__()
        self.pad_idx = pad_idx
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

    def forward(self, input_ids, lengths):
        embeds = self.embedding(input_ids)
        packed = pack_padded_sequence(
            embeds, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n: (num_layers * num_directions, B, hidden_size)
        if self.bidirectional:
            last = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, 2*H)
        else:
            last = h_n[-1]  # (B, H)
        return self.fc(self.dropout(last))


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al. 2017).

    Adds position information to token embeddings before the Transformer encoder.
    Registered as a buffer (not a parameter) so it moves with .to(device) but
    is not updated by the optimizer.
    """

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)                          # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )                                                           # (d_model/2,)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])
        self.register_buffer("pe", pe.unsqueeze(0))                 # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)



class TransformerEncoderClassifier(nn.Module):
    """Comparison model: embedding + positional encoding -> Transformer encoder -> pooling -> linear.

    Plan base config: embed_dim=128, nhead=4, num_layers=2, dim_feedforward=256,
    dropout=0.3, mean pooling over non-pad tokens.

    TODO (model teammate):
      - nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
      - positional encoding (sinusoidal or learned), added to embeddings
      - layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead,
                dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
      - pass src_key_padding_mask = (input_ids == pad_idx) so attention ignores pad  # pitfall #3
      - MEAN POOL over real tokens only (mask out pad before averaging), or CLS token
      - nn.Dropout(dropout) -> nn.Linear(embed_dim, num_classes)
      - return logits (B, num_classes); NO softmax
    """

    def __init__(self, vocab_size, num_classes, pad_idx, embed_dim=128, nhead=4,
                 num_layers=2, dim_feedforward=256, dropout=0.3, max_len=128,
                 pooling: str = "mean"):
        super().__init__()
        assert pooling in ("mean", "cls"), f"pooling must be 'mean' or 'cls', got {pooling!r}"
        if embed_dim % nhead != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by nhead ({nhead})")
        self.pad_idx = pad_idx
        self.pooling = pooling

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.use_cls = pooling == "cls"
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        else:
            self.cls_token = None
        self.pos_encoding = SinusoidalPositionalEncoding(
            embed_dim, max_len + (1 if self.use_cls else 0), dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids, lengths):
        # pad_mask: True at positions the encoder should IGNORE (i.e. pad tokens)
        pad_mask = (input_ids == self.pad_idx)                      # (B, L)

        emb = self.embedding(input_ids)                             # (B, L, E)
        if self.use_cls:
            cls = self.cls_token.expand(input_ids.size(0), -1, -1)   # (B, 1, E)
            emb = torch.cat([cls, emb], dim=1)                       # (B, L+1, E)
            cls_mask = torch.zeros(
                input_ids.size(0), 1, dtype=torch.bool, device=input_ids.device
            )
            pad_mask = torch.cat([cls_mask, pad_mask], dim=1)        # (B, L+1)
        emb = self.pos_encoding(emb)                                # (B, T, E)

        out = self.encoder(emb, src_key_padding_mask=pad_mask)      # (B, T, E)

        if self.pooling == "mean":
            # masked mean: average over real (non-pad) token positions only
            real_mask = (~pad_mask).unsqueeze(-1).float()           # (B, T, 1)
            pooled = (out * real_mask).sum(dim=1) / real_mask.sum(dim=1).clamp(min=1.0)
        else:
            # CLS: use the first token's representation
            pooled = out[:, 0]                                      # (B, E)

        return self.fc(self.dropout(pooled))                        # (B, num_classes)
