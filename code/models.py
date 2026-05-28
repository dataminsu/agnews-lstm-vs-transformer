

from __future__ import annotations

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
                 num_layers=2, dim_feedforward=256, dropout=0.3, max_len=128, pooling="mean"):
        super().__init__()
        # TODO: define embedding, positional encoding, encoder, dropout, fc
        raise NotImplementedError("TransformerEncoderClassifier.__init__")

    def forward(self, input_ids, lengths):
        # TODO: return logits of shape (batch, num_classes)
        raise NotImplementedError("TransformerEncoderClassifier.forward")
