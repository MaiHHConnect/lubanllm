"""Trainable decision head for an early representation of the shared backbone.

Runtime supplies a four-layer representation and labelled calibration; this
module only defines the head. Untrained scores are not reliable decisions.
"""
import torch
from torch import nn


class FastDecisionHead(nn.Module):
    actions = ("local_core", "neural_expert", "teacher")

    def __init__(self, hidden_size: int):
        super().__init__()
        self.projection = nn.Linear(hidden_size, len(self.actions))

    def forward(self, hidden_states, attention_mask):
        if hidden_states.ndim != 3 or attention_mask.shape != hidden_states.shape[:2]:
            raise ValueError("expected hidden [batch, tokens, hidden] and mask [batch, tokens]")
        mask = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
        counts = mask.sum(dim=1)
        if torch.any(counts == 0):
            raise ValueError("decision requires at least one unmasked token")
        pooled = (hidden_states * mask).sum(dim=1) / counts
        return self.projection(pooled)
