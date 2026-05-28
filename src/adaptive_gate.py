"""
adaptive_gate.py
NEW — Adaptive fusion gate.

  g = sigmoid( FC( concat(f_spatial, f_freq) ) )   scalar in [0,1]
  fused = g * f_spatial_proj + (1-g) * f_freq_proj
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn


class AdaptiveFusionGate(nn.Module):
    """
    Inputs:
        f_s : (B, spatial_dim)   e.g. 1792 from EfficientNet-B4
        f_f : (B, freq_dim)      e.g. 1280 from EfficientNet-B0

    Output:
        fused : (B, proj_dim)    gated feature
        g     : (B, 1)           gate value  (1=full spatial, 0=full freq)
    """

    def __init__(
        self,
        spatial_dim: int = 1792,
        freq_dim: int    = 1280,
        proj_dim: int    = 512,
        hidden_dim: int  = 512,
        dropout: float   = 0.3,
    ):
        super().__init__()

        self.spatial_proj = nn.Sequential(
            nn.Linear(spatial_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
        )
        self.freq_proj = nn.Sequential(
            nn.Linear(freq_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
        )
        self.gate_fc = nn.Sequential(
            nn.Linear(proj_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, f_spatial: torch.Tensor, f_freq: torch.Tensor):
        fs = self.spatial_proj(f_spatial)        # (B, proj_dim)
        ff = self.freq_proj(f_freq)              # (B, proj_dim)
        g  = self.gate_fc(torch.cat([fs, ff], dim=1))   # (B, 1)
        fused = g * fs + (1.0 - g) * ff         # (B, proj_dim)
        return fused, g
