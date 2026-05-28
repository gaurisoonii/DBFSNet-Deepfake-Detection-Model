"""
model.py
Full DBFSNet: Dual-Branch Frequency-Spatial Network.

Architecture:
  Input (3,224,224)
    |
    +-- Spatial branch --> EfficientNet-B4 --> (1792) -+
    |                                                   |--> AdaptiveFusionGate --> FC --> logit
    +-- Freq branch    --> FreqMapGen(9ch)              |
                          + EfficientNet-B0 --> (1280) -+
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from src.spatial_branch import SpatialBranch
from src.freq_branch    import FreqBranch
from src.adaptive_gate  import AdaptiveFusionGate


class DBFSNet(nn.Module):
    def __init__(
        self,
        spatial_pretrained: bool = True,
        freq_pretrained:    bool = True,
        spatial_feat_dim:   int  = 1792,
        freq_feat_dim:      int  = 1280,
        gate_proj_dim:      int  = 512,
        gate_hidden_dim:    int  = 512,
        dropout:            float = 0.3,
        num_classes:        int  = 1,
    ):
        super().__init__()
        self.spatial_branch = SpatialBranch(pretrained=spatial_pretrained, out_dim=spatial_feat_dim)
        self.freq_branch    = FreqBranch(freq_channels=9, out_dim=freq_feat_dim, pretrained=freq_pretrained)
        self.fusion_gate    = AdaptiveFusionGate(
            spatial_dim=spatial_feat_dim,
            freq_dim=freq_feat_dim,
            proj_dim=gate_proj_dim,
            hidden_dim=gate_hidden_dim,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gate_proj_dim, num_classes),
        )

    def forward(self, x: torch.Tensor):
        """
        x      : (B, 3, H, W)
        returns: logits (B,1),  gate g (B,1)
        """
        f_spatial        = self.spatial_branch(x)
        f_freq           = self.freq_branch(x)
        fused, g         = self.fusion_gate(f_spatial, f_freq)
        logits           = self.classifier(fused)
        return logits, g

    def predict(self, x: torch.Tensor):
        logits, g = self.forward(x)
        prob  = torch.sigmoid(logits).squeeze(1)
        label = (prob > 0.5).long()
        return prob, label, g
