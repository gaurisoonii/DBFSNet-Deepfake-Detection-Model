"""
spatial_branch.py
EfficientNet-B4 spatial branch.
Input : (B, 3, 224, 224)
Output: (B, 1792)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import timm


class SpatialBranch(nn.Module):
    def __init__(self, pretrained: bool = True, out_dim: int = 1792):
        super().__init__()
        self.encoder = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=0,   # remove classifier head, return pooled features
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, 3, H, W)  ->  returns (B, 1792)
        return self.encoder(x)
