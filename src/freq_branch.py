"""
freq_branch.py
NEW — Frequency Map Generator + EfficientNet-B0 encoder.

Generates a 9-channel frequency tensor from an RGB image:
  ch 0-2 : FFT magnitude spectrum (log-scaled, per channel)
  ch 3-5 : DCT coefficient map (per channel)
  ch 6-8 : High-frequency residual (image - Gaussian blur)

Then passes the 9-ch map through EfficientNet-B0 -> 1280-d feature vector.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class FreqMapGenerator(nn.Module):
    """
    Input : (B, 3, H, W)
    Output: (B, 9, H, W)  — FFT(3) + DCT(3) + Residual(3)
    """

    def __init__(self, blur_sigma: float = 2.0, blur_kernel: int = 11):
        super().__init__()
        self.blur_kernel = blur_kernel
        kernel = self._make_gaussian_kernel(blur_kernel, blur_sigma)
        self.register_buffer("gauss_kernel", kernel)   # (3, 1, K, K)

    @staticmethod
    def _make_gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-0.5 * (coords / sigma) ** 2)
        g = g / g.sum()
        k2d = torch.outer(g, g)
        return k2d.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)  # (3,1,K,K)

    def _fft_maps(self, x):
        fft = torch.fft.fft2(x)
        fft = torch.fft.fftshift(fft, dim=(-2, -1))
        return torch.log1p(torch.abs(fft))             # (B, 3, H, W)

    def _dct_maps(self, x):
        B, C, H, W = x.shape
        xm = torch.cat([x, x.flip(-1)], dim=-1)
        xm = torch.cat([xm, xm.flip(-2)], dim=-2)
        dct = torch.abs(torch.fft.rfft2(xm))[..., :H, :W]
        mn = dct.amin(dim=(-1, -2), keepdim=True)
        mx = dct.amax(dim=(-1, -2), keepdim=True)
        return (dct - mn) / (mx - mn + 1e-8)           # (B, 3, H, W)

    def _residual_maps(self, x):
        pad = self.blur_kernel // 2
        blurred = F.conv2d(x, self.gauss_kernel, padding=pad, groups=3)
        res = x - blurred
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        return res                                      # (B, 3, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            self._fft_maps(x),
            self._dct_maps(x),
            self._residual_maps(x),
        ], dim=1)                                       # (B, 9, H, W)


class FreqBranch(nn.Module):
    """
    RGB image -> FreqMapGenerator (9ch) -> EfficientNet-B0 -> 1280-d feature
    """

    def __init__(self, freq_channels: int = 9, out_dim: int = 1280, pretrained: bool = True):
        super().__init__()
        self.freq_gen = FreqMapGenerator()

        backbone = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=0)

        # Replace first conv to accept 9 channels instead of 3
        old_conv = backbone.conv_stem
        new_conv = nn.Conv2d(
            freq_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            # Initialise by averaging the 3-channel weights across freq_channels
            avg_w = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(avg_w.repeat(1, freq_channels, 1, 1))
        backbone.conv_stem = new_conv

        self.encoder = backbone
        self.out_dim  = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)  ->  returns (B, 1280)
        freq_map = self.freq_gen(x)
        print(freq_map.shape)
        if freq_map.shape[1] == 1:
         freq_map = freq_map.repeat(1, 3, 1, 1)

        return self.encoder(freq_map)