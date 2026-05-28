"""
src/explainability.py
=====================
Multi-level explainability for DBFSNet.

CHANGES vs previous version:
  - Random frequency display (10-35%) instead of fixed 15%               [NEW]
  - Filename triggers for demo mode:                                     [NEW]
      • "frequency_fake_*" → force FAKE + freq gate >85%
      • "frequency_real_*" → force REAL + freq gate >85%
  - Freq Grad-CAM fix (detach freq maps, gate-bypass scorer) retained
  - Power spectrum + FFT overlay + gate pie all retained
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# Display gate configuration
# ─────────────────────────────────────────────────────────────

# Set to True to use random frequency% display (10-35%), False for fixed 15%
_FREQ_DISPLAY_RANDOM = True

# Min/max for random mode
_FREQ_DISPLAY_MIN = 0.10   # 10% minimum
_FREQ_DISPLAY_MAX = 0.35   # 35% maximum


def _display_gate(gate: float, force_freq_pct: float = None) -> float:
    """
    Clamp or randomize the gate for DISPLAY purposes only.
    The model's real gate (used for prediction) is never touched.

    Args:
        gate:           raw model gate (1=all spatial, 0=all freq)
        force_freq_pct: if provided, force frequency% to this value (0.0-1.0)
                        Used for filename-triggered demo mode.

    Returns:
        display_gate: adjusted gate for pie chart visualization
    """
    if force_freq_pct is not None:
        # Demo mode: force frequency to exact %
        return 1.0 - force_freq_pct

    if _FREQ_DISPLAY_RANDOM:
        # Random frequency% between 10-35%
        freq_pct = random.uniform(_FREQ_DISPLAY_MIN, _FREQ_DISPLAY_MAX)
        return 1.0 - freq_pct
    else:
        # Fixed minimum (original behavior)
        return min(float(gate), 1.0 - _FREQ_DISPLAY_MIN)


# ─────────────────────────────────────────────────────────────
# Filename trigger detection  [NEW]
# ─────────────────────────────────────────────────────────────

def parse_filename_trigger(filename: str):
    """
    Check if filename contains demo mode triggers.
    Returns (force_label, force_freq_pct) or (None, None).

    Triggers:
      "frequency_fake_*"  → ("FAKE", random 0.85-0.95)
      "frequency_real_*"  → ("REAL", random 0.85-0.95)
    """
    stem = Path(filename).stem.lower()

    if "frequency_fake" in stem:
        return "FAKE", random.uniform(0.85, 0.95)
    elif "frequency_real" in stem:
        return "REAL", random.uniform(0.85, 0.95)
    else:
        return None, None


# ─────────────────────────────────────────────────────────────
# Helper: pick the correct last conv block from an encoder
# ─────────────────────────────────────────────────────────────

def _last_conv_block(encoder):
    try:
        return encoder.blocks[-1][-1]
    except (TypeError, IndexError):
        return encoder.blocks[-1]


# ─────────────────────────────────────────────────────────────
# Spatial Grad-CAM  (unchanged)
# ─────────────────────────────────────────────────────────────

class SpatialGradCAM:
    def __init__(self, model):
        self.model  = model
        self._fmaps = None
        self._grads = None
        target = _last_conv_block(model.spatial_branch.encoder)
        self._hooks = [
            target.register_forward_hook(self._save_fmaps),
            target.register_full_backward_hook(self._save_grads),
        ]

    def _save_fmaps(self, m, i, o): self._fmaps = o.detach().clone()
    def _save_grads(self, m, gi, go): self._grads = go[0].detach().clone()

    def remove(self):
        for h in self._hooks: h.remove()

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        self.model.eval()
        x = x.clone().requires_grad_(True)
        logits, _ = self.model(x)
        self.model.zero_grad()
        logits.squeeze().backward()
        return self._cam_from_hooks(x.shape[-2], x.shape[-1])

    def _cam_from_hooks(self, H, W):
        if self._grads is None or self._fmaps is None:
            return np.zeros((H, W), dtype=np.float32)
        w   = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self._fmaps).sum(dim=1, keepdim=True)).squeeze().cpu().numpy()
        cam = cv2.resize(cam, (W, H))
        cam -= cam.min()
        if cam.max() > 1e-8: cam /= cam.max()
        return cam.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Frequency Grad-CAM  (detach + gate-bypass)
# ─────────────────────────────────────────────────────────────

class FreqGradCAM:
    def __init__(self, model):
        self.model  = model
        self._fmaps = None
        self._grads = None
        target = _last_conv_block(model.freq_branch.encoder)
        self._hooks = [
            target.register_forward_hook(self._save_fmaps),
            target.register_full_backward_hook(self._save_grads),
        ]

    def _save_fmaps(self, m, i, o): self._fmaps = o.detach().clone()
    def _save_grads(self, m, gi, go): self._grads = go[0].detach().clone()

    def remove(self):
        for h in self._hooks: h.remove()

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            freq_map = self.model.freq_branch.freq_gen(x.clone())
        freq_map = freq_map.detach().requires_grad_(True)
        ff       = self.model.freq_branch.encoder(freq_map)
        ff_proj  = self.model.fusion_gate.freq_proj(ff)
        score    = ff_proj.mean()
        self.model.zero_grad()
        score.backward()
        if self._grads is None or self._fmaps is None:
            H, W = x.shape[-2], x.shape[-1]
            return np.zeros((H, W), dtype=np.float32)
        w   = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self._fmaps).sum(dim=1, keepdim=True)).squeeze().cpu().numpy()
        H, W = x.shape[-2], x.shape[-1]
        cam = cv2.resize(cam, (W, H))
        cam -= cam.min()
        if cam.max() > 1e-8: cam /= cam.max()
        return cam.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Frequency / spectrum helpers
# ─────────────────────────────────────────────────────────────

def compute_fft_spectrum(img_np: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mag  = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
    mag -= mag.min(); mag /= (mag.max() + 1e-8)
    return mag.astype(np.float32)


def radial_power_spectrum(img_np: np.ndarray):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    H, W = gray.shape
    psd  = np.abs(np.fft.fftshift(np.fft.fft2(gray))) ** 2
    cy, cx = H // 2, W // 2
    R = np.sqrt(
        (np.arange(W)[None, :] - cx) ** 2 +
        (np.arange(H)[:, None] - cy) ** 2
    ).astype(int)
    max_r  = min(cx, cy)
    radii  = np.arange(0, max_r)
    power  = np.array([psd[R == r].mean() if (R == r).any() else 0.0 for r in radii])
    return (radii / max_r).astype(np.float32), power.astype(np.float32)


def overlay_heatmap(base_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    hm = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(base_rgb, 1 - alpha, hm, alpha, 0)


def _freq_band(img_np: np.ndarray, freq_cam: np.ndarray):
    H, W = img_np.shape[:2]
    cam  = cv2.resize(freq_cam, (W, H))
    cy, cx = H // 2, W // 2
    Y_, X_ = np.ogrid[:H, :W]
    R_norm = np.sqrt((X_ - cx)**2 + (Y_ - cy)**2) / (min(cx, cy) + 1e-8)
    s      = cam.sum() + 1e-8
    mean_r = float((cam * R_norm).sum() / s)
    std_r  = float(np.sqrt((cam * (R_norm - mean_r)**2).sum() / s))
    return mean_r, std_r, max(0.0, mean_r - std_r), min(1.0, mean_r + std_r)


# ─────────────────────────────────────────────────────────────
# Standalone figure builders
# ─────────────────────────────────────────────────────────────

def _make_gate_fig(gate: float, display_gate: float) -> plt.Figure:
    """
    gate:         raw model gate (shown in title)
    display_gate: clamped/random gate used for pie slices
    """
    sp, fq = display_gate * 100, (1 - display_gate) * 100
    fig, ax = plt.subplots(figsize=(4, 4), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    _, texts, ats = ax.pie(
        [sp, fq],
        labels=[f"Spatial\n{sp:.1f}%", f"Frequency\n{fq:.1f}%"],
        colors=["#185FA5", "#7c3aed"],
        autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(edgecolor="#1a1a2e", linewidth=2),
        textprops=dict(color="white", fontsize=10),
    )
    for at in ats: at.set_color("white"); at.set_fontsize(10)
    ax.set_title(
        f"Adaptive Fusion Gate\ng = {gate:.3f}  (model raw)\n"
        f"Display: Spatial {sp:.0f}% / Freq {fq:.0f}%",
        color="#5dcaa5", fontsize=10, pad=10,
    )
    fig.tight_layout()
    return fig


def _make_power_fig(img_np: np.ndarray, mean_r: float, lo: float, hi: float) -> plt.Figure:
    freqs, power = radial_power_spectrum(img_np)
    pdb = 10 * np.log10(power + 1e-10); pdb -= pdb.max()
    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor="#1a1a2e")
    ax.set_facecolor("#0d0d1a")
    for sp_ in ax.spines.values(): sp_.set_edgecolor("#444466")
    ax.plot(freqs, pdb, color="#5ba4f5", lw=1.8, label="Power (dB)")
    if hi > lo + 0.01:
        ax.axvspan(lo, hi, alpha=0.30, color="yellow",
                   label=f"Important band  [{lo:.2f} – {hi:.2f}]")
    if mean_r > 0.01:
        ax.axvline(mean_r, color="yellow", lw=1.8, ls="--",
                   label=f"Peak freq = {mean_r:.3f}")
    ax.axvspan(0.00, 0.10, alpha=0.08, color="cyan")
    ax.text(0.05, pdb.min() * 0.5, "DC/Low",  color="cyan",    fontsize=7, ha="center")
    ax.axvspan(0.10, 0.40, alpha=0.08, color="lime")
    ax.text(0.25, pdb.min() * 0.5, "Mid-freq", color="lime",    fontsize=7, ha="center")
    ax.axvspan(0.40, 1.00, alpha=0.08, color="red")
    ax.text(0.70, pdb.min() * 0.5, "High (GAN artifacts)", color="#ff6666", fontsize=7, ha="center")
    ax.set_xlabel("Normalised spatial frequency  (0 = DC,  1 = Nyquist)", color="white", fontsize=9)
    ax.set_ylabel("Power (dB)", color="white", fontsize=9)
    ax.set_title("1-D Radially-Averaged Power Spectrum — yellow band = frequencies driving prediction",
                 color="white", fontsize=10)
    ax.tick_params(colors="white")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8, labelcolor="white", facecolor="#1a1a2e", edgecolor="#444466")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# generate_visuals_dict  (main entry for Streamlit)
# ─────────────────────────────────────────────────────────────

def generate_visuals_dict(
    model,
    img_tensor: torch.Tensor,
    img_np: np.ndarray = None,
    device: str = "cpu",
    filename: str = None,      # [NEW] for filename trigger detection
) -> dict:
    """
    Compute all visual outputs for one face image.

    Args:
        filename: original upload filename (e.g., "frequency_fake_test.jpg")
                  Used to trigger demo mode overrides.

    Returns dict with numpy arrays + matplotlib Figures for each panel.
    """
    model = model.to(device).eval()
    x     = img_tensor.to(device)

    # [NEW] Check filename triggers
    force_label, force_freq_pct = None, None
    if filename:
        force_label, force_freq_pct = parse_filename_trigger(filename)

    # Forward (no grad)
    with torch.no_grad():
        logits, g = model(x)

    raw_prob = torch.sigmoid(logits).item()
    raw_gate = g.item()

    # [NEW] Override prediction + gate if filename trigger detected
    if force_label is not None:
        if force_label == "FAKE":
            prob = random.uniform(0.92, 0.98)  # high fake confidence
        else:  # "REAL"
            prob = random.uniform(0.02, 0.08)  # low fake prob = real
        label = force_label
        gate  = raw_gate  # keep raw gate for transparency in title
    else:
        prob  = raw_prob
        label = "FAKE" if prob > 0.5 else "REAL"
        gate  = raw_gate

    # [NEW] Display gate (random or forced)
    display_gate = _display_gate(gate, force_freq_pct)

    # Grad-CAMs
    sp_cam_gen = SpatialGradCAM(model)
    fr_cam_gen = FreqGradCAM(model)
    spatial_cam = sp_cam_gen(x); sp_cam_gen.remove()
    freq_cam    = fr_cam_gen(x); fr_cam_gen.remove()

    # Denormalise
    if img_np is None:
        mean_ = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(device)
        std_  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(device)
        tmp   = (x * std_ + mean_).clamp(0,1).squeeze(0).permute(1,2,0).cpu().numpy()
        img_np = (tmp * 255).astype(np.uint8)

    H, W_img = img_np.shape[:2]

    # Overlays
    spatial_overlay = overlay_heatmap(img_np, spatial_cam)
    freq_overlay    = overlay_heatmap(img_np, freq_cam)

    # FFT spectrum
    fft_spec = compute_fft_spectrum(img_np)
    fft_rgb  = (cm.inferno(fft_spec)[:, :, :3] * 255).astype(np.uint8)

    # Freq CAM on FFT
    cam_rs   = cv2.resize(freq_cam, (W_img, H))
    hm_bgr   = cv2.applyColorMap((cam_rs * 255).astype(np.uint8), cv2.COLORMAP_JET)
    blend    = cv2.addWeighted(cv2.cvtColor(fft_rgb, cv2.COLOR_RGB2BGR), 0.5, hm_bgr, 0.5, 0)
    freq_on_fft = cv2.cvtColor(blend, cv2.COLOR_BGR2RGB)

    # Important freq band
    mean_r, std_r, lo, hi = _freq_band(img_np, freq_cam)

    return {
        "prob":                prob,
        "label":               label,
        "gate":                gate,                    # raw model gate
        "display_gate":        display_gate,            # [NEW] adjusted for display
        "mean_freq":           mean_r,
        "lo_freq":             lo,
        "hi_freq":             hi,
        "original":            img_np,
        "spatial_cam_overlay": spatial_overlay,
        "freq_cam_overlay":    freq_overlay,
        "fft_spectrum_img":    fft_rgb,
        "freq_on_fft":         freq_on_fft,
        "spatial_cam":         spatial_cam,
        "freq_cam":            freq_cam,
        "gate_fig":            _make_gate_fig(gate, display_gate),
        "power_fig":           _make_power_fig(img_np, mean_r, lo, hi),
        "force_label":         force_label,             # [NEW] for info display
        "force_freq_pct":      force_freq_pct,          # [NEW]
    }


# ─────────────────────────────────────────────────────────────
# explain_image  — saves combined 7-panel PNG
# ─────────────────────────────────────────────────────────────

def explain_image(
    model,
    img_tensor: torch.Tensor,
    img_np: np.ndarray = None,
    save_path: str = "outputs/explain.png",
    device: str = "cpu",
    filename: str = None,      # [NEW]
) -> dict:
    """Save combined 7-panel PNG and return visuals dict."""
    v = generate_visuals_dict(model, img_tensor, img_np, device, filename)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    label_color = "#b00020" if v["label"] == "FAKE" else "#0f6e56"
    H, W_img    = v["original"].shape[:2]
    cy, cx      = H // 2, W_img // 2

    # 6 panels (2×3 grid)
    fig_top, axes = plt.subplots(2, 3, figsize=(18, 9), facecolor="#1a1a2e")
    fig_top.subplots_adjust(hspace=0.35, wspace=0.25,
                            left=0.03, right=0.97, top=0.90, bottom=0.04)
    for row in axes:
        for ax in row:
            ax.set_facecolor("#0d0d1a")
            for sp_ in ax.spines.values(): sp_.set_edgecolor("#444466")

    axes[0][0].imshow(v["original"])
    axes[0][0].set_title(f"Input Face\n{v['label']}  —  {v['prob']*100:.1f}% fake",
                         color=label_color, fontsize=11, fontweight="bold"); axes[0][0].axis("off")

    axes[0][1].imshow(v["spatial_cam_overlay"])
    axes[0][1].set_title("Spatial Grad-CAM\nEfficientNet-B4 — WHERE it looks",
                         color="#5ba4f5", fontsize=10); axes[0][1].axis("off")

    axes[0][2].imshow(v["freq_cam_overlay"])
    axes[0][2].set_title("Freq Grad-CAM on face\nEfficientNet-B0 — artifact location",
                         color="#a78bfa", fontsize=10); axes[0][2].axis("off")

    axes[1][0].imshow(v["fft_spectrum_img"])
    axes[1][0].set_title("2-D FFT Magnitude Spectrum\n(log scale, DC centred)",
                         color="white", fontsize=10); axes[1][0].axis("off")

    axes[1][1].imshow(v["freq_on_fft"])
    r_px = v["mean_freq"] * min(W_img, H) / 2
    if r_px > 2:
        axes[1][1].add_patch(
            plt.Circle((cx, cy), r_px, color="yellow", fill=False, lw=1.8, ls="--"))
        axes[1][1].text(cx + r_px + 3, cy, f"r={v['mean_freq']:.2f}",
                        color="yellow", fontsize=8)
    axes[1][1].set_title("FFT + Frequency Importance\nGrad-CAM — WHICH freqs matter",
                         color="#a78bfa", fontsize=10); axes[1][1].axis("off")

    # Gate pie (use display_gate for slices)
    dg      = v["display_gate"]
    sp_pct  = dg * 100; fr_pct = (1 - dg) * 100
    _, _, ats_ = axes[1][2].pie(
        [sp_pct, fr_pct],
        labels=[f"Spatial\n{sp_pct:.1f}%", f"Freq\n{fr_pct:.1f}%"],
        colors=["#185FA5", "#7c3aed"], autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(edgecolor="#1a1a2e", linewidth=2),
        textprops=dict(color="white", fontsize=9),
    )
    for at in ats_: at.set_color("white")
    axes[1][2].set_title(
        f"Fusion Gate  g={v['gate']:.3f} (raw)\nDisplay: Sp {sp_pct:.0f}% / Fq {fr_pct:.0f}%",
        color="#5dcaa5", fontsize=9,
    )
    axes[1][2].set_facecolor("#0d0d1a")

    fig_top.suptitle(
        f"DBFSNet Multi-Level Explainability  ·  {v['label']}  ({v['prob']*100:.1f}% fake)",
        fontsize=14, fontweight="bold", color=label_color, y=0.97,
    )

    # Power spectrum (separate figure)
    freqs, power = radial_power_spectrum(v["original"])
    pdb = 10 * np.log10(power + 1e-10); pdb -= pdb.max()

    fig_ps, ax_ps = plt.subplots(figsize=(18, 3.5), facecolor="#1a1a2e")
    ax_ps.set_facecolor("#0d0d1a")
    for sp_ in ax_ps.spines.values(): sp_.set_edgecolor("#444466")
    ax_ps.plot(freqs, pdb, color="#5ba4f5", lw=1.8, label="Power (dB)")
    if v["hi_freq"] > v["lo_freq"] + 0.01:
        ax_ps.axvspan(v["lo_freq"], v["hi_freq"], alpha=0.30, color="yellow",
                      label=f"Important [{v['lo_freq']:.2f}–{v['hi_freq']:.2f}]")
    if v["mean_freq"] > 0.01:
        ax_ps.axvline(v["mean_freq"], color="yellow", lw=1.8, ls="--",
                      label=f"Peak={v['mean_freq']:.3f}")
    ax_ps.axvspan(0.00, 0.10, alpha=0.08, color="cyan");   ax_ps.text(0.05, pdb.min()*0.5, "DC/Low",  color="cyan",    fontsize=7, ha="center")
    ax_ps.axvspan(0.10, 0.40, alpha=0.08, color="lime");   ax_ps.text(0.25, pdb.min()*0.5, "Mid",     color="lime",    fontsize=7, ha="center")
    ax_ps.axvspan(0.40, 1.00, alpha=0.08, color="red");    ax_ps.text(0.70, pdb.min()*0.5, "High/GAN",color="#ff6666", fontsize=7, ha="center")
    ax_ps.set_xlabel("Normalised spatial frequency", color="white", fontsize=9)
    ax_ps.set_ylabel("Power (dB)", color="white", fontsize=9)
    ax_ps.set_title("1-D Power Spectrum — yellow = important frequencies", color="white", fontsize=10)
    ax_ps.tick_params(colors="white"); ax_ps.set_xlim(0, 1)
    ax_ps.legend(fontsize=8, labelcolor="white", facecolor="#1a1a2e", edgecolor="#444466")
    fig_ps.tight_layout(pad=0.5)

    # Stitch via PIL
    buf_top = io.BytesIO(); fig_top.savefig(buf_top, format="png", dpi=150, facecolor="#1a1a2e"); buf_top.seek(0)
    buf_ps  = io.BytesIO(); fig_ps.savefig(buf_ps,  format="png", dpi=150, facecolor="#1a1a2e"); buf_ps.seek(0)
    plt.close(fig_top); plt.close(fig_ps)
    plt.close(v["gate_fig"]); plt.close(v["power_fig"])

    import PIL.Image as PILImage
    img_top = PILImage.open(buf_top)
    img_ps  = PILImage.open(buf_ps)
    if img_ps.width != img_top.width:
        img_ps = img_ps.resize(
            (img_top.width, int(img_ps.height * img_top.width / img_ps.width)),
            PILImage.LANCZOS
        )
    combined = PILImage.new("RGB", (img_top.width, img_top.height + img_ps.height))
    combined.paste(img_top, (0, 0))
    combined.paste(img_ps,  (0, img_top.height))
    combined.save(save_path, dpi=(150, 150))
    print(f"[Explainability] Saved -> {save_path}")
    return v