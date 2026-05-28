"""
src/infer.py
============
Inference script for DBFSNet.

CHANGES vs previous version:
  - --explain flag now defaults to True for images (always saves heatmaps)   [CHANGED]
  - Video inference saves per-frame explain PNGs for the top-3 frames         [NEW]
  - Added --no_explain flag to skip visualisation if you want speed            [NEW]
  - Suppressed FutureWarning spam from torch.load via weights_only=False      [CHANGED]
  - All other logic unchanged
"""
import sys, os
# Suppress torch FutureWarning about weights_only
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import cv2
import numpy as np
import torch
from pathlib import Path
from torchvision import transforms
from PIL import Image

from src.model          import DBFSNet
from src.explainability import explain_image


# ── Preprocessing ─────────────────────────────────────────────

def preprocess(face_bgr: np.ndarray, size: int = 224) -> torch.Tensor:
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size))
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return t(rgb).unsqueeze(0)   # (1, 3, H, W)


def detect_face(frame_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    """Detect + crop face with MTCNN. Falls back to full frame."""
    try:
        from facenet_pytorch import MTCNN
        m    = MTCNN(image_size=size, margin=20, keep_all=False, post_process=False)
        face = m(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
        if face is not None:
            return face.permute(1, 2, 0).numpy().astype(np.uint8)
    except Exception:
        pass
    return cv2.resize(frame_bgr, (size, size))


# ── Optical flow weight ───────────────────────────────────────

def optical_flow_weight(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(mag.mean())


# ── Video inference ───────────────────────────────────────────

def infer_video(
    path: str, model, device: str,
    num_frames: int = 30, size: int = 224,
    explain: bool = True, out_dir: str = "outputs",
) -> dict:
    """
    Optical-flow-weighted voting over sampled frames.
    If explain=True, saves explainability PNGs for top 3 most-weighted frames.
    """
    cap   = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs  = np.linspace(0, total - 1, num_frames, dtype=int)
    frames_bgr = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, f = cap.read()
        if ret:
            frames_bgr.append(f)
    cap.release()

    if not frames_bgr:
        return {"error": "No frames could be extracted"}

    # Optical flow weights
    weights = [1.0]
    for i in range(1, len(frames_bgr)):
        pg = cv2.cvtColor(frames_bgr[i-1], cv2.COLOR_BGR2GRAY)
        cg = cv2.cvtColor(frames_bgr[i],   cv2.COLOR_BGR2GRAY)
        weights.append(max(optical_flow_weight(pg, cg), 0.1))
    weights = np.array(weights)
    weights /= weights.sum()

    # Per-frame prediction
    model.eval()
    probs, gates, faces_rgb = [], [], []
    for fr in frames_bgr:
        face   = detect_face(fr, size)
        face_r = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) if face.shape[-1] == 3 else face
        tensor = preprocess(face, size).to(device)
        with torch.no_grad():
            logits, g = model(tensor)
        probs.append(torch.sigmoid(logits).item())
        gates.append(g.item())
        faces_rgb.append(face_r)

    weighted_prob = float(np.dot(weights, probs))
    label = "FAKE" if weighted_prob > 0.5 else "REAL"

    # [NEW] Save explainability for top-3 highest-weighted frames
    if explain:
        top3 = np.argsort(weights)[::-1][:3]
        stem = Path(path).stem
        for rank, fi in enumerate(top3):
            tensor = preprocess(
                cv2.cvtColor(faces_rgb[fi], cv2.COLOR_RGB2BGR), size
            ).to(device)
            sp = str(Path(out_dir) / f"{stem}_frame{fi:03d}_rank{rank+1}_explain.png")
            explain_image(model, tensor, img_np=faces_rgb[fi],
                          save_path=sp, device=device)
        print(f"[Video] Explain PNGs saved for top-3 frames in: {out_dir}")

    return {
        "label":       label,
        "fake_prob":   weighted_prob,
        "confidence":  weighted_prob if label == "FAKE" else 1 - weighted_prob,
        "mean_gate":   float(np.mean(gates)),
        "frame_probs": probs,
    }


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DBFSNet inference with multi-level explainability"
    )
    parser.add_argument("--input",      required=True,
                        help="Path to image (.jpg/.png) or video (.mp4/.avi)")
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to .pth — defaults to checkpoints/best.pth")
    # CHANGED: --explain is now on by default; use --no_explain to skip
    parser.add_argument("--no_explain", action="store_true",
                        help="Skip Grad-CAM visualisation (faster)")
    args = parser.parse_args()

    explain = not args.no_explain   # [CHANGED] default True

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.checkpoint or str(
        Path(cfg["paths"]["checkpoint_dir"]) / "best.pth"
    )
    out_dir   = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = DBFSNet(
        spatial_feat_dim = cfg["model"]["spatial_feat_dim"],
        freq_feat_dim    = cfg["model"]["freq_feat_dim"],
        gate_proj_dim    = cfg["model"]["gate_hidden_dim"],
        gate_hidden_dim  = cfg["model"]["gate_hidden_dim"],
        dropout          = cfg["model"]["dropout"],
    ).to(device)

    # [CHANGED] weights_only=False silences FutureWarning
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Model loaded: {ckpt_path}\n")

    sz  = cfg["data"]["image_size"]
    ext = Path(args.input).suffix.lower()

    # ── VIDEO ────────────────────────────────────────────────
    if ext in {".mp4", ".avi", ".mov", ".mkv"}:
        result = infer_video(
            args.input, model, str(device),
            num_frames=cfg["data"]["frames_per_video"],
            size=sz,
            explain=explain,
            out_dir=str(out_dir),
        )
        if "error" in result:
            print(f"Error: {result['error']}")
            return
        print("=" * 45)
        print(f"  Prediction : {result['label']}")
        print(f"  Fake prob  : {result['fake_prob']*100:.1f}%")
        print(f"  Confidence : {result['confidence']*100:.1f}%")
        print(f"  Mean gate  : {result['mean_gate']:.3f}  (1=spatial, 0=freq)")
        print("=" * 45)

    # ── IMAGE ────────────────────────────────────────────────
    else:
        img_bgr = cv2.imread(args.input)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read: {args.input}")

        face    = detect_face(img_bgr, sz)
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        tensor  = preprocess(face, sz).to(device)

        with torch.no_grad():
            logits, g = model(tensor)
        prob  = torch.sigmoid(logits).item()
        label = "FAKE" if prob > 0.5 else "REAL"

        print("=" * 45)
        print(f"  Prediction : {label}")
        print(f"  Fake prob  : {prob*100:.1f}%")
        print(f"  Confidence : {(prob if label=='FAKE' else 1-prob)*100:.1f}%")
        print(f"  Gate (g)   : {g.item():.3f}  (1=spatial, 0=freq)")
        print("=" * 45)

        # [CHANGED] Always produce explainability unless --no_explain
        if explain:
            stem      = Path(args.input).stem
            save_path = str(out_dir / f"{stem}_explain.png")
            explain_image(
                model, tensor,
                img_np=face_rgb,
                save_path=save_path,
                device=str(device),
            )
            print(f"\n  Explainability saved -> {save_path}")


if __name__ == "__main__":
    main()
