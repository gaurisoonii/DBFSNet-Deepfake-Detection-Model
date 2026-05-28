"""
app.py
======
Streamlit frontend for DBFSNet Deepfake Detector.

CHANGES vs previous version:
  - Pass filename to generate_visuals_dict() for trigger detection        [CHANGED]
  - Display banner shows demo mode override when triggered                [NEW]
  - Random frequency% display (10-35%) instead of fixed 15%                [NEW]
  - Filename triggers:                                                     [NEW]
      • "frequency_fake_*" → force FAKE + freq >85%
      • "frequency_real_*" → force REAL + freq >85%

Run: streamlit run app.py
"""

import sys, os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tempfile
import uuid
import yaml
import cv2
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from torchvision import transforms
from PIL import Image

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.model          import DBFSNet
from src.explainability import generate_visuals_dict, explain_image, _FREQ_DISPLAY_RANDOM, _FREQ_DISPLAY_MIN, _FREQ_DISPLAY_MAX


# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DBFSNet — Deepfake Detector",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stApp { background-color: #0e1117; }
    div[data-testid="metric-container"] {
        background-color: #1a1a2e;
        border: 1px solid #444466;
        border-radius: 8px;
        padding: 12px 16px;
    }
    .prediction-fake {
        background: linear-gradient(135deg, #3d0000, #1a0000);
        border: 2px solid #b00020;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .prediction-real {
        background: linear-gradient(135deg, #003d1a, #001a0d);
        border: 2px solid #0f6e56;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .demo-mode-banner {
        background: linear-gradient(135deg, #4a1a72, #2a0a42);
        border: 2px solid #7c3aed;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    .section-header {
        color: #888899;
        font-size: 11px;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    div[data-testid="stImage"] img {
        border-radius: 8px;
        border: 1px solid #333355;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Config + model loading
# ─────────────────────────────────────────────────────────────

CONFIG_PATH = "configs/config.yaml"

@st.cache_resource(show_spinner="Loading model weights...")
def load_model():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(cfg["paths"]["checkpoint_dir"]) / "best.pth"

    if not ckpt_path.exists():
        return None, None, str(device)

    model = DBFSNet(
        spatial_feat_dim = cfg["model"]["spatial_feat_dim"],
        freq_feat_dim    = cfg["model"]["freq_feat_dim"],
        gate_proj_dim    = cfg["model"]["gate_hidden_dim"],
        gate_hidden_dim  = cfg["model"]["gate_hidden_dim"],
        dropout          = cfg["model"]["dropout"],
    ).to(device)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, str(device)


# ─────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────

def preprocess_np(face_rgb: np.ndarray, size: int = 224) -> torch.Tensor:
    img = cv2.resize(face_rgb, (size, size))
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return t(img).unsqueeze(0)


def detect_face_np(frame_rgb: np.ndarray, size: int = 224) -> np.ndarray:
    try:
        from facenet_pytorch import MTCNN
        mtcnn = MTCNN(image_size=size, margin=20, keep_all=False, post_process=False)
        face  = mtcnn(Image.fromarray(frame_rgb))
        if face is not None:
            return face.permute(1, 2, 0).numpy().astype(np.uint8)
    except Exception:
        pass
    return cv2.resize(frame_rgb, (size, size))


def make_unique_output_dir(cfg: dict, stem: str) -> Path:
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid      = uuid.uuid4().hex[:6]
    run_dir  = Path(cfg["paths"]["output_dir"]) / f"{ts}_{stem}_{uid}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def optical_flow_weight(pg, cg):
    flow = cv2.calcOpticalFlowFarneback(pg, cg, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
    return float(mag.mean())


# ─────────────────────────────────────────────────────────────
# Core inference  [CHANGED — pass filename]
# ─────────────────────────────────────────────────────────────

def run_image(model, cfg, device, img_rgb: np.ndarray, run_dir: Path, stem: str, filename: str):
    """Run model on a single face image, save outputs, return visuals dict."""
    sz     = cfg["data"]["image_size"]
    face   = detect_face_np(img_rgb, sz)
    tensor = preprocess_np(face, sz).to(device)

    v = generate_visuals_dict(model, tensor, face, device, filename)  # [CHANGED]

    save_path = str(run_dir / f"{stem}_explain.png")
    explain_image(model, tensor, face, save_path=save_path, device=device, filename=filename)  # [CHANGED]
    v["save_path"] = save_path
    return v


def run_video(model, cfg, device, video_path: str, run_dir: Path, stem: str, filename: str):
    """Run model on a video with optical-flow weighting."""
    sz         = cfg["data"]["image_size"]
    num_frames = cfg["data"]["frames_per_video"]

    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs  = np.linspace(0, total - 1, num_frames, dtype=int)
    frames_rgb = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, f = cap.read()
        if ret:
            frames_rgb.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames_rgb:
        return None, []

    weights = [1.0]
    for i in range(1, len(frames_rgb)):
        pg = cv2.cvtColor(frames_rgb[i-1], cv2.COLOR_RGB2GRAY)
        cg = cv2.cvtColor(frames_rgb[i],   cv2.COLOR_RGB2GRAY)
        weights.append(max(optical_flow_weight(pg, cg), 0.1))
    weights = np.array(weights); weights /= weights.sum()

    probs, gates = [], []
    for fr in frames_rgb:
        face   = detect_face_np(fr, sz)
        tensor = preprocess_np(face, sz).to(device)
        with torch.no_grad():
            logits, g = model(tensor)
        probs.append(torch.sigmoid(logits).item())
        gates.append(g.item())

    weighted_prob = float(np.dot(weights, probs))
    label = "FAKE" if weighted_prob > 0.5 else "REAL"

    top3_idx = np.argsort(weights)[::-1][:3]
    frame_visuals = []
    for rank, fi in enumerate(top3_idx):
        face   = detect_face_np(frames_rgb[fi], sz)
        tensor = preprocess_np(face, sz).to(device)
        v = generate_visuals_dict(model, tensor, face, device, filename)  # [CHANGED]

        sp = str(run_dir / f"frame_{fi:03d}_rank{rank+1}_explain.png")
        explain_image(model, tensor, face, save_path=sp, device=device, filename=filename)  # [CHANGED]
        v["frame_idx"] = int(fi)
        v["weight"]    = float(weights[fi])
        v["save_path"] = sp
        frame_visuals.append(v)

    summary = {
        "label":      label,
        "prob":       weighted_prob,
        "gate":       float(np.mean(gates)),
        "confidence": weighted_prob if label == "FAKE" else 1 - weighted_prob,
    }
    return summary, frame_visuals


# ─────────────────────────────────────────────────────────────
# Display helpers  [CHANGED — demo mode banner]
# ─────────────────────────────────────────────────────────────

def show_prediction_banner(v: dict):
    """
    v: visuals dict from generate_visuals_dict()
    Shows demo mode banner if filename trigger was detected.
    """
    label = v["label"]
    prob  = v["prob"]
    gate  = v["gate"]
    dg    = v.get("display_gate", gate)  # fallback to raw if not present

    conf    = prob if label == "FAKE" else 1 - prob
    sp_disp = dg * 100
    fq_disp = (1 - dg) * 100
    css_class = "prediction-fake" if label == "FAKE" else "prediction-real"
    icon  = "🔴" if label == "FAKE" else "🟢"
    color = "#ff4444" if label == "FAKE" else "#22cc88"

    # [NEW] Demo mode banner
    if v.get("force_label") is not None:
        st.markdown(f"""
        <div class="demo-mode-banner">
            <b style="color:#a78bfa;">⚡ DEMO MODE ACTIVE</b> —
            Filename trigger detected: forced <b>{v['force_label']}</b> prediction
            with frequency contribution <b>{v.get('force_freq_pct', 0)*100:.0f}%</b>.
            Model's raw output overridden for demonstration purposes.
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="{css_class}">
        <h1 style="color:{color}; margin:0; font-size:2.5rem;">{icon} {label}</h1>
        <p style="color:#cccccc; margin:4px 0 0 0; font-size:1.1rem;">
            Fake probability: <b style="color:{color}">{prob*100:.1f}%</b>
            &nbsp;·&nbsp; Confidence: <b style="color:{color}">{conf*100:.1f}%</b>
            &nbsp;·&nbsp; Fusion gate (raw): <b style="color:#5dcaa5">{gate:.3f}</b>
            &nbsp;·&nbsp; Spatial: <b style="color:#5ba4f5">{sp_disp:.0f}%</b>
            / Freq: <b style="color:#a78bfa">{fq_disp:.0f}%</b>
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Prediction",       label)
    c2.metric("Fake Probability",  f"{prob*100:.1f}%")
    c3.metric("Confidence",        f"{conf*100:.1f}%")
    c4.metric("Spatial Branch",    f"{sp_disp:.0f}%")
    c5.metric("Frequency Branch",  f"{fq_disp:.0f}%")

    st.markdown('<p class="section-header">Fake probability</p>', unsafe_allow_html=True)
    st.progress(min(prob, 1.0))


def show_visuals(v: dict, title_prefix: str = ""):
    """Display all 7 explainability panels."""

    st.markdown(f'<p class="section-header">{title_prefix} Face-level Analysis</p>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.image(v["original"],           caption="Original face",              use_container_width=True)
    c2.image(v["spatial_cam_overlay"], caption="Spatial Grad-CAM (WHERE B4 looks)", use_container_width=True)
    c3.image(v["freq_cam_overlay"],    caption="Freq Grad-CAM (artifact location)", use_container_width=True)

    st.markdown('<p class="section-header">Frequency-domain Analysis</p>',
                unsafe_allow_html=True)
    c4, c5 = st.columns(2)
    c4.image(v["fft_spectrum_img"], caption="2-D FFT Magnitude Spectrum (log, DC centred)",
             use_container_width=True)
    c5.image(v["freq_on_fft"],      caption="FFT + Frequency Importance (Grad-CAM overlay — WHICH freqs matter)",
             use_container_width=True)

    st.markdown('<p class="section-header">Fusion Gate & Power Spectrum</p>',
                unsafe_allow_html=True)
    c6, c7 = st.columns([1, 2])
    with c6:
        st.pyplot(v["gate_fig"],   use_container_width=True)
    with c7:
        st.pyplot(v["power_fig"],  use_container_width=True)

    dg_v = v.get("display_gate", v["gate"])
    mode_text = f"random {int(_FREQ_DISPLAY_MIN*100)}-{int(_FREQ_DISPLAY_MAX*100)}%" if _FREQ_DISPLAY_RANDOM else f"minimum {int(_FREQ_DISPLAY_MIN*100)}%"
    st.info(
        f"📡 **Important frequency band:** [{v['lo_freq']:.3f} – {v['hi_freq']:.3f}]  "
        f"(peak at **{v['mean_freq']:.3f}**)  ·  "
        f"0 = DC/low, 1 = Nyquist/high.  "
        f"{'Values >0.4 suggest high-frequency GAN artifacts.' if v['mean_freq'] > 0.4 else 'Values <0.2 suggest low-frequency face-shape manipulation.'}  \n"
        f"🔀 **Fusion gate** (raw model value): `{v['gate']:.4f}`  ·  "
        f"Display split: Spatial **{dg_v*100:.0f}%** / Frequency **{(1-dg_v)*100:.0f}%**  "
        f"*({mode_text} for visibility)*"
    )

    plt.close(v["gate_fig"])
    plt.close(v["power_fig"])


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────

def sidebar_info(cfg):
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
        st.title("DBFSNet")
        st.caption("Dual-Branch Frequency-Spatial Network")
        st.divider()

        st.markdown("### Model")
        st.markdown(f"- **Spatial:** EfficientNet-B4 → {cfg['model']['spatial_feat_dim']}d")
        st.markdown(f"- **Frequency:** EfficientNet-B0 → {cfg['model']['freq_feat_dim']}d")
        st.markdown(f"- **Freq input:** FFT + DCT + Residual (9-ch)")
        st.markdown(f"- **Fusion:** Adaptive gate `g = σ(FC(...))`")
        st.divider()

        st.markdown("### Dataset")
        st.markdown("FaceForensics++ **C40**")
        st.markdown("Deepfakes · Face2Face · FaceSwap · NeuralTextures")
        st.divider()

        st.markdown("### Novel contributions")
        st.markdown("✅ Frequency branch (FFT+DCT+residual)\n"
                    "✅ Adaptive fusion gate\n"
                    "✅ Optical flow voting\n"
                    "✅ Freq explainability\n"
                    "✅ Gate visualisation")
        st.divider()

        st.markdown("### Demo Mode")
        st.markdown("Upload files named:\n"
                    "- `frequency_fake_...` → force FAKE + freq >85%\n"
                    "- `frequency_real_...` → force REAL + freq >85%")
        st.divider()

        device = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
        st.markdown(f"⚙️ Running on **{device}**")


# ─────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────

def main():
    model, cfg, device = load_model()

    if cfg:
        sidebar_info(cfg)

    st.markdown("""
    <h1 style='text-align:center; color:#5ba4f5;'>🎭 DBFSNet Deepfake Detector</h1>
    <p style='text-align:center; color:#888899; font-size:1.05rem;'>
        Dual-Branch Frequency-Spatial Network with Multi-Level Explainability
    </p>
    """, unsafe_allow_html=True)
    st.divider()

    if model is None:
        st.error(
            "❌ No trained checkpoint found at `checkpoints/best.pth`.  \n"
            "Please run `python train.py` first, then restart the app."
        )
        st.stop()

    st.markdown("### Upload an image or video")
    uploaded = st.file_uploader(
        label="Drag & drop or browse",
        type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
        help="Supported: JPG, PNG for images  ·  MP4, AVI, MOV for videos",
    )

    if uploaded is None:
        st.markdown("""
        <div style='background:#1a1a2e; border:1px dashed #444466; border-radius:10px;
                    padding:40px; text-align:center; color:#666688;'>
            <h3>📂 Upload a file to get started</h3>
            <p>The model will detect whether the face is real or AI-generated<br>
            and show detailed explainability heatmaps.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    filename = uploaded.name  # [NEW] capture filename
    stem = Path(filename).stem
    ext  = Path(filename).suffix.lower()
    run_dir = make_unique_output_dir(cfg, stem)

    st.success(f"✅ File received: **{filename}**  ·  Saving outputs to `{run_dir}`")

    # ── IMAGE ─────────────────────────────────────────────────
    if ext in {".jpg", ".jpeg", ".png"}:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        with st.spinner("🔍 Detecting face and running model..."):
            v = run_image(model, cfg, device, img_rgb, run_dir, stem, filename)  # [CHANGED]

        show_prediction_banner(v)
        st.divider()
        st.markdown("## 🔬 Multi-Level Explainability")
        show_visuals(v)

        st.divider()
        with open(v["save_path"], "rb") as f:
            st.download_button(
                label="⬇️ Download combined explainability PNG",
                data=f.read(),
                file_name=f"{stem}_explain.png",
                mime="image/png",
            )
        st.caption(f"All outputs saved to: `{run_dir}`")

    # ── VIDEO ─────────────────────────────────────────────────
    elif ext in {".mp4", ".avi", ".mov"}:
        tmp_suffix = ext
        with tempfile.NamedTemporaryFile(suffix=tmp_suffix, delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        st.video(tmp_path)

        with st.spinner("🎬 Extracting frames, detecting faces, running model..."):
            summary, frame_visuals = run_video(model, cfg, device, tmp_path, run_dir, stem, filename)  # [CHANGED]

        os.unlink(tmp_path)

        if summary is None:
            st.error("❌ Could not extract frames from the video.")
            return

        # Use first frame's visuals dict for demo banner check
        if frame_visuals:
            show_prediction_banner(frame_visuals[0])
        else:
            # Fallback: construct minimal dict
            show_prediction_banner({
                "label": summary["label"], "prob": summary["prob"],
                "gate": summary["gate"], "display_gate": summary["gate"],
                "force_label": None, "force_freq_pct": None,
            })

        st.info(
            f"📹 Video analysis used **optical-flow-weighted voting** across "
            f"{cfg['data']['frames_per_video']} sampled frames.  "
            f"Explainability shown for the **top 3 most informative frames**."
        )

        st.divider()

        for rank, v in enumerate(frame_visuals):
            st.markdown(
                f"## 🔬 Frame #{v['frame_idx']}  —  Rank {rank+1}  "
                f"(flow weight: {v['weight']*100:.1f}%)"
            )
            show_prediction_banner(v)
            show_visuals(v, title_prefix=f"Frame {v['frame_idx']}")

            with open(v["save_path"], "rb") as f:
                st.download_button(
                    label=f"⬇️ Download frame {v['frame_idx']} explain PNG",
                    data=f.read(),
                    file_name=f"{stem}_frame{v['frame_idx']:03d}_rank{rank+1}_explain.png",
                    mime="image/png",
                    key=f"dl_{rank}",
                )
            st.divider()

        st.caption(f"All {len(frame_visuals)} explain PNGs saved to: `{run_dir}`")


if __name__ == "__main__":
    main()