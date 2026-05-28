"""
src/evaluate.py  — Full evaluation with visual plots.
Run via:  python evaluate.py   (from C:/Users/sanid/Desktop/DBFS)

CHANGES vs previous version:
  - Full visual evaluation dashboard saved as single PNG            [NEW]
  - Plots: Confusion Matrix heatmap, ROC Curve, Precision-Recall,
           Fake Probability Distribution, Gate Distribution,
           Class Prediction Breakdown (bar chart)                  [NEW]
  - Prints all text metrics to console (unchanged)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    average_precision_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.dataset import get_dataloaders
from src.model   import DBFSNet


# ── Load config ───────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Inference pass ────────────────────────────────────────────────────────────

@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    all_probs, all_labels, all_gates = [], [], []
    for imgs, labels in tqdm(loader, desc="Evaluating"):
        imgs      = imgs.to(device)
        logits, g = model(imgs)
        all_probs.extend(torch.sigmoid(logits.squeeze(1)).cpu().tolist())
        all_labels.extend(labels.tolist())
        all_gates.extend(g.squeeze(1).cpu().tolist())
    return (
        np.array(all_probs),
        np.array(all_labels, dtype=int),
        np.array(all_gates),
    )


# ── Visual dashboard ──────────────────────────────────────────────────────────

def plot_evaluation_dashboard(probs, labels, gates, out_dir: Path):
    """
    Save a 6-panel evaluation dashboard as outputs/evaluation_dashboard.png

    Panels:
      1. Confusion Matrix heatmap (actual counts + normalised %)
      2. ROC Curve with AUC
      3. Precision-Recall Curve with AP
      4. Fake Probability Distribution (real vs fake histogram)
      5. Fusion Gate Distribution (real vs fake histogram)
      6. Per-class Prediction Breakdown (TP/FP/TN/FN bar chart)
    """
    preds = (probs > 0.5).astype(int)
    auc   = roc_auc_score(labels, probs)
    acc   = accuracy_score(labels, preds)
    cm    = confusion_matrix(labels, preds)
    ap    = average_precision_score(labels, probs)

    # Colour palette
    BG, PANEL = "#0e1117", "#1a1a2e"
    BLUE, RED  = "#3B82F6", "#EF4444"
    GREEN, GOLD = "#10B981", "#F59E0B"
    PURPLE, TEAL = "#7C3AED", "#06B6D4"
    WHITE, GRAY  = "#F9FAFB", "#9CA3AF"

    fig = plt.figure(figsize=(20, 13), facecolor=BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.38, wspace=0.32,
                            left=0.06, right=0.97, top=0.91, bottom=0.07)

    def styled_ax(pos):
        ax = fig.add_subplot(pos)
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_color("#444466")
        ax.tick_params(colors=WHITE)
        ax.xaxis.label.set_color(WHITE)
        ax.yaxis.label.set_color(WHITE)
        ax.title.set_color(WHITE)
        return ax

    # ── Panel 1: Confusion Matrix ─────────────────────────────
    ax1 = styled_ax(gs[0, 0])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax1.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax1, fraction=0.046)
    tn, fp, fn, tp = cm.ravel()
    for r in range(2):
        for c in range(2):
            val_raw  = cm[r, c]
            val_pct  = cm_norm[r, c] * 100
            color    = "white" if cm_norm[r, c] > 0.5 else "black"
            ax1.text(c, r, f"{val_raw}\n({val_pct:.1f}%)",
                     ha="center", va="center", color=color,
                     fontsize=14, fontweight="bold")
    ax1.set_xticks([0, 1]); ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Real", "Fake"], color=WHITE, fontsize=11)
    ax1.set_yticklabels(["Real", "Fake"], color=WHITE, fontsize=11)
    ax1.set_xlabel("Predicted", fontsize=12); ax1.set_ylabel("Actual", fontsize=12)
    ax1.set_title(f"Confusion Matrix  (Acc={acc*100:.2f}%)", fontsize=12, pad=10)

    # ── Panel 2: ROC Curve ────────────────────────────────────
    ax2 = styled_ax(gs[0, 1])
    fpr, tpr, _ = roc_curve(labels, probs)
    ax2.plot(fpr, tpr, color=BLUE, lw=2.2, label=f"DBFSNet (AUC = {auc:.4f})")
    ax2.plot([0, 1], [0, 1], "--", color=GRAY, lw=1.2, label="Random baseline")
    ax2.fill_between(fpr, tpr, alpha=0.12, color=BLUE)
    ax2.set_xlabel("False Positive Rate", fontsize=11)
    ax2.set_ylabel("True Positive Rate", fontsize=11)
    ax2.set_title("ROC Curve", fontsize=12, pad=10)
    ax2.legend(fontsize=10, facecolor=PANEL, edgecolor="#444466", labelcolor=WHITE)
    ax2.grid(True, alpha=0.15, color=GRAY)
    ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1.02])

    # ── Panel 3: Precision-Recall Curve ──────────────────────
    ax3 = styled_ax(gs[0, 2])
    precision, recall, _ = precision_recall_curve(labels, probs)
    ax3.plot(recall, precision, color=PURPLE, lw=2.2, label=f"AP = {ap:.4f}")
    ax3.fill_between(recall, precision, alpha=0.12, color=PURPLE)
    ax3.set_xlabel("Recall", fontsize=11)
    ax3.set_ylabel("Precision", fontsize=11)
    ax3.set_title("Precision-Recall Curve", fontsize=12, pad=10)
    ax3.legend(fontsize=10, facecolor=PANEL, edgecolor="#444466", labelcolor=WHITE)
    ax3.grid(True, alpha=0.15, color=GRAY)
    ax3.set_xlim([0, 1]); ax3.set_ylim([0, 1.02])

    # ── Panel 4: Probability Distribution ────────────────────
    ax4 = styled_ax(gs[1, 0])
    bins = np.linspace(0, 1, 50)
    ax4.hist(probs[labels == 0], bins=bins, alpha=0.65, color=GREEN,
             label=f"Real  (n={int((labels==0).sum())})", edgecolor="none")
    ax4.hist(probs[labels == 1], bins=bins, alpha=0.65, color=RED,
             label=f"Fake  (n={int((labels==1).sum())})", edgecolor="none")
    ax4.axvline(0.5, color=GOLD, lw=1.8, ls="--", label="Threshold = 0.5")
    ax4.set_xlabel("Predicted Fake Probability", fontsize=11)
    ax4.set_ylabel("Count", fontsize=11)
    ax4.set_title("Fake Probability Distribution", fontsize=12, pad=10)
    ax4.legend(fontsize=10, facecolor=PANEL, edgecolor="#444466", labelcolor=WHITE)
    ax4.grid(True, alpha=0.15, color=GRAY)

    # ── Panel 5: Gate Distribution ────────────────────────────
    ax5 = styled_ax(gs[1, 1])
    ax5.hist(gates[labels == 0], bins=40, alpha=0.65, color=BLUE,
             label=f"Real faces  mean={gates[labels==0].mean():.3f}", edgecolor="none")
    ax5.hist(gates[labels == 1], bins=40, alpha=0.65, color=RED,
             label=f"Fake faces  mean={gates[labels==1].mean():.3f}", edgecolor="none")
    ax5.axvline(gates.mean(), color=GOLD, lw=1.8, ls="--",
                label=f"Overall mean = {gates.mean():.3f}")
    ax5.set_xlabel("Fusion Gate Value  g  (1=spatial, 0=freq)", fontsize=11)
    ax5.set_ylabel("Count", fontsize=11)
    ax5.set_title("Fusion Gate Distribution", fontsize=12, pad=10)
    ax5.legend(fontsize=10, facecolor=PANEL, edgecolor="#444466", labelcolor=WHITE)
    ax5.grid(True, alpha=0.15, color=GRAY)

    # ── Panel 6: TP/FP/TN/FN Breakdown ───────────────────────
    ax6 = styled_ax(gs[1, 2])
    categories = ["True Neg\n(Real→Real)", "False Pos\n(Real→Fake)",
                  "False Neg\n(Fake→Real)", "True Pos\n(Fake→Fake)"]
    values  = [tn, fp, fn, tp]
    colors_ = [GREEN, RED, RED, GREEN]
    bars = ax6.bar(categories, values, color=colors_, edgecolor="#1a1a2e",
                   linewidth=1.5, width=0.6)
    for bar, val in zip(bars, values):
        ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                 f"{val:,}", ha="center", va="bottom", color=WHITE,
                 fontsize=11, fontweight="bold")
    ax6.set_ylabel("Count", fontsize=11)
    ax6.set_title("Prediction Breakdown  (TP / FP / TN / FN)", fontsize=12, pad=10)
    ax6.tick_params(axis="x", labelsize=9, colors=WHITE)
    ax6.grid(True, alpha=0.15, color=GRAY, axis="y")

    # ── Super title ───────────────────────────────────────────
    fig.suptitle(
        f"DBFSNet Evaluation  ·  FaceForensics++ C40  "
        f"·  AUC={auc:.4f}  Acc={acc*100:.2f}%  AP={ap:.4f}",
        fontsize=15, fontweight="bold", color=WHITE, y=0.97,
    )

    save_path = out_dir / "evaluation_dashboard.png"
    plt.savefig(save_path, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"\n[Evaluate] Dashboard saved -> {save_path}")
    return save_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.checkpoint or str(
        Path(cfg["paths"]["checkpoint_dir"]) / "best.pth"
    )

    model = DBFSNet(
        spatial_feat_dim = cfg["model"]["spatial_feat_dim"],
        freq_feat_dim    = cfg["model"]["freq_feat_dim"],
        gate_proj_dim    = cfg["model"]["gate_hidden_dim"],
        gate_hidden_dim  = cfg["model"]["gate_hidden_dim"],
        dropout          = cfg["model"]["dropout"],
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded: {ckpt_path}")

    # Val loader always unbalanced to reflect real class distribution
    _, val_loader = get_dataloaders(cfg)
    probs, labels, gates = run_eval(model, val_loader, device)
    preds = (probs > 0.5).astype(int)

    # ── Text metrics (console) ────────────────────────────────
    print(f"\nAUC      : {roc_auc_score(labels, probs):.4f}")
    print(f"Accuracy : {accuracy_score(labels, preds)*100:.2f}%")
    print(f"\nConfusion Matrix:\n{confusion_matrix(labels, preds)}")
    print(f"\n{classification_report(labels, preds, target_names=['Real', 'Fake'])}")
    print(f"Mean gate: {gates.mean():.4f}  (1=spatial, 0=freq)")

    # ── Visual dashboard ──────────────────────────────────────
    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_evaluation_dashboard(probs, labels, gates, out_dir)

    print(f"\nNext step → run:  python plot_metrics.py")


if __name__ == "__main__":
    main()
