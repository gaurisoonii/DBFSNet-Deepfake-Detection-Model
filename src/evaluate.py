"""
src/evaluate.py
Do NOT run directly. Run  python evaluate.py  from C:/Users/sanid/Desktop/DBFS
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
    classification_report, roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset import get_dataloaders
from src.model   import DBFSNet


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    all_probs, all_labels, all_gates = [], [], []
    for imgs, labels in tqdm(loader, desc="Evaluating"):
        imgs   = imgs.to(device)
        logits, g = model(imgs)
        all_probs.extend(torch.sigmoid(logits.squeeze(1)).cpu().tolist())
        all_labels.extend(labels.tolist())
        all_gates.extend(g.squeeze(1).cpu().tolist())
    return np.array(all_probs), np.array(all_labels), np.array(all_gates)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.checkpoint or str(Path(cfg["paths"]["checkpoint_dir"]) / "best.pth")

    model = DBFSNet(
        spatial_feat_dim = cfg["model"]["spatial_feat_dim"],
        freq_feat_dim    = cfg["model"]["freq_feat_dim"],
        gate_proj_dim    = cfg["model"]["gate_hidden_dim"],
        gate_hidden_dim  = cfg["model"]["gate_hidden_dim"],
        dropout          = cfg["model"]["dropout"],
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded: {ckpt_path}")

    _, val_loader = get_dataloaders(cfg)
    probs, labels, gates = run_eval(model, val_loader, device)
    preds = (probs > 0.5).astype(int)

    print(f"\nAUC      : {roc_auc_score(labels, probs):.4f}")
    print(f"Accuracy : {accuracy_score(labels, preds)*100:.2f}%")
    print(f"\nConfusion Matrix:\n{confusion_matrix(labels, preds)}")
    print(f"\n{classification_report(labels, preds, target_names=['Real','Fake'])}")
    print(f"Mean gate: {gates.mean():.4f}  (1=spatial, 0=freq)")

    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(labels,probs):.4f}", color="#185FA5", lw=2)
    plt.plot([0,1],[0,1],"k--",lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("DBFSNet ROC — FaceForensics++ C40")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(gates[labels==0], bins=40, alpha=0.6, label="Real", color="#185FA5")
    plt.hist(gates[labels==1], bins=40, alpha=0.6, label="Fake", color="#b00020")
    plt.xlabel("Gate g"); plt.ylabel("Count")
    plt.title("Fusion Gate Distribution"); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "gate_distribution.png", dpi=150); plt.close()
    print(f"\nPlots saved to: {out_dir}")


if __name__ == "__main__":
    main()
