"""
plot_metrics.py  — Visualize training curves from checkpoints/training_metrics.csv
Run via:  python plot_metrics.py   (from C:/Users/sanid/Desktop/DBFS)

Reads:  checkpoints/training_metrics.csv  (written by train.py each epoch)
Saves:  outputs/training_curves.png

6 Panels:
  1. Train Loss vs Val Loss
  2. Train Accuracy vs Val Accuracy
  3. Validation AUC (with best-epoch star)
  4. Learning Rate Schedule
  5. Generalisation Gap (overfitting monitor)
  6. Summary metrics table
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import csv
import yaml
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_metrics(csv_path: str) -> dict:
    if not Path(csv_path).exists():
        print(f"\n❌  Not found: {csv_path}")
        print("   Run  python train.py  first.\n")
        sys.exit(1)
    data = {"epoch":[], "train_loss":[], "train_acc":[],
            "val_loss":[], "val_acc":[], "val_auc":[], "lr":[]}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            data["epoch"].append(int(row["epoch"]))
            data["train_loss"].append(float(row["train_loss"]))
            data["train_acc"].append(float(row["train_acc"]))
            data["val_loss"].append(float(row["val_loss"]))
            data["val_acc"].append(float(row["val_acc"]))
            data["val_auc"].append(float(row["val_auc"]))
            data["lr"].append(float(row["lr"]))
    return {k: np.array(v) for k, v in data.items()}


def plot_training_curves(csv_path: str, output_path: str):
    d  = load_metrics(csv_path)
    ep = d["epoch"]
    n  = len(ep)
    bi = int(np.argmax(d["val_auc"]))
    be = ep[bi]

    BG, PANEL = "#0e1117", "#1a1a2e"
    BLUE, RED = "#3B82F6", "#EF4444"
    GREEN, GOLD = "#10B981", "#F59E0B"
    PURPLE, TEAL = "#7C3AED", "#06B6D4"
    WHITE, GRAY = "#F9FAFB", "#9CA3AF"

    fig = plt.figure(figsize=(20, 13), facecolor=BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32,
                            left=0.06, right=0.97, top=0.91, bottom=0.07)

    def sax(pos):
        ax = fig.add_subplot(pos)
        ax.set_facecolor(PANEL)
        for s in ax.spines.values(): s.set_color("#444466")
        ax.tick_params(colors=WHITE)
        ax.xaxis.label.set_color(WHITE)
        ax.yaxis.label.set_color(WHITE)
        ax.title.set_color(WHITE)
        ax.grid(True, alpha=0.15, color=GRAY)
        ax.set_xlim([ep[0], ep[-1]])
        return ax

    def best_line(ax):
        ax.axvline(be, color=GOLD, lw=1.5, ls=":", alpha=0.7, label=f"Best epoch ({be})")

    # Panel 1 — Loss
    ax = sax(gs[0,0])
    ax.plot(ep, d["train_loss"], color=BLUE, lw=2, marker="o", ms=3.5, label="Train Loss")
    ax.plot(ep, d["val_loss"],   color=RED,  lw=2, marker="s", ms=3.5, label="Val Loss")
    best_line(ax)
    ax.set_xlabel("Epoch", fontsize=12); ax.set_ylabel("BCE Loss", fontsize=12)
    ax.set_title("Training & Validation Loss", fontsize=13, pad=10)
    ax.legend(fontsize=10, facecolor=PANEL, edgecolor="#444466", labelcolor=WHITE)

    # Panel 2 — Accuracy
    ax = sax(gs[0,1])
    ax.plot(ep, d["train_acc"]*100, color=GREEN, lw=2, marker="o", ms=3.5, label="Train Acc")
    ax.plot(ep, d["val_acc"]*100,   color=GOLD,  lw=2, marker="s", ms=3.5, label="Val Acc")
    best_line(ax)
    ax.set_xlabel("Epoch", fontsize=12); ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Training & Validation Accuracy", fontsize=13, pad=10)
    ax.legend(fontsize=10, facecolor=PANEL, edgecolor="#444466", labelcolor=WHITE)

    # Panel 3 — AUC
    ax = sax(gs[0,2])
    ax.plot(ep, d["val_auc"], color=PURPLE, lw=2, marker="^", ms=4, label="Val AUC")
    ax.scatter([be], [d["val_auc"][bi]], color=GOLD, s=220, marker="*",
               edgecolors=WHITE, linewidths=1.5, zorder=6,
               label=f"Best = {d['val_auc'][bi]:.4f} @ ep{be}")
    ax.set_ylim([max(0.5, d["val_auc"].min()-0.02), 1.01])
    ax.set_xlabel("Epoch", fontsize=12); ax.set_ylabel("AUC", fontsize=12)
    ax.set_title("Validation AUC", fontsize=13, pad=10)
    ax.legend(fontsize=10, facecolor=PANEL, edgecolor="#444466", labelcolor=WHITE)

    # Panel 4 — LR
    ax = sax(gs[1,0])
    ax.plot(ep, d["lr"], color=TEAL, lw=2, marker="d", ms=3.5, label="Learning Rate")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontsize=12); ax.set_ylabel("LR", fontsize=12)
    ax.set_title("LR Schedule (Warmup → Cosine)", fontsize=13, pad=10)
    ax.legend(fontsize=10, facecolor=PANEL, edgecolor="#444466", labelcolor=WHITE)

    # Panel 5 — Generalisation gap
    ax = sax(gs[1,1])
    ax.plot(ep, d["val_loss"]-d["train_loss"],       color=RED,  lw=2, label="Val Loss − Train Loss")
    ax.plot(ep, (d["train_acc"]-d["val_acc"])*100,   color=BLUE, lw=2, ls="--", label="Train Acc − Val Acc (%)")
    ax.axhline(0, color=GRAY, lw=1.2, ls=":")
    best_line(ax)
    ax.set_xlabel("Epoch", fontsize=12); ax.set_ylabel("Gap", fontsize=12)
    ax.set_title("Generalisation Gap  (↓ = less overfitting)", fontsize=13, pad=10)
    ax.legend(fontsize=10, facecolor=PANEL, edgecolor="#444466", labelcolor=WHITE)

    # Panel 6 — Summary table
    ax6 = fig.add_subplot(gs[1,2])
    ax6.set_facecolor(PANEL); ax6.axis("off")
    rows = [
        ["Metric",      "Best Epoch",                          "Final Epoch"],
        ["Epoch",       str(be),                               str(ep[-1])],
        ["Train Loss",  f"{d['train_loss'][bi]:.4f}",          f"{d['train_loss'][-1]:.4f}"],
        ["Val Loss",    f"{d['val_loss'][bi]:.4f}",            f"{d['val_loss'][-1]:.4f}"],
        ["Train Acc",   f"{d['train_acc'][bi]*100:.2f}%",      f"{d['train_acc'][-1]*100:.2f}%"],
        ["Val Acc",     f"{d['val_acc'][bi]*100:.2f}%",        f"{d['val_acc'][-1]*100:.2f}%"],
        ["Val AUC",     f"{d['val_auc'][bi]:.4f}",             f"{d['val_auc'][-1]:.4f}"],
        ["LR",          f"{d['lr'][bi]:.2e}",                  f"{d['lr'][-1]:.2e}"],
    ]
    tbl = ax6.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 2.0)
    for j in range(3):
        c = tbl[0,j]; c.set_facecolor("#2a1f60"); c.set_text_props(color=WHITE, fontweight="bold")
    for i in range(1, len(rows)):
        for j in range(3):
            c = tbl[i,j]
            c.set_facecolor("#2a2000" if j==1 else "#001a30" if j==2 else PANEL)
            c.set_text_props(color=WHITE)
    ax6.set_title("Summary", fontsize=13, color=WHITE, pad=16)

    fig.suptitle(
        f"DBFSNet Training Curves  ·  {n} Epochs  ·  "
        f"Best AUC = {d['val_auc'][bi]:.4f}  ·  Best Val Acc = {d['val_acc'][bi]*100:.2f}%",
        fontsize=15, fontweight="bold", color=WHITE, y=0.97,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"\n✅  Training curves saved -> {output_path}")
    print(f"\n── Summary ──────────────────────────────────")
    print(f"  Total epochs    : {n}")
    print(f"  Best val AUC    : {d['val_auc'][bi]:.4f}  (epoch {be})")
    print(f"  Best val Acc    : {d['val_acc'][bi]*100:.2f}%")
    print(f"  Final train Acc : {d['train_acc'][-1]*100:.2f}%")
    print(f"  Final val Acc   : {d['val_acc'][-1]*100:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    csv_path = cfg["paths"].get("metrics_csv",
        str(Path(cfg["paths"]["checkpoint_dir"]) / "training_metrics.csv"))
    out_path = str(Path(cfg["paths"]["output_dir"]) / "training_curves.png")
    print(f"CSV  : {csv_path}")
    print(f"Save : {out_path}\n")
    plot_training_curves(csv_path, out_path)

if __name__ == "__main__":
    main()
