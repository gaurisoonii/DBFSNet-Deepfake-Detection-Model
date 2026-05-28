"""
src/train.py — actual training logic.
Optimized for high-end CPU/GPU systems.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
import yaml
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.model   import DBFSNet

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def compute_accuracy(logits, labels):
    preds = (torch.sigmoid(logits.squeeze(1)) > 0.5).long()
    return (preds == labels).float().mean().item()

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, grad_clip):
    model.train()
    total_loss = total_acc = n = 0
    
    # Modern AMP detection
    use_amp = (device.type == 'cuda')
    
    for imgs, labels in tqdm(loader, desc="   train", leave=False):
        imgs   = imgs.to(device)
        labels = labels.float().to(device)
        optimizer.zero_grad()
        
        # Use modern torch.amp API
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits, _ = model(imgs)
            loss = criterion(logits.squeeze(1), labels)
        
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard path for CPU (i9-14900K handles this well)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_acc  += compute_accuracy(logits, labels.long()) * bs
        n += bs
    return total_loss / n, total_acc / n

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = total_acc = n = 0
    all_probs, all_labels = [], []
    
    for imgs, labels in tqdm(loader, desc="   val   ", leave=False):
        imgs     = imgs.to(device)
        labels_f = labels.float().to(device)
        
        # No autocast needed for validation usually, but keep device consistent
        logits, _ = model(imgs)
        loss = criterion(logits.squeeze(1), labels_f)
        
        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_acc  += compute_accuracy(logits, labels.to(device)) * bs
        n += bs
        all_probs.append(torch.sigmoid(logits.squeeze(1)).cpu())
        all_labels.append(labels)
        
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(torch.cat(all_labels).numpy(),
                        torch.cat(all_probs).numpy())
    return total_loss / n, total_acc / n, auc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument("--resume",  default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    # Intelligent device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running on: {device} ---")
    if device.type == 'cpu':
        print("Note: No GPU detected. Utilizing i9-14900K CPU cores.")

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["output_dir"]).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = get_dataloaders(cfg)

    model = DBFSNet(
        spatial_feat_dim = cfg["model"]["spatial_feat_dim"],
        freq_feat_dim    = cfg["model"]["freq_feat_dim"],
        gate_proj_dim    = cfg["model"]["gate_hidden_dim"],
        gate_hidden_dim  = cfg["model"]["gate_hidden_dim"],
        dropout          = cfg["model"]["dropout"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(),
                      lr=cfg["training"]["learning_rate"],
                      weight_decay=cfg["training"]["weight_decay"])

    warmup = cfg["training"]["warmup_epochs"]
    total  = cfg["training"]["epochs"]
    
    # Schedulers
    scheduler_warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=total - warmup, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup])

    # Scaler enabled only if CUDA is available
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    start_epoch = 0
    best_auc     = 0.0

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_auc     = ckpt.get("best_auc", 0.0)
        print(f"Resumed from epoch {ckpt['epoch']}  best AUC={best_auc:.4f}")

    print(f"\n{'='*55}")
    print(f"   DBFSNet Training — {total} epochs")
    print(f"{'='*55}\n")

    for epoch in range(start_epoch, total):
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1:03d}/{total}]  lr={lr:.2e}")

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, cfg["training"]["grad_clip"])
        
        va_loss, va_acc, va_auc = validate(model, val_loader, criterion, device)
        
        scheduler.step()

        print(f"   train  loss={tr_loss:.4f}  acc={tr_acc*100:.1f}%")
        print(f"   val    loss={va_loss:.4f}  acc={va_acc*100:.1f}%  AUC={va_auc:.4f}")

        # Save Latest
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_auc": best_auc, "cfg": cfg},
                    ckpt_dir / "latest.pth")

        # Save Best
        if va_auc > best_auc:
            best_auc = va_auc
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "best_auc": best_auc, "cfg": cfg},
                        ckpt_dir / "best.pth")
            print(f"   ★ New best AUC: {best_auc:.4f}  saved -> best.pth")

    print(f"\nDone. Best AUC = {best_auc:.4f}")

if __name__ == "__main__":
    main()