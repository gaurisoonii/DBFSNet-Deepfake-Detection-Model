   """
extract_faces.py
Run this ONCE before training to extract face crops from all raw videos.

Run from C:/Users/sanid/Desktop/DBFS :
    python extract_faces.py

This will create:
    C:/Users/sanid/Desktop/DBFS/data/extracted_faces/real/...
    C:/Users/sanid/Desktop/DBFS/data/extracted_faces/fake/...
"""

import yaml
import torch
from src.dataset import extract_dataset_faces

CONFIG_PATH = "configs/config.yaml"

if __name__ == "__main__":
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Config loaded from: {CONFIG_PATH}\n")
    print("Starting face extraction — this may take a while for large datasets...")

    extract_dataset_faces(cfg, device=device)
