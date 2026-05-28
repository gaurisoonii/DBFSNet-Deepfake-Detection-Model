"""
src/dataset.py
==============
FaceForensics++ C40 dataset loader.

CHANGES vs previous version:
  - Added get_balanced_sampler() to FFDataset                    [NEW]
    WeightedRandomSampler fixes 84% fake / 16% real imbalance.
  - Updated get_dataloaders() to use sampler when balance=true   [CHANGED]
  - Stores self.labels list for fast sampler construction        [NEW]
  - Prints class distribution percentages on load               [NEW]

Data layout:
  Real  : data/extracted_faces/real/**/*.png
  Fake  : data/extracted_faces/fake/**/*.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image


# ── Face extractor ────────────────────────────────────────────────────────────

class FaceExtractor:
    def __init__(self, image_size: int = 224, margin: int = 20, device: str = "cpu"):
        self.image_size = image_size
        from facenet_pytorch import MTCNN
        self.mtcnn = MTCNN(image_size=image_size, margin=margin,
                           keep_all=False, post_process=False, device=device)

    def extract_from_video(self, video_path: str, num_frames: int = 30,
                           save_dir: Optional[str] = None) -> List[np.ndarray]:
        cap   = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            return []
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        faces   = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            pil  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            face = self.mtcnn(pil)
            if face is not None:
                face_np = face.permute(1, 2, 0).numpy().astype(np.uint8)
                faces.append(face_np)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    cv2.imwrite(
                        os.path.join(save_dir, f"frame_{idx:05d}.png"),
                        cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
                    )
        cap.release()
        return faces


# ── One-time extraction ───────────────────────────────────────────────────────

def extract_dataset_faces(cfg: dict, device: str = "cpu"):
    """Run once via  python extract_faces.py"""
    extractor  = FaceExtractor(image_size=cfg["data"]["image_size"], device=device)
    num_frames = cfg["data"]["frames_per_video"]
    out_base   = Path(cfg["data"]["extracted_faces_dir"])

    real_dir = Path(cfg["data"]["real_video_dir"])
    print(f"\n[REAL] {real_dir}")
    for vp in sorted(list(real_dir.glob("*.mp4")) + list(real_dir.glob("*.avi"))):
        sd = out_base / "real" / vp.stem
        if sd.exists() and any(sd.iterdir()):
            continue
        print(f"  {vp.name}")
        extractor.extract_from_video(str(vp), num_frames, str(sd))

    for manip, manip_dir in cfg["data"]["fake_video_dirs"].items():
        mp = Path(manip_dir)
        if not mp.exists():
            print(f"[WARN] Not found: {mp}")
            continue
        print(f"\n[FAKE/{manip}] {mp}")
        for vp in sorted(list(mp.glob("*.mp4")) + list(mp.glob("*.avi"))):
            sd = out_base / "fake" / manip / vp.stem
            if sd.exists() and any(sd.iterdir()):
                continue
            print(f"  {vp.name}")
            extractor.extract_from_video(str(vp), num_frames, str(sd))

    print(f"\n[DONE] Faces saved to: {out_base}")


# ── Dataset ───────────────────────────────────────────────────────────────────

class FFDataset(Dataset):
    def __init__(self, extracted_faces_dir: str, split: str = "train",
                 train_ratio: float = 0.8, image_size: int = 224, seed: int = 42):
        self.image_size = image_size
        self.transform  = self._build_transform(split)

        base = Path(extracted_faces_dir)
        real_files = sorted((base / "real").glob("**/*.png")) + \
                     sorted((base / "real").glob("**/*.jpg"))
        fake_files = sorted((base / "fake").glob("**/*.png")) + \
                     sorted((base / "fake").glob("**/*.jpg"))

        if not real_files or not fake_files:
            raise RuntimeError(
                f"No face images found in {base}\n"
                "Run  python extract_faces.py  first!\n"
                f"  real: {len(real_files)}, fake: {len(fake_files)}"
            )

        all_samples = [(str(p), 0) for p in real_files] + \
                      [(str(p), 1) for p in fake_files]
        random.seed(seed)
        random.shuffle(all_samples)

        n = int(len(all_samples) * train_ratio)
        self.samples = all_samples[:n] if split == "train" else all_samples[n:]

        # [NEW] Keep labels list separately for get_balanced_sampler()
        self.labels = [lbl for _, lbl in self.samples]

        r     = sum(1 for l in self.labels if l == 0)
        f     = sum(1 for l in self.labels if l == 1)
        total = len(self.samples)
        print(f"[FFDataset] {split:5s}: {total} samples  "
              f"({r} real [{r/total*100:.1f}%], {f} fake [{f/total*100:.1f}%])")

    def _build_transform(self, split):
        if split == "train":
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        return self.transform(img), label

    # [NEW] ───────────────────────────────────────────────────────────────────
    def get_balanced_sampler(self) -> WeightedRandomSampler:
        """
        Build a WeightedRandomSampler so each CLASS has equal probability per batch.

        Problem without this:
          84% of samples are fake → model sees 5× more fakes per epoch
          → learns to predict "fake" for everything
          → fusion gate stays glued at g≈1 (always trust spatial)
          → real face recall is poor

        Solution:
          weight_for_class_c = 1.0 / count(class_c)
          Each sample gets weight = weight_for_class[its_label]
          Sampler draws num_samples samples proportional to weights
          Result: every batch is ~50% real / ~50% fake
        """
        labels_arr    = np.array(self.labels)
        class_counts  = np.bincount(labels_arr)               # [n_real, n_fake]
        class_weights = 1.0 / class_counts.astype(float)      # [w_real, w_fake]
        sample_weights = class_weights[labels_arr]             # one weight per sample

        print(f"[FFDataset] Balanced sampler:")
        print(f"  Real  count={class_counts[0]:6d}  weight={class_weights[0]:.6f}")
        print(f"  Fake  count={class_counts[1]:6d}  weight={class_weights[1]:.6f}")
        print(f"  → ~50% real / ~50% fake per batch")

        return WeightedRandomSampler(
            weights     = sample_weights,
            num_samples = len(sample_weights),
            replacement = True,   # must be True to oversample minority class
        )


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader).
    If cfg["data"]["balance_dataset"]=True the train loader uses
    WeightedRandomSampler for balanced class distribution.
    Val loader is never balanced (we want real distribution for metrics).
    """
    fd      = cfg["data"]["extracted_faces_dir"]
    bs      = cfg["training"]["batch_size"]
    workers = cfg["data"]["num_workers"]
    balance = cfg["data"].get("balance_dataset", True)   # [NEW]

    train_ds = FFDataset(fd, "train", cfg["data"]["train_split"],
                         cfg["data"]["image_size"], cfg["training"]["seed"])
    val_ds   = FFDataset(fd, "val",   cfg["data"]["train_split"],
                         cfg["data"]["image_size"], cfg["training"]["seed"])

    # [NEW] Build sampler only for training
    sampler = train_ds.get_balanced_sampler() if balance else None

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size  = bs,
        sampler     = sampler,
        shuffle     = (sampler is None),   # shuffle only when no sampler
        num_workers = workers,
        pin_memory  = pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = bs,
        shuffle     = False,
        num_workers = workers,
        pin_memory  = pin,
    )
    return train_loader, val_loader
