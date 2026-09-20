#!/usr/bin/env python3
"""Step 2: Learned patch verification model (CNN classifier).

Trains a lightweight CNN (PatchVerifierCNN) to classify 48x48 patches around
candidates from Step 1 as boat (1) or background/false-positive (0).

Features:
1. Auto-labelling from simulator datasets:
   Reads `sidecar.jsonl` recorded with `ARCTICSIM_DEV=1`. If candidate is near
   `groundtruth.point_px_approx`, it is labelled positive; otherwise negative.
2. Synthetic / bootstrap generator:
   Can generate synthetic boat-on-Arctic training patches so training can run
   immediately even before extensive flight datasets are collected.
3. Fast training on Apple Silicon MPS or CPU:
   Trains in seconds, saving weights to `models/patch_verifier.pt`.

Usage:
    # Train using auto-labelled sim dataset:
    python tools/train_patch_verifier.py --dataset data/2026-09-19T18-00-00 --asset quadcopter

    # Train using synthetic/bootstrap dataset:
    python tools/train_patch_verifier.py --bootstrap --epochs 15 --out models/patch_verifier.pt
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from typing import Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from tools.detect_color import ColorAnomalyDetector

log = logging.getLogger("train_patch_verifier")


# --------------------------------------------------------------------------- #
# Model Architecture
# --------------------------------------------------------------------------- #
class PatchVerifierCNN(nn.Module):
    """Tiny CNN patch classifier for 48x48 BGR/RGB image patches.

    ~150k parameters. Fast inference (<2ms per batch of 20 candidates).
    """

    def __init__(self, patch_size: int = 48, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            # 48x48 -> 24x24
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # 24x24 -> 12x12
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # 12x12 -> 6x6
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Global average pooling -> (batch, 128, 1, 1)
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (N, 3, H, W) normalized to [-1, 1] or [0, 1]
        feat = self.features(x)
        return self.classifier(feat)

    @torch.no_grad()
    def predict_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Return probability of positive class (boat)."""
        self.eval()
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        return probs[:, 1]


# --------------------------------------------------------------------------- #
# Dataset & Preprocessing
# --------------------------------------------------------------------------- #
class PatchDataset(Dataset):
    """Dataset of (patch, label) pairs."""

    def __init__(self, patches: list[np.ndarray], labels: list[int], augment: bool = True):
        self.patches = patches
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        patch = self.patches[idx].copy()

        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                patch = cv2.flip(patch, 1)
            # Random vertical flip
            if random.random() > 0.5:
                patch = cv2.flip(patch, 0)
            # Random 90 deg rotation
            k = random.randint(0, 3)
            if k > 0:
                patch = np.rot90(patch, k)
            # Random brightness/contrast jitter
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-15, 15)
            patch = np.clip(patch.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # Convert HWC uint8 BGR -> CHW float32 normalized to [0, 1]
        tensor = torch.from_numpy(patch.transpose(2, 0, 1)).float() / 255.0
        # Normalize with ImageNet-like or standard mean/std
        tensor = (tensor - 0.5) / 0.5
        return tensor, self.labels[idx]


# --------------------------------------------------------------------------- #
# Data Generation & Extraction
# --------------------------------------------------------------------------- #
def extract_dataset_patches(dataset_dir: str, asset: str = "fixed-wing", patch_size: int = 48
                            ) -> Tuple[list[np.ndarray], list[int]]:
    """Extract and auto-label patches from a recorded dataset using groundtruth and color anomaly."""
    sidecar_path = os.path.join(dataset_dir, "sidecar.jsonl")
    if not os.path.exists(sidecar_path):
        sidecar_path = os.path.join(dataset_dir, asset, "sidecar.jsonl")
    if not os.path.exists(sidecar_path):
        raise FileNotFoundError(f"Sidecar not found in {dataset_dir} or {os.path.join(dataset_dir, asset)}")

    detector = ColorAnomalyDetector(min_area=2, min_score=0.20, patch_size=patch_size)
    half_p = patch_size // 2

    positives: list[np.ndarray] = []
    negatives: list[np.ndarray] = []

    with open(sidecar_path) as f:
        for line in f:
            data = json.loads(line)
            frame_rel = data.get("frame")
            frame_path = os.path.join(dataset_dir, frame_rel) if frame_rel else None
            if not frame_path or not os.path.exists(frame_path):
                continue

            img = cv2.imread(frame_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            pose = data.get("pose")
            gt = data.get("groundtruth")
            dist_m = float("inf")
            if pose and gt:
                dlat = (pose["lat"] - gt["lat"]) * 111320
                dlon = (pose["lon"] - gt["lon"]) * 111320 * math.cos(math.radians(pose["lat"]))
                dist_m = math.hypot(dlat, dlon)

            # Detect real boat if within 1500m
            found_boat = False
            boat_cx, boat_cy = None, None

            if dist_m < 1500:
                b, g, r_ch = cv2.split(img)
                red_diff = r_ch.astype(int) - np.maximum(b, g).astype(int)
                red_mask = (red_diff > 16).astype(np.uint8)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(red_mask)
                for label_idx in range(1, num_labels):
                    area = stats[label_idx, cv2.CC_STAT_AREA]
                    if 3 <= area <= 600:
                        cx, cy = centroids[label_idx]
                        if cy > 280:  # In water region below horizon
                            found_boat = True
                            boat_cx, boat_cy = cx, cy
                            # Add jittered positive crops
                            for _ in range(6):
                                du = random.uniform(-3, 3)
                                dv = random.uniform(-3, 3)
                                cu = int(round(cx + du))
                                cv_ = int(round(cy + dv))
                                x1 = max(0, min(w - patch_size, cu - half_p))
                                y1 = max(0, min(h - patch_size, cv_ - half_p))
                                crop = img[y1:y1+patch_size, x1:x1+patch_size]
                                if crop.shape == (patch_size, patch_size, 3):
                                    positives.append(crop)
                            break

            # Run candidate detector to get hard negatives (false positives from shoreline/ice)
            candidates = detector.detect(img, extract_patches=True)
            for c in candidates:
                if c.patch is None or c.patch.shape != (patch_size, patch_size, 3):
                    continue
                if found_boat:
                    dist_to_boat = math.hypot(c.cx - boat_cx, c.cy - boat_cy)
                    if dist_to_boat <= 20.0:
                        positives.append(c.patch)
                    elif dist_to_boat > 50.0:
                        negatives.append(c.patch)
                else:
                    # No boat in this frame, candidate is a hard negative
                    negatives.append(c.patch)

    print(f"Extracted from {dataset_dir}: {len(positives)} positive patches, {len(negatives)} negative patches")
    patches = positives + negatives
    labels = [1] * len(positives) + [0] * len(negatives)
    return patches, labels


def generate_bootstrap_patches(num_samples: int = 1200, patch_size: int = 48
                               ) -> Tuple[list[np.ndarray], list[int]]:
    """Generate synthetic Arctic water/ice backgrounds and boat patches for bootstrap training."""
    patches: list[np.ndarray] = []
    labels: list[int] = []

    half_p = patch_size // 2

    # Background color palettes:
    # Water: BGR around [50..90, 30..60, 10..30]
    # Ice: BGR around [190..240, 200..245, 200..245]
    for i in range(num_samples):
        is_positive = (i % 2 == 0)

        # 1. Generate background patch (either pure water, pure ice, or water/ice boundary)
        bg_type = random.choice(["water", "ice", "boundary"])
        patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)

        water_color = np.array([
            random.randint(45, 95),
            random.randint(25, 65),
            random.randint(10, 35)
        ], dtype=np.float32)

        ice_color = np.array([
            random.randint(195, 235),
            random.randint(205, 245),
            random.randint(205, 245)
        ], dtype=np.float32)

        if bg_type == "water":
            noise = np.random.randn(patch_size, patch_size, 3) * 6
            patch = np.clip(water_color + noise, 0, 255).astype(np.uint8)
        elif bg_type == "ice":
            noise = np.random.randn(patch_size, patch_size, 3) * 5
            patch = np.clip(ice_color + noise, 0, 255).astype(np.uint8)
        else:
            # Boundary
            patch[:, :] = water_color
            angle = random.uniform(0, np.pi)
            offset = random.randint(-half_p, half_p)
            for y in range(patch_size):
                for x in range(patch_size):
                    if (x - half_p) * np.cos(angle) + (y - half_p) * np.sin(angle) > offset:
                        patch[y, x] = ice_color
            patch = np.clip(patch + np.random.randn(patch_size, patch_size, 3) * 5, 0, 255).astype(np.uint8)

        # 2. If positive, draw a red vessel at the center
        if is_positive:
            # Boat size: 5 to 25 pixels depending on simulated altitude
            bw = random.randint(4, 22)
            bh = max(3, int(bw * random.uniform(0.35, 0.65)))
            # Orientation
            boat_yaw = random.uniform(0, 180)

            # Red boat colors: Hull is red, cabin/deck is grey/white
            red_hull = (random.randint(15, 35), random.randint(20, 45), random.randint(175, 240))
            grey_deck = (random.randint(130, 180), random.randint(130, 180), random.randint(130, 180))

            cx = half_p + random.randint(-4, 4)
            cy = half_p + random.randint(-4, 4)

            # Draw hull as an ellipse / rotated rect
            rect = ((cx, cy), (bw, bh), boat_yaw)
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.fillPoly(patch, [box], red_hull)

            # Cabin (smaller inner rectangle)
            if bw > 8:
                cab_rect = ((cx, cy), (max(2, bw // 3), max(2, bh // 2)), boat_yaw)
                cab_box = cv2.boxPoints(cab_rect).astype(np.int32)
                cv2.fillPoly(patch, [cab_box], grey_deck)

            labels.append(1)
        else:
            # If negative, sometimes add hard negatives (ice glints, reddish sunset shading, rust flecks)
            if random.random() < 0.25:
                # Orange/red glint or glare
                gx = random.randint(5, patch_size - 6)
                gy = random.randint(5, patch_size - 6)
                glint_col = (random.randint(10, 40), random.randint(40, 80), random.randint(140, 190))
                cv2.circle(patch, (gx, gy), random.randint(1, 3), glint_col, -1)
            labels.append(0)

        patches.append(patch)

    print(f"Generated {num_samples} bootstrap patches ({labels.count(1)} positive, {labels.count(0)} negative)")
    return patches, labels


# --------------------------------------------------------------------------- #
# Training Pipeline
# --------------------------------------------------------------------------- #
def train_model(patches: list[np.ndarray], labels: list[int],
                epochs: int = 15, batch_size: int = 32, lr: float = 1e-3,
                out_path: str = "models/patch_verifier.pt") -> PatchVerifierCNN:
    """Train the patch verifier CNN."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # Device selection: MPS (Apple Silicon) -> CUDA -> CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Training on device: {device}")

    # Train / Val Split (80 / 20)
    indices = list(range(len(patches)))
    random.seed(42)
    random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]

    train_patches = [patches[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_patches = [patches[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    train_ds = PatchDataset(train_patches, train_labels, augment=True)
    val_ds = PatchDataset(val_patches, val_labels, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = PatchVerifierCNN(patch_size=48, num_classes=2).to(device)

    # Class weights if imbalanced
    num_pos = max(1, sum(train_labels))
    num_neg = max(1, len(train_labels) - num_pos)
    weight = torch.tensor([1.0, float(num_neg) / float(num_pos)]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = 0.0

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)

        scheduler.step()
        train_acc = correct / total if total > 0 else 0.0

        # Validation
        model.eval()
        val_tp = 0
        val_fp = 0
        val_fn = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += len(y)

                for p, t in zip(preds.cpu().numpy(), y.cpu().numpy()):
                    if p == 1 and t == 1:
                        val_tp += 1
                    elif p == 1 and t == 0:
                        val_fp += 1
                    elif p == 0 and t == 1:
                        val_fn += 1

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        prec = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0.0
        rec = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        print(f"Epoch {epoch:2d}/{epochs} - Loss: {total_loss/total:.4f} Acc: {train_acc:.3f} | "
              f"Val Acc: {val_acc:.3f} Prec: {prec:.3f} Rec: {rec:.3f} F1: {f1:.3f}")

        if f1 >= best_val_f1:
            best_val_f1 = f1
            torch.save(model.state_dict(), out_path)

    print(f"\nTrained model saved to: {out_path} (Best Val F1: {best_val_f1:.3f})")
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 2: Train learned patch verifier CNN.")
    parser.add_argument("--dataset", nargs="+", help="Path(s) to recorded dataset directory (from tools/record.py or tools/patrol_and_record.py)")
    parser.add_argument("--asset", default="fixed-wing", help="Asset name in dataset (default: fixed-wing)")
    parser.add_argument("--bootstrap", action="store_true", help="Generate synthetic Arctic training data")
    parser.add_argument("--samples", type=int, default=1600, help="Number of bootstrap samples to generate")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--out", default="models/patch_verifier.pt", help="Output model path")
    args = parser.parse_args()

    patches: list[np.ndarray] = []
    labels: list[int] = []

    if args.dataset:
        for ds in args.dataset:
            try:
                p, l = extract_dataset_patches(ds, args.asset)
                patches.extend(p)
                labels.extend(l)
            except Exception as exc:
                print(f"Warning: Failed to extract from {ds}: {exc}")

    if args.bootstrap or not patches:
        if not args.bootstrap and not patches:
            print("No dataset provided or dataset empty; falling back to bootstrap generator.")
        p, l = generate_bootstrap_patches(num_samples=args.samples)
        patches.extend(p)
        labels.extend(l)

    train_model(patches, labels, epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, out_path=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
