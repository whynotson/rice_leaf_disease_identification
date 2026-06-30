import argparse
import sys
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from configs.config import (
    EFFICIENTNET_EPOCHS, EFFICIENTNET_LR, EFFICIENTNET_BATCH,
    EFFICIENTNET_NUM_CLASSES, EFFICIENTNET_FREEZE_EPOCHS,
    EFFICIENTNET_WEIGHTS, ALL_CLASSES, LOG_DIR
)
from src.tier2_classification.model import build_model, unfreeze_all
from src.tier2_classification.dataset import build_dataloaders


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir",  required=True)
    p.add_argument("--val_dir",    required=True)
    p.add_argument("--epochs",     type=int,   default=EFFICIENTNET_EPOCHS)
    p.add_argument("--batch",      type=int,   default=EFFICIENTNET_BATCH)
    p.add_argument("--lr",         type=float, default=EFFICIENTNET_LR)
    p.add_argument("--device",     type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--patience",   type=int,   default=7, help="Early stopping patience")
    return p.parse_args()


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


def train(args):
    print("=" * 60)
    print("🚀 TẦNG 2: TRAIN EFFICIENTNET-B4 CLASSIFIER")
    print("=" * 60)
    print(f"  Device: {args.device}")
    print(f"  Epochs: {args.epochs} (freeze {EFFICIENTNET_FREEZE_EPOCHS} + finetune {args.epochs - EFFICIENTNET_FREEZE_EPOCHS})")
    print(f"  LR: {args.lr}")

    device = torch.device(args.device)

    # ── Data ──────────────────────────────────────────────────
    loaders = build_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch,
    )

    # ── Model (Phase 1: freeze backbone) ─────────────────────
    model = build_model(
        num_classes=EFFICIENTNET_NUM_CLASSES,
        pretrained=True,
        freeze_backbone=True,
        device=str(device),
    )

    # Tính class weights cho loss
    class_weights = loaders["train"].dataset.get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Phase 1 optimizer: chỉ train classifier head
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr * 10,  # LR cao hơn cho warm-up
        weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EFFICIENTNET_FREEZE_EPOCHS)

    best_val_acc = 0.0
    best_epoch   = 0
    patience_cnt = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Phase 2: unfreeze backbone sau freeze_epochs ──────
        if epoch == EFFICIENTNET_FREEZE_EPOCHS + 1:
            print(f"\n🔓 Epoch {epoch}: Unfreeze backbone — bắt đầu fine-tune toàn bộ")
            model = unfreeze_all(model)
            optimizer = optim.Adam(
                model.parameters(),
                lr=args.lr,
                weight_decay=1e-4
            )
            scheduler = CosineAnnealingLR(
                optimizer, T_max=args.epochs - EFFICIENTNET_FREEZE_EPOCHS
            )

        # Train + Val
        train_loss, train_acc = train_epoch(model, loaders["train"], criterion, optimizer, device)
        val_loss,   val_acc   = val_epoch(model, loaders["val"], criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"Train {train_loss:.4f}/{train_acc:.4f} | "
              f"Val {val_loss:.4f}/{val_acc:.4f} | "
              f"LR {scheduler.get_last_lr()[0]:.2e} | "
              f"{elapsed:.1f}s")

        # Lưu best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_cnt = 0
            EFFICIENTNET_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), EFFICIENTNET_WEIGHTS)
            print(f"  💾 Best model saved! val_acc={val_acc:.4f}")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"\n⏹️  Early stopping tại epoch {epoch} (best: {best_epoch})")
                break

    # Lưu history
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / "efficientnet_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✅ Train hoàn tất! Best val_acc = {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"   Weights: {EFFICIENTNET_WEIGHTS}")
    return history


if __name__ == "__main__":
    train(parse_args())
