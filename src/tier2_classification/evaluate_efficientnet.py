import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from configs.config import (
    EFFICIENTNET_WEIGHTS, EFFICIENTNET_BATCH,
    ACTIVE_CLASSES, LOG_DIR,
)
from src.tier2_classification.model import build_model
from src.tier2_classification.dataset import RiceDiseaseDataset
from torch.utils.data import DataLoader

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate EfficientNet-B4 classifier")
    p.add_argument("--val_dir", type=str, required=True)
    p.add_argument("--weights", type=str, default=str(EFFICIENTNET_WEIGHTS))
    p.add_argument("--batch", type=int, default=EFFICIENTNET_BATCH)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--split", type=str, default="val", choices=["val", "test"])
    return p.parse_args()

@torch.no_grad()
def evaluate(args):
    try:
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
    except ImportError:
        print("scikit-learn is not installed. Run: pip install scikit-learn")
        sys.exit(1)

    device = torch.device(args.device)

    dataset = RiceDiseaseDataset(
        root_dir=args.val_dir,
        split=args.split,
        class_names=ACTIVE_CLASSES,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"{args.split} set: {len(dataset)} images | {len(ACTIVE_CLASSES)} classes")

    model = build_model(
        num_classes=len(ACTIVE_CLASSES),
        pretrained=False,
        weights_path=args.weights,
        device=str(device),
    )
    model.eval()

    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    f1_mac = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_wei = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=ACTIVE_CLASSES, zero_division=0)

    print("\nEVALUATION RESULTS")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Macro: {f1_mac:.4f}")
    print(f"F1 Weighted: {f1_wei:.4f}")
    print("\nCLASSIFICATION REPORT:")
    print(report)

    print("CONFUSION MATRIX:")
    header = f"{'':>25}" + "".join(f"{c[:8]:>10}" for c in ACTIVE_CLASSES)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{ACTIVE_CLASSES[i][:24]:>25}" + "".join(f"{v:>10}" for v in row)
        print(row_str)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "split": args.split,
        "weights": args.weights,
        "accuracy": float(acc),
        "f1_macro": float(f1_mac),
        "f1_weighted": float(f1_wei),
        "per_class": {},
        "confusion_matrix": cm.tolist(),
    }

    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    for i, cls in enumerate(ACTIVE_CLASSES):
        result["per_class"][cls] = {
            "f1": float(per_class_f1[i]),
            "support": int((all_labels == i).sum()),
        }

    out_path = LOG_DIR / f"efficientnet_eval_{args.split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {out_path}")
    return result

if __name__ == "__main__":
    evaluate(parse_args())