import sys
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from configs.config import YOLO_DATASET_DIR, ALL_CLASSES, ROOT_DIR, CROP_PADDING

CROP_OUTPUT_DIR = ROOT_DIR / "data" / "crops"

def yolo_bbox_to_pixel(cx, cy, bw, bh, img_w, img_h):
    x1 = int((cx - bw / 2) * img_w)
    y1 = int((cy - bh / 2) * img_h)
    x2 = int((cx + bw / 2) * img_w)
    y2 = int((cy + bh / 2) * img_h)
    return x1, y1, x2, y2

def crop_split(split: str):
    images_dir = YOLO_DATASET_DIR / "images" / split
    labels_dir = YOLO_DATASET_DIR / "labels" / split
    out_dir = CROP_OUTPUT_DIR / split

    if not images_dir.exists():
        print(f"Missing directory: {images_dir}")
        return

    count = 0
    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        H, W = img.shape[:2]
        with open(label_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) < 5:
                continue

            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            x1, y1, x2, y2 = yolo_bbox_to_pixel(cx, cy, bw, bh, W, H)

            x1p = max(0, x1 - CROP_PADDING)
            y1p = max(0, y1 - CROP_PADDING)
            x2p = min(W, x2 + CROP_PADDING)
            y2p = min(H, y2 + CROP_PADDING)

            crop = img[y1p:y2p, x1p:x2p]
            if crop.size == 0:
                continue

            cls_name = ALL_CLASSES[cls_id] if cls_id < len(ALL_CLASSES) else f"class_{cls_id}"
            save_dir = out_dir / cls_name
            save_dir.mkdir(parents=True, exist_ok=True)

            save_path = save_dir / f"{img_path.stem}_crop{i:02d}.jpg"
            cv2.imwrite(str(save_path), crop)
            count += 1

    print(f"[{split}] {count} crops -> {out_dir}")

def main():
    print("=" * 60)
    print("CROPPING DISEASE REGIONS FROM YOLO LABELS")
    print("=" * 60)
    for split in ["train", "val", "test"]:
        crop_split(split)
    print(f"Crop output directory: {CROP_OUTPUT_DIR}")
    print(f"Use for training: python src/tier2_classification/train_efficientnet.py --train_dir {CROP_OUTPUT_DIR}/train --val_dir {CROP_OUTPUT_DIR}/val")

if __name__ == "__main__":
    main()