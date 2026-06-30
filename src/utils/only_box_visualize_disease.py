"""
visualize_disease.py
Vẽ bounding box + số thứ tự nhỏ ở góc trên phải trong box + Total góc ảnh.

Usage:
  python src/utils/visualize_disease.py --image leaf.jpg --label leaf.txt
  python src/utils/visualize_disease.py --image leaf.jpg --model models/yolo/yolo9c.pt
  python src/utils/visualize_disease.py --folder data/yolo_dataset/images/test/ --label_dir data/yolo_dataset/labels/test/
"""

import argparse
import sys
from pathlib import Path

CLASSES = [
    "bacterial_leaf_blight",
    "brown_spot",
    "healthy",
    "leaf_blast",
    "leaf_scald",
    "sheath_blight",
]

COLORS_BGR = [
    (0,   80, 220),
    (0,  165, 255),
    (60, 180,  60),
    (60,  60, 220),
    (0,  200, 200),
    (200, 60, 200),
]

CONF_THRESHOLD = 0.25


def draw(image_path: Path, detections: list, output_dir: Path):
    """detections: list of (cls_id, x1, y1, x2, y2)"""
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERROR] Cannot read: {image_path}")
        return

    h, w = img.shape[:2]
    thickness = max(1, int(min(w, h) / 600))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.3, min(w, h) / 1800)

    for idx, (cls_id, x1, y1, x2, y2) in enumerate(detections, start=1):
        color = COLORS_BGR[cls_id % len(COLORS_BGR)]

        # Bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Số thứ tự nhỏ ở góc trên phải trong box
        num_text = str(idx)
        (tw, th), _ = cv2.getTextSize(num_text, font, font_scale, 1)
        # Vị trí: góc trên phải trong box, cách cạnh 2px
        tx = x2 - tw - 2
        ty = y1 + th + 2
        # Nền nhỏ cho dễ đọc
        cv2.rectangle(img, (tx - 1, y1 + 1), (x2 - 1, y1 + th + 4), color, -1)
        cv2.putText(img, num_text, (tx, ty), font, font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)

    # Total ở góc trên phải ảnh
    total_text = f"Total: {len(detections)}"
    (tw, th), _ = cv2.getTextSize(total_text, font, font_scale * 1.4, 2)
    cv2.rectangle(img, (w - tw - 14, 4), (w - 4, th + 14), (20, 20, 20), -1)
    cv2.putText(img, total_text, (w - tw - 10, th + 10),
                font, font_scale * 1.4, (230, 230, 230), 2, cv2.LINE_AA)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{image_path.stem}_detected{image_path.suffix}"
    cv2.imwrite(str(out_path), img)
    print(f"[OK] {image_path.name} → {len(detections)} detections → {out_path}")


def from_txt(image_path: Path, txt_path: Path, output_dir: Path):
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERROR] Cannot read: {image_path}"); return
    h, w = img.shape[:2]

    detections = []
    if not txt_path.exists():
        print(f"[WARN] Label not found: {txt_path}")
    else:
        for line in txt_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = max(0, int((cx - bw / 2) * w))
            y1 = max(0, int((cy - bh / 2) * h))
            x2 = min(w - 1, int((cx + bw / 2) * w))
            y2 = min(h - 1, int((cy + bh / 2) * h))
            detections.append((cls_id, x1, y1, x2, y2))

    draw(image_path, detections, output_dir)


def from_model(image_path: Path, model_path: Path, conf: float, output_dir: Path):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] pip install ultralytics"); sys.exit(1)

    model = YOLO(str(model_path))
    results = model.predict(str(image_path), conf=conf, verbose=False)[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        detections.append((cls_id, x1, y1, x2, y2))

    draw(image_path, detections, output_dir)


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image",  type=str)
    src.add_argument("--folder", type=str)

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--model",     type=str)
    mode.add_argument("--label",     type=str)
    mode.add_argument("--label_dir", type=str)

    p.add_argument("--conf",   type=float, default=CONF_THRESHOLD)
    p.add_argument("--output", type=str,   default="outputs/visualizations")
    args = p.parse_args()

    output_dir = Path(args.output)

    if args.image:
        img_path = Path(args.image)
        if args.model:
            from_model(img_path, Path(args.model), args.conf, output_dir)
        else:
            txt = Path(args.label) if args.label else img_path.with_suffix(".txt")
            from_txt(img_path, txt, output_dir)
    else:
        folder = Path(args.folder)
        images = sorted(p for p in folder.iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        print(f"[INFO] {len(images)} images found")
        for img_path in images:
            if args.model:
                from_model(img_path, Path(args.model), args.conf, output_dir)
            else:
                ldir = Path(args.label_dir) if args.label_dir else folder
                from_txt(img_path, ldir / img_path.with_suffix(".txt").name, output_dir)


if __name__ == "__main__":
    main()