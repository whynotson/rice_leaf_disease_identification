"""
visualize_disease.py
--------------------
Chạy local: Đọc ảnh + file label .txt (YOLO format) hoặc chạy model YOLO đã train,
vẽ bounding box, đếm số lượng từng loại bệnh, lưu ảnh kết quả.

Usage (2 chế độ):
  1. Dùng file label .txt có sẵn (không cần model):
     python visualize_disease.py --image leaf.jpg --label leaf.txt

  2. Chạy model YOLO để infer:
     python visualize_disease.py --image leaf.jpg --model models/yolo/yolo9c.pt

  3. Batch folder:
     python visualize_disease.py --folder data/images/ --model models/yolo/yolo9c.pt
     python visualize_disease.py --folder data/images/ --label_dir data/labels/
"""

import argparse
import sys
import json
from pathlib import Path

# ─── CLASS CONFIG (khớp với config.py của project) ────────────────────────────
CLASSES = [
    "bacterial_leaf_blight",  # 0
    "brown_spot",             # 1
    "healthy",                # 2
    "leaf_blast",             # 3
    "leaf_scald",             # 4
    "sheath_blight",          # 5
]

CLASS_LABELS = {
    "bacterial_leaf_blight": "Bacterial Leaf Blight",
    "brown_spot":            "Brown Spot",
    "healthy":               "Healthy",
    "leaf_blast":            "Leaf Blast",
    "leaf_scald":            "Leaf Scald",
    "sheath_blight":         "Sheath Blight",
}

SEVERITY = {
    "bacterial_leaf_blight": "High",
    "brown_spot":            "Medium",
    "healthy":               "None",
    "leaf_blast":            "Very High",
    "leaf_scald":            "Medium",
    "sheath_blight":         "High",
}

# Màu BGR cho từng class (dùng với OpenCV)
COLORS_BGR = [
    (0,   80,  220),   # bacterial_leaf_blight – đỏ đậm
    (0,  160,  200),   # brown_spot            – cam
    (60, 180,   60),   # healthy               – xanh lá
    (200,  60,  60),   # leaf_blast            – đỏ tươi
    (200, 160,   0),   # leaf_scald            – vàng
    (160,  40, 200),   # sheath_blight         – tím
]

CONF_THRESHOLD = 0.25   # chỉ dùng khi infer bằng model


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def draw_boxes_from_txt(img, txt_path: Path, conf_default=1.0):
    """Đọc file YOLO label .txt, vẽ box lên ảnh (numpy array BGR)."""
    import cv2
    import numpy as np

    h, w = img.shape[:2]
    counts = {c: 0 for c in CLASSES}
    detections = []

    if not txt_path.exists():
        print(f"[WARN] Label file not found: {txt_path}")
        return img, counts, detections

    with open(txt_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        conf = float(parts[5]) if len(parts) >= 6 else conf_default

        if cls_id >= len(CLASSES):
            continue

        class_name = CLASSES[cls_id]
        counts[class_name] += 1

        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        color = COLORS_BGR[cls_id % len(COLORS_BGR)]
        label_text = f"{CLASS_LABELS[class_name]}"
        if conf < 1.0:
            label_text += f" {conf:.2f}"

        # Vẽ box
        thickness = max(2, int(min(w, h) / 400))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Background cho chữ
        font_scale = max(0.45, min(w, h) / 1200)
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        ty = y1 - 6 if y1 - th - 8 >= 0 else y2 + th + 6
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty + baseline), color, -1)
        cv2.putText(img, label_text, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        detections.append({
            "class": class_name,
            "class_id": cls_id,
            "conf": round(conf, 4),
            "bbox_norm": [cx, cy, bw, bh],
            "bbox_pixel": [x1, y1, x2, y2],
        })

    return img, counts, detections


def draw_summary_panel(img, counts, source_path: Path):
    """Vẽ bảng tóm tắt ở góc trên trái."""
    import cv2

    h, w = img.shape[:2]
    panel_w = max(320, w // 4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.45, min(w, h) / 1400)
    lh = int(fs * 38 + 4)

    active = {k: v for k, v in counts.items() if v > 0}
    n_lines = len(active) + 3   # title + filename + blank + classes
    panel_h = lh * (n_lines + 1) + 16

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)

    y = lh
    cv2.putText(img, "Rice Disease Detection", (8, y), font, fs * 1.05,
                (230, 230, 230), 1, cv2.LINE_AA)
    y += lh
    cv2.putText(img, source_path.name, (8, y), font, fs * 0.85,
                (160, 160, 160), 1, cv2.LINE_AA)
    y += int(lh * 0.6)

    total = sum(counts.values())
    cv2.putText(img, f"Total detections: {total}", (8, y + lh), font, fs,
                (200, 230, 200), 1, cv2.LINE_AA)
    y += lh * 2

    for cls_id, (cls, cnt) in enumerate(counts.items()):
        color = COLORS_BGR[cls_id % len(COLORS_BGR)]
        label = f"{CLASS_LABELS[cls]}: {cnt}"
        # Dấu chấm màu
        cy_dot = y - lh // 3
        cv2.circle(img, (12, cy_dot), 6, color, -1)
        cv2.putText(img, label, (26, y), font, fs * 0.9,
                    (220, 220, 220) if cnt > 0 else (90, 90, 90), 1, cv2.LINE_AA)
        y += lh

    return img


def run_yolo_model(image_path: Path, model_path: Path, conf: float):
    """Chạy model YOLO, trả về img (BGR) + counts + detections."""
    try:
        from ultralytics import YOLO
        import cv2
    except ImportError:
        print("[ERROR] Cần cài: pip install ultralytics opencv-python")
        sys.exit(1)

    model = YOLO(str(model_path))
    results = model.predict(str(image_path), conf=conf, verbose=False)[0]
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]

    counts = {c: 0 for c in CLASSES}
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf_val = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        if cls_id >= len(CLASSES):
            continue

        class_name = CLASSES[cls_id]
        counts[class_name] += 1
        color = COLORS_BGR[cls_id % len(COLORS_BGR)]
        label_text = f"{CLASS_LABELS[class_name]} {conf_val:.2f}"

        thickness = max(2, int(min(w, h) / 400))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        font_scale = max(0.45, min(w, h) / 1200)
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        ty = y1 - 6 if y1 - th - 8 >= 0 else y2 + th + 6
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty + baseline), color, -1)
        cv2.putText(img, label_text, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        detections.append({
            "class": class_name,
            "class_id": cls_id,
            "conf": round(conf_val, 4),
            "bbox_norm": [round(v, 6) for v in [cx, cy, bw, bh]],
            "bbox_pixel": [x1, y1, x2, y2],
        })

    return img, counts, detections


def process_single(image_path: Path, output_dir: Path, args):
    import cv2

    image_path = Path(image_path)
    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        return

    if args.model:
        img, counts, detections = run_yolo_model(image_path, Path(args.model), args.conf)
    else:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"[ERROR] Cannot read image: {image_path}")
            return
        txt_path = Path(args.label) if args.label else image_path.with_suffix(".txt")
        img, counts, detections = draw_boxes_from_txt(img, txt_path)

    img = draw_summary_panel(img, counts, image_path)

    # Lưu ảnh output
    output_dir.mkdir(parents=True, exist_ok=True)
    out_img_path = output_dir / f"{image_path.stem}_detected{image_path.suffix}"
    cv2.imwrite(str(out_img_path), img)
    print(f"[OK] Saved: {out_img_path}")

    # Lưu JSON summary
    summary = {
        "image": str(image_path),
        "total_detections": sum(counts.values()),
        "counts": counts,
        "detections": detections,
    }
    out_json = output_dir / f"{image_path.stem}_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Summary: {out_json}")

    # In ra terminal
    print(f"\n{'='*45}")
    print(f"  Image   : {image_path.name}")
    print(f"  Total   : {summary['total_detections']} detections")
    print(f"  ───────────────────────────────────────────")
    for cls, cnt in counts.items():
        if cnt > 0:
            sev = SEVERITY[cls]
            print(f"  {CLASS_LABELS[cls]:<30} {cnt:>3}  [{sev}]")
    print(f"{'='*45}\n")


def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize rice leaf disease bounding boxes from YOLO label or model inference."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image",  type=str, help="Đường dẫn 1 ảnh")
    src.add_argument("--folder", type=str, help="Folder chứa nhiều ảnh (.jpg/.png)")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--model",     type=str, help="Đường dẫn model YOLO (.pt) để infer")
    mode.add_argument("--label",     type=str, help="Đường dẫn file .txt label (chỉ dùng với --image)")
    mode.add_argument("--label_dir", type=str, help="Folder chứa .txt label (chỉ dùng với --folder)")

    p.add_argument("--conf",   type=float, default=CONF_THRESHOLD, help=f"Ngưỡng confidence (default: {CONF_THRESHOLD})")
    p.add_argument("--output", type=str,   default="outputs/visualizations", help="Thư mục lưu kết quả")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output)

    if args.image:
        process_single(Path(args.image), output_dir, args)
    else:
        # Batch folder
        folder = Path(args.folder)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = [p for p in folder.iterdir() if p.suffix.lower() in exts]

        if not images:
            print(f"[ERROR] No images found in {folder}")
            sys.exit(1)

        print(f"[INFO] Processing {len(images)} images...")
        for img_path in sorted(images):
            # Tìm label tương ứng nếu dùng label_dir
            if args.label_dir:
                args.label = str(Path(args.label_dir) / img_path.with_suffix(".txt").name)
            process_single(img_path, output_dir, args)
