"""
generate_augmentation_figure.py
Tạo figure minh họa augmentation cho thesis — chạy độc lập, không cần import project.

Usage:
    python generate_augmentation_figure.py \
        --image path/to/image.jpg \
        --output path/to/augmentation_figure.png
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Augmentation functions (khớp đúng params thesis) ─────────────────────────

def aug_original(img):
    return img.copy()

def aug_fliplr(img):
    return cv2.flip(img, 1)

def aug_flipud(img):
    return cv2.flip(img, 0)

def aug_rotation(img, angle=10):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def aug_translation(img, ratio=0.10):
    h, w = img.shape[:2]
    M = np.float32([[1, 0, int(w * ratio)], [0, 1, int(h * ratio * 0.5)]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def aug_scale(img, factor=0.75):
    h, w = img.shape[:2]
    nh, nw = int(h * factor), int(w * factor)
    resized = cv2.resize(img, (nw, nh))
    pad_top    = (h - nh) // 2
    pad_bottom = h - nh - pad_top
    pad_left   = (w - nw) // 2
    pad_right  = w - nw - pad_left
    return cv2.copyMakeBorder(resized, pad_top, pad_bottom,
                              pad_left, pad_right, cv2.BORDER_REFLECT)

def aug_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] + 0.015 * 180 * np.random.uniform(-1, 1), 0, 179)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.3, 1.7), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * np.random.uniform(0.6, 1.4), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

def aug_mosaic(img):
    h, w = img.shape[:2]
    hh, hw = h // 2, w // 2
    tiles = [
        img.copy(),
        aug_fliplr(img),
        aug_rotation(img, 8),
        aug_hsv(img),
    ]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:hh, :hw]  = cv2.resize(tiles[0], (hw, hh))
    canvas[:hh, hw:]  = cv2.resize(tiles[1], (w - hw, hh))
    canvas[hh:, :hw]  = cv2.resize(tiles[2], (hw, h - hh))
    canvas[hh:, hw:]  = cv2.resize(tiles[3], (w - hw, h - hh))
    cv2.line(canvas, (hw, 0), (hw, h), (200, 200, 200), 1)
    cv2.line(canvas, (0, hh), (w, hh), (200, 200, 200), 1)
    return canvas

def aug_random_crop(img, size=380):
    big = cv2.resize(img, (int(img.shape[1] * 1.18), int(img.shape[0] * 1.18)))
    bh, bw = big.shape[:2]
    y0 = random.randint(0, max(0, bh - size))
    x0 = random.randint(0, max(0, bw - size))
    crop = big[y0:y0 + size, x0:x0 + size]
    return cv2.resize(crop, (size, size))

def aug_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


# ── Panel definitions ─────────────────────────────────────────────────────────

YOLO_PANELS = [
    ("Original",                     aug_original),
    ("Horizontal Flip\n(p = 0.50)",  aug_fliplr),
    ("Vertical Flip\n(p = 0.10)",    aug_flipud),
    ("Rotation\n(± 10°)",            lambda i: aug_rotation(i, 10)),
    ("Translation\n(0.10)",          aug_translation),
    ("Scaling\n(0.50)",              aug_scale),
    ("HSV Jitter\n(H=0.015 S=0.70 V=0.40)", aug_hsv),
    ("Mosaic\n(p = 1.00)",           aug_mosaic),
]

EFF_PANELS = [
    ("Original\n(380 × 380)",        lambda i: cv2.resize(i, (380, 380))),
    ("Random Crop\n(380 × 380)",     aug_random_crop),
    ("Horizontal Flip",              aug_fliplr),
    ("Vertical Flip",                aug_flipud),
    ("Color Jitter\n(HSV)",          aug_hsv),
    ("Grayscale\n(occasional)",      aug_grayscale),
]


# ── Draw helper ───────────────────────────────────────────────────────────────

def make_figure(src_rgb, panels, title, nrows, ncols, out_path: Path, dpi=150):
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.8, nrows * 3.6))
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.015)

    for i, ax in enumerate(axes.flat):
        if i < len(panels):
            label, fn = panels[i]
            out_img = fn(src_rgb.copy())
            ax.imshow(out_img)
            ax.set_title(label, fontsize=9.5, pad=6, linespacing=1.45)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#cccccc")
                spine.set_linewidth(0.7)
        else:
            ax.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(pad=0.9, w_pad=0.6, h_pad=1.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"[OK] Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate augmentation figure for thesis."
    )
    parser.add_argument("--image",  type=str, required=True,
                        help="Path to a sample leaf image (.jpg/.png)")
    parser.add_argument("--output", type=str,
                        default="outputs/visualizations/augmentation_figure.png",
                        help="Output path for the combined figure")
    parser.add_argument("--dpi",    type=int, default=150)
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"[ERROR] Image not found: {img_path}")
        sys.exit(1)

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"[ERROR] Cannot read image: {img_path}")
        sys.exit(1)

    src_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Resize to square 380×380 as base
    src_rgb = cv2.resize(src_rgb, (380, 380))
    print(f"[INFO] Loaded: {img_path.name}  →  {src_rgb.shape}")

    out_base = Path(args.output)

    # Figure (a): YOLO — 2 rows × 4 cols
    make_figure(
        src_rgb, YOLO_PANELS,
        "(a) YOLO Detector — Training Augmentation Examples",
        nrows=2, ncols=4,
        out_path=out_base.parent / (out_base.stem + "_yolo" + out_base.suffix),
        dpi=args.dpi,
    )

    # Figure (b): EfficientNet — 2 rows × 3 cols
    make_figure(
        src_rgb, EFF_PANELS,
        "(b) EfficientNet-B4 Classifier — Training Augmentation Examples",
        nrows=2, ncols=3,
        out_path=out_base.parent / (out_base.stem + "_efficientnet" + out_base.suffix),
        dpi=args.dpi,
    )

    print("[DONE] Both figures saved.")


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    main()
