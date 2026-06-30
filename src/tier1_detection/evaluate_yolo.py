import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from configs.config import (
    ACTIVE_MODEL, MODEL_CONFIGS, resolve_yolo_config,
    YOLO_DATA_YAML, YOLO_DEVICE,
    ALL_CLASSES, LOG_DIR,
)

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate YOLO for rice disease detection")
    p.add_argument(
        "--model",
        type=str,
        default=ACTIVE_MODEL,
        choices=list(MODEL_CONFIGS.keys()),
        help="Key model trong MODEL_CONFIGS: yolov8m / yolo11m / yolo26m / yolo26x",
    )
    p.add_argument("--weights", type=str, default=None, help="Override đường dẫn weights (mặc định lấy theo --model)")
    p.add_argument("--split", type=str, default="val", choices=["val", "test"])
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    return p.parse_args()

def evaluate(args):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics is not installed. Run: pip install ultralytics")
        sys.exit(1)

    ycfg = resolve_yolo_config(args.model)
    weights = args.weights if args.weights is not None else str(ycfg["trained_weights"])

    print("=" * 60)
    print(f"Model: {weights}")

    model = YOLO(weights)
    metrics = model.val(
        data=str(YOLO_DATA_YAML),
        imgsz=ycfg["imgsz"],
        batch=16,
        conf=args.conf,
        iou=args.iou,
        device=YOLO_DEVICE,
        split=args.split,
        plots=True,
        save_json=True,
    )

    print("\nEVALUATION RESULTS")
    print(f"mAP@0.50: {metrics.box.map50:.4f}")
    print(f"mAP@0.50:0.95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")

    print("\nPER-CLASS")
    print(f"{'Class':<30} {'P':>8} {'R':>8} {'mAP50':>8} {'mAP50-95':>10}")
    print("-" * 68)

    names = metrics.names if hasattr(metrics, "names") else {}
    num_cls = len(metrics.box.maps)

    for i in range(num_cls):
        cls_name = names[i] if isinstance(names, dict) and i in names else f"class_{i}"
        p = float(metrics.box.p[i]) if hasattr(metrics.box, "p") and len(metrics.box.p) > i else 0.0
        r = float(metrics.box.r[i]) if hasattr(metrics.box, "r") and len(metrics.box.r) > i else 0.0
        ap50 = float(metrics.box.ap50[i]) if hasattr(metrics.box, "ap50") and len(metrics.box.ap50) > i else 0.0
        map95 = float(metrics.box.maps[i]) if len(metrics.box.maps) > i else 0.0
        print(f"{cls_name:<30} {p:>8.4f} {r:>8.4f} {ap50:>8.4f} {map95:>10.4f}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    result_path = ycfg["eval_json"]
    result = {
        "model": ycfg["model_key"],
        "split": args.split,
        "weights": weights,
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {result_path}")

if __name__ == "__main__":
    evaluate(parse_args())
