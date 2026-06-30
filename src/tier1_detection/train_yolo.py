import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from configs.config import (
    ACTIVE_MODEL, MODEL_CONFIGS, resolve_yolo_config,
    YOLO_DATA_YAML, YOLO_WORKERS, YOLO_DEVICE,
    ALL_CLASSES, ACTIVE_CLASSES,
)

def parse_args():
    p = argparse.ArgumentParser(description="Train YOLO for rice disease detection")
    p.add_argument(
        "--model",
        type=str,
        default=ACTIVE_MODEL,
        help="Key model trong MODEL_CONFIGS: yolov8m / yolo11m / yolo26m / yolo26x",
    )
    p.add_argument("--epochs", type=int, default=None, help="Override epochs (mặc định lấy theo model)")
    p.add_argument("--imgsz", type=int, default=None, help="Override imgsz (mặc định lấy theo model)")
    p.add_argument("--batch", type=int, default=None, help="Override batch (mặc định lấy theo model)")
    p.add_argument("--device", type=str, default=YOLO_DEVICE)
    p.add_argument("--resume", action="store_true", help="Resume from the last checkpoint")
    return p.parse_args()

def train(args):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics is not installed. Run: pip install ultralytics")
        sys.exit(1)

    ycfg = resolve_yolo_config(args.model)
    epochs = args.epochs if args.epochs is not None else ycfg["epochs"]
    imgsz = args.imgsz if args.imgsz is not None else ycfg["imgsz"]
    batch = args.batch if args.batch is not None else ycfg["batch"]

    print("=" * 60)
    print(f"TRAIN YOLO - model = {ycfg['model_key']}")
    print(f"Pretrained: {ycfg['pretrain']}")
    print(f"Data: {YOLO_DATA_YAML}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print(f"Device: {args.device}")
    print(f"Run name: {ycfg['run_name']}")
    print(f"Best weights -> {ycfg['trained_weights']}")
    print(f"Classes: {len(ACTIVE_CLASSES)} active / {len(ALL_CLASSES)} total")
    print("=" * 60)

    if not Path(YOLO_DATA_YAML).exists():
        print(f"Missing file: {YOLO_DATA_YAML}")
        print("Run: python src/utils/merge_datasets.py first")
        sys.exit(1)

    last_ckpt = ycfg["project_dir"] / ycfg["run_name"] / "weights" / "last.pt"
    if args.resume and last_ckpt.exists():
        print(f"Resuming from {last_ckpt}")
        model = YOLO(str(last_ckpt))
    else:
        model = YOLO(ycfg["pretrain"])

    results = model.train(
        data=str(YOLO_DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=args.device,
        workers=YOLO_WORKERS,
        patience=ycfg["patience"],
        project=str(ycfg["project_dir"]),
        name=ycfg["run_name"],
        resume=args.resume,
        mosaic=1.0,
        flipud=0.1,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        copy_paste=0.0,
        mixup=0.0,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
    )

    best_src = ycfg["project_dir"] / ycfg["run_name"] / "weights" / "best.pt"
    if best_src.exists():
        ycfg["trained_weights"].parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_src, ycfg["trained_weights"])
        print(f"Best weights saved to {ycfg['trained_weights']}")
    else:
        print(f"Best checkpoint not found: {best_src}")

    print("Training complete")
    print(f"Run directory: {ycfg['project_dir'] / ycfg['run_name']}")
    return results

if __name__ == "__main__":
    train(parse_args())
