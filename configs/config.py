import os
from pathlib import Path

def _find_project_root() -> Path:
    if os.environ.get("RICE_PROJECT_ROOT"):
        return Path(os.environ["RICE_PROJECT_ROOT"]).resolve()
    return Path(__file__).resolve().parent.parent

ROOT_DIR = _find_project_root()

def _find_raw_dataset(raw_dir: Path, disease_key: str) -> Path | None:
    if not raw_dir.exists():
        return None

    keywords = disease_key.lower().replace("_", " ").split()
    for folder in raw_dir.iterdir():
        if not folder.is_dir():
            continue
        folder_norm = (
            folder.name.lower()
            .replace("_", " ")
            .replace("-", " ")
            .replace(".", " ")
        )
        if all(kw in folder_norm for kw in keywords):
            return folder
    return None

RAW_DATASETS_DIR = ROOT_DIR / "data" / "raw_datasets"

ALL_CLASSES = [
    "bacterial_leaf_blight",#0
    "brown_spot",           #1
    "healthy",              #2
    "leaf_blast",           #3
    "leaf_scald",           #4  
    "sheath_blight"         #5
]

ACTIVE_CLASSES = list(ALL_CLASSES)
_DISEASE_KEYS = ACTIVE_CLASSES

RAW_DATASETS: dict[str, Path] = {}
for _key in _DISEASE_KEYS:
    _found = _find_raw_dataset(RAW_DATASETS_DIR, _key)
    RAW_DATASETS[_key] = _found if _found else RAW_DATASETS_DIR / f"{_key}.yolov8"

YOLO_DATASET_DIR = ROOT_DIR / "data" / "yolo_dataset"
YOLO_IMAGES_TRAIN = YOLO_DATASET_DIR / "images" / "train"
YOLO_IMAGES_VAL = YOLO_DATASET_DIR / "images" / "val"
YOLO_IMAGES_TEST = YOLO_DATASET_DIR / "images" / "test"
YOLO_LABELS_TRAIN = YOLO_DATASET_DIR / "labels" / "train"
YOLO_LABELS_VAL = YOLO_DATASET_DIR / "labels" / "val"
YOLO_LABELS_TEST = YOLO_DATASET_DIR / "labels" / "test"
YOLO_DATA_YAML = YOLO_DATASET_DIR / "data.yaml"

MODEL_CONFIGS = {
    # ----- YOLOv9 -----
    "yolo9c": {
        "weights": "yolov9c.pt",
        "imgsz": 640,
        "batch": 16,
        "epochs": 100,
        "patience": 30,
        "run_name": "rice_yolo9c",
    },
    "yolo9e": {  # v9 extreme
        "weights": "yolov9e.pt",
        "imgsz": 640,
        "batch": 16,
        "epochs": 100,
        "patience": 30,
        "run_name": "rice_yolo9e",
    },

    # ----- YOLOv10 -----
    "yolo10l": {  # v10 large
        "weights": "yolov10l.pt",
        "imgsz": 640,
        "batch": 16,
        "epochs": 100,
        "patience": 30,
        "run_name": "rice_yolo10l",
    },
    "yolo10x": {  # v10 extreme
        "weights": "yolov10x.pt",
        "imgsz": 640,
        "batch": 16,
        "epochs": 100,
        "patience": 30,
        "run_name": "rice_yolo10x",
    },

    # ----- YOLOv11 -----
    "yolo11l": {  # v11 large
        "weights": "yolo11l.pt",      # lưu ý: v11 bỏ chữ "v" trong tên file
        "imgsz": 640,
        "batch": 16,
        "epochs": 100,
        "patience": 30,
        "run_name": "rice_yolo11l",
    },
    "yolo11x": {  # v11 extreme
        "weights": "yolo11x.pt",
        "imgsz": 640,
        "batch": 16,
        "epochs": 100,
        "patience": 30,
        "run_name": "rice_yolo11x",
    },
}

# Model mặc định khi KHÔNG truyền --model
ACTIVE_MODEL = "yolo9c" #e
_cfg = MODEL_CONFIGS[ACTIVE_MODEL]

# Các biến module-level dưới đây là DEFAULT (theo ACTIVE_MODEL).
# Giữ lại để tương thích ngược với các script chưa nhận --model.
YOLO_WEIGHTS_PRETRAIN = _cfg["weights"]
YOLO_IMGSZ = _cfg["imgsz"]
YOLO_BATCH = _cfg["batch"]
YOLO_EPOCHS = _cfg["epochs"]
YOLO_PATIENCE = _cfg["patience"]
YOLO_RUN_NAME = _cfg["run_name"]

YOLO_WEIGHTS_TRAINED = ROOT_DIR / "models" / "yolo" / f"{ACTIVE_MODEL}.pt"
YOLO_PROJECT_DIR = ROOT_DIR / "models" / "yolo"
YOLO_WORKERS = 4
YOLO_DEVICE = "0"

EFFICIENTNET_WEIGHTS = ROOT_DIR / "models" / "efficientnet" / "efficientnet_b4_best.pth"
EFFICIENTNET_NUM_CLASSES = len(ACTIVE_CLASSES)
EFFICIENTNET_IMG_SIZE = 380
EFFICIENTNET_BATCH = 32
EFFICIENTNET_EPOCHS = 30
EFFICIENTNET_LR = 1e-4
EFFICIENTNET_DROPOUT1 = 0.4
EFFICIENTNET_DROPOUT2 = 0.3
EFFICIENTNET_FREEZE_EPOCHS = 5

VLM_MODEL_ID = "Salesforce/blip2-opt-2.7b"
VLM_DEVICE_MAP = "auto"
VLM_LOAD_4BIT = True

YOLO_CONF_THRESHOLD = 0.25
YOLO_IOU_THRESHOLD = 0.45
CROP_PADDING = 10

OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DETECTIONS = OUTPUT_DIR / "detections"
OUTPUT_CROPS = OUTPUT_DIR / "crops"
OUTPUT_CAPTIONS = OUTPUT_DIR / "captions"
OUTPUT_VIZ = OUTPUT_DIR / "visualizations"
LOG_DIR = ROOT_DIR / "logs"

CLASS_LABELS_VI = {
    "bacterial_leaf_blight": "Bacterial leaf blight",
    "brown_spot": "Brown spot",
    "healthy": "Healthy",
    "leaf_blast": "Leaf blast",
    "leaf_scald": "Leaf scald",
    "sheath_blight": "Sheath blight",
}

SEVERITY_LEVEL = {
    "bacterial_leaf_blight": "High",
    "brown_spot": "Medium",
    "healthy": "None",
    "leaf_blast": "Very high",
    "leaf_scald": "Medium",
    "sheath_blight": "High",
}

def resolve_yolo_config(model_key: str | None = None) -> dict:
    """Suy ra toàn bộ đường dẫn/tham số cho 1 model YOLO theo key.

    Dùng trong train_yolo.py / evaluate_yolo.py để mỗi model có:
      - run dir riêng (rice_<model>)
      - file best weights riêng (models/yolo/<model>.pt)
      - file eval json riêng (logs/yolo_eval_<model>.json)
    => Chạy nhiều model liên tiếp không bị ghi đè lên nhau.
    """
    model_key = model_key or ACTIVE_MODEL
    if model_key not in MODEL_CONFIGS:
        raise ValueError(
            f"Model '{model_key}' không có trong MODEL_CONFIGS. "
            f"Chọn một trong: {list(MODEL_CONFIGS.keys())}"
        )
    cfg = MODEL_CONFIGS[model_key]
    project_dir = ROOT_DIR / "models" / "yolo"
    return {
        "model_key": model_key,
        "pretrain": cfg["weights"],
        "imgsz": cfg["imgsz"],
        "batch": cfg["batch"],
        "epochs": cfg["epochs"],
        "patience": cfg["patience"],
        "run_name": cfg["run_name"],
        "project_dir": project_dir,
        "trained_weights": project_dir / f"{model_key}.pt",
        "eval_json": LOG_DIR / f"yolo_eval_{model_key}.json",
    }

def print_config():
    print(f"ROOT_DIR: {ROOT_DIR}")
    print(f"RAW_DATASETS_DIR: {RAW_DATASETS_DIR}")
    print(f"ACTIVE_MODEL: {ACTIVE_MODEL}")
    print(f"YOLO_WEIGHTS: {YOLO_WEIGHTS_PRETRAIN}")
    print(f"ACTIVE_CLASSES: {ACTIVE_CLASSES}")
    print(f"Num classes: {len(ACTIVE_CLASSES)}")
    print("RAW_DATASETS (resolved):")
    for k, v in RAW_DATASETS.items():
        status = "OK" if v.exists() else "NOT FOUND"
        print(f"  {k:<30} -> {v.name} {status}")
    print(f"YOLO_DATASET_DIR: {YOLO_DATASET_DIR}")

if __name__ == "__main__":
    print_config()