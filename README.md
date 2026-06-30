# 🌾 Rice Disease Detection — Pipeline 3 Tầng

Pipeline gồm 3 tầng:

1. **YOLO** — phát hiện vùng bệnh trên lá (detection).
2. **EfficientNet-B4** — phân loại từng crop vùng bệnh (classification).
3. **VLM / Template** — sinh mô tả + khuyến nghị cho từng vùng.

Hệ thống dùng **6 lớp active**, cho phép **đổi YOLO model ngay trên dòng lệnh** (`--model`), và lưu kết quả **riêng theo từng model** để train/evaluate nhiều backbone liên tiếp mà không ghi đè nhau.

---

## 📁 Cấu Trúc Thư Mục

```text
rice_disease_project/
├── configs/
│   └── config.py                     ← Cấu hình tập trung (paths, 6 classes, MODEL_CONFIGS, resolve_yolo_config)
│
├── data/
│   ├── raw_datasets/                 ← Dataset gốc từ Roboflow (1 folder / 1 bệnh)
│   ├── yolo_dataset/                 ← Dataset merge (auto-generated): images/, labels/, data.yaml
│   └── crops/                        ← Ảnh crop cho EfficientNet (auto-generated): train/ val/ test/
│
├── models/
│   ├── yolo/
│   │   ├── yolov9c.pt                
│   │   ├── yolov9e.pt
│   │   ├── yolov10l.pt
│   │   ├── yolov10x.pt
│   └── efficientnet/
│       └── efficientnet_b4_best.pth  ← 1 classifier dùng chung cho mọi YOLO model
│
├── src/
│   ├── utils/
│   │   ├── merge_datasets.py
│   │   └── crop_for_classifier.py
│   ├── tier1_detection/
│   │   ├── train_yolo.py             ← Train YOLO (chọn model bằng --model)
│   │   └── evaluate_yolo.py          ← Evaluate, lưu logs/yolo_eval_<model>.json
│   ├── tier2_classification/
│   │   ├── model.py                  ← EfficientNet-B4 (build_model, unfreeze_all)
│   │   ├── dataset.py                ← Dataset + DataLoader + WeightedSampler
│   │   ├── train_efficientnet.py     ← Train 2-phase (freeze → finetune)
│   │   └── evaluate_efficientnet.py  ← Accuracy, F1, confusion matrix
│   ├── tier3_vlm/
│   │   └── vlm_caption.py            ← BLIP-2 / LLaVA / Template caption
│   └── pipeline/
│       └── pipeline.py               ← END-TO-END: YOLO → EfficientNet → VLM
│
├── outputs/                          ← detections/ crops/ captions/ visualizations/
├── logs/                             ← yolo_eval_<model>.json, efficientnet_history.json, efficientnet_eval_<split>.json
├── requirements.txt
└── README.md
```

---

## ✅ Active Classes (6 lớp)

| ID | Class | Tên tiếng Việt | Mức độ |
|----|-------|----------------|--------|
| 0 | `bacterial_leaf_blight` | Cháy bìa lá vi khuẩn | High |
| 1 | `brown_spot` | Đốm nâu | Medium |
| 2 | `healthy` | Lá khỏe mạnh | None |
| 3 | `leaf_blast` | Đạo ôn lá | Very high |
| 4 | `leaf_scald` | Cháy lá | Medium |
| 5 | `sheath_blight` | Khô vằn bẹ lá | High |

`narrow_brown_spot` và `rice_tungro` đã được loại khỏi cấu hình active.

---

## ⚙️ Config & cơ chế đổi model

`config.py` là nơi cấu hình trung tâm. Việc chọn YOLO model được điều khiển qua `MODEL_CONFIGS` + tham số `--model`:

```python
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
   }
ACTIVE_MODEL = "yolo9c"   # model mặc định khi KHÔNG truyền --model
```

Hàm `resolve_yolo_config(model_key)` suy ra mọi đường dẫn theo model, đảm bảo **không ghi đè**:

| Model | Pretrained | Run dir | Best weights | Eval JSON |
|-------|-----------|---------|--------------|-----------|
| `yolov9c` | `yolov9c.pt` | `rice_yolov9c/` | `models/yolo/yolov9c.pt` | `logs/yolo_eval_yolov9c.json` |
| `yolov9e` | `yolov9e.pt` | `rice_yolov9e/` | `models/yolo/yolov9e.pt` | `logs/yolo_eval_yolov9e.json` |
| `yolov10l` | `yolov10l.pt` | `rice_yolov10l/` | `models/yolo/yolov10l.pt` | `logs/yolo_eval_yolov10l.json` |
| `yolov10x` | `yolov10x.pt` | `rice_yolov10x/` | `models/yolo/yolov10x.pt` | `logs/yolo_eval_yolov10x.json` |

> ⚠️ **Lưu ý tên model:** YOLO26 là bản mới nhất của Ultralytics (biến thể `n / s / m / l / x` → file `yolov10l.pt`, `yolov10x.pt`…). **Không có file `yolov10xml`.** Ultralytics cũng không khuyến nghị YOLO12/YOLO13 cho production, nên project bỏ chúng.

> ℹ️ **EfficientNet độc lập với YOLO:** classifier train trên crop từ **nhãn ground-truth** (qua `crop_for_classifier.py`), không phụ thuộc YOLO model nào. Vì vậy **chỉ cần train EfficientNet 1 lần**, dùng chung cho mọi YOLO. Khi so sánh backbone, bạn chỉ cần train + evaluate lại Tầng 1.

---

## 🚀 Hướng Dẫn Chạy

### 1) Cài đặt
```bash
pip install -r requirements.txt
```

### 2) Đặt dataset vào `data/raw_datasets/`
Mỗi bệnh 1 folder (tên chỉ cần chứa đúng từ khóa class, ví dụ `bacterial leaf blight`, `brown spot`…). Config tự quét folder động nên hậu tố version của Roboflow không sao.

### 3) Merge dataset YOLO
```bash
python src/utils/merge_datasets.py
```
→ Tạo `data/yolo_dataset/` (images, labels, `data.yaml` cho 6 classes).

### 4) Train YOLO (Tầng 1) — chọn model bằng `--model`
```bash
# Mặc định (ACTIVE_MODEL = yolov9e)
python src/tier1_detection/train_yolo.py

# Chọn model cụ thể; epochs/batch tự lấy theo model (100 / 16)
python src/tier1_detection/train_yolo.py --model yolov10x
python src/tier1_detection/train_yolo.py --model yolov9e

# Override nếu cần
python src/tier1_detection/train_yolo.py --model yolov10l --epochs 80 --batch 8 --device 0

# Resume đúng run của model đó
python src/tier1_detection/train_yolo.py --model yolov10x --resume
```
Best checkpoint được copy thành `models/yolo/<model>.pt`, run dir là `models/yolo/rice_<model>/`.

### 5) Evaluate YOLO
```bash
python src/tier1_detection/evaluate_yolo.py --model yolov9c --split val
python src/tier1_detection/evaluate_yolo.py --model yolov10x --split test
```
→ Lưu `logs/yolo_eval_<model>.json` (mỗi model 1 file, không đè nhau).

### 6) Tạo crop dataset cho EfficientNet
```bash
python src/utils/crop_for_classifier.py
```
→ `data/crops/train|val|test`.

### 7) Train EfficientNet (Tầng 2) — chỉ cần 1 lần
```bash
python src/tier2_classification/train_efficientnet.py \
  --train_dir data/crops/train \
  --val_dir data/crops/val
```
Lịch 2 phase: freeze backbone (5 epoch đầu) → unfreeze fine-tune. Lưu best vào `efficientnet_b4_best.pth`, history vào `logs/efficientnet_history.json`.

### 8) Evaluate EfficientNet
```bash
python src/tier2_classification/evaluate_efficientnet.py --val_dir data/crops/val  --split val
python src/tier2_classification/evaluate_efficientnet.py --val_dir data/crops/test --split test
```
→ `logs/efficientnet_eval_<split>.json` (accuracy, F1 macro/weighted, confusion matrix).

### 9) Pipeline end-to-end
```bash
python src/pipeline/pipeline.py --image path/to/leaf.jpg
python src/pipeline/pipeline.py --image path/to/leaf.jpg --vlm_mode blip2
python src/pipeline/pipeline.py --folder path/to/folder --save_json
```
Pipeline dùng `YOLO_WEIGHTS_TRAINED` (theo `ACTIVE_MODEL`) cho Tầng 1. Muốn demo bằng model khác thì đổi `ACTIVE_MODEL` trong config, hoặc truyền `yolo_weights` khi khởi tạo `RiceDiseasePipeline`.

---

## 🔁 Quy trình so sánh nhiều backbone

```bash
# Làm 1 lần cho cả project
python src/utils/merge_datasets.py
python src/utils/crop_for_classifier.py
python src/tier2_classification/train_efficientnet.py --train_dir data/crops/train --val_dir data/crops/val

# Lặp cho từng YOLO model (kết quả tách riêng tự động)
for M in yolov9c yolov9e yolov10l yolov10x; do
  python src/tier1_detection/train_yolo.py    --model $M
  python src/tier1_detection/evaluate_yolo.py --model $M --split test
done
```
So sánh các file `logs/yolo_eval_*.json` để chọn backbone tốt nhất.

---

## 🔧 VLM Options

| Mode | Model | VRAM | Cách chạy |
|------|-------|------|-----------|
| `template` | Rule-based | 0 GB | mặc định |
| `blip2` | BLIP-2 (4-bit) | ~6–8 GB | `--vlm_mode blip2` |
| `llava` | LLaVA-1.5-7B (4-bit) | ~8 GB | `--vlm_mode llava` |

Config mặc định: `VLM_MODEL_ID = "Salesforce/blip2-opt-2.7b"`, `VLM_LOAD_4BIT = True`.

---

## 🐞 Các lỗi đã sửa so với bản trước

- `build_model(...)` thiếu `num_classes` trong `train_efficientnet.py` và `pipeline.py` → đã truyền `num_classes`.
- `model.unfreeze_backbone()` (method không tồn tại) → thay bằng `unfreeze_all(model)` từ `model.py`.
- `train_yolo.py --model` trước chỉ đổi pretrained, gây ghi đè run/best → nay derive toàn bộ qua `resolve_yolo_config`.
- `evaluate_yolo.py` lưu cố định `yolo_eval_results.json` (đè nhau) → nay lưu `yolo_eval_<model>.json`.
- Epoch YOLO: 150 → 100 (có `patience=30` tự early-stop).

> Gợi ý còn lại (không bắt buộc): bạn đang dùng đồng thời `WeightedRandomSampler` **và** `CrossEntropyLoss(weight=...)`. Cả hai cùng cân bằng lớp nên dễ over-correct về lớp hiếm — cân nhắc chỉ giữ một trong hai và ghi rõ lý do khi bảo vệ.
