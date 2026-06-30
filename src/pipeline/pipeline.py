import argparse
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from configs.config import (
    YOLO_WEIGHTS_TRAINED, EFFICIENTNET_WEIGHTS,
    YOLO_CONF_THRESHOLD, YOLO_IOU_THRESHOLD, CROP_PADDING,
    ALL_CLASSES, CLASS_LABELS_VI, SEVERITY_LEVEL,
    OUTPUT_DETECTIONS, OUTPUT_CROPS, OUTPUT_CAPTIONS, OUTPUT_VIZ,
)

from src.tier2_classification.model import build_model
from src.tier2_classification.dataset import get_transforms
from src.tier3_vlm.vlm_caption import template_caption, get_bbox_position, get_captioner

@dataclass
class Detection:
    bbox: List[int]
    yolo_class: int
    yolo_conf: float
    class_name: str = ""
    classify_conf: float = 0.0
    caption: str = ""
    bbox_position: str = ""
    area_ratio: float = 0.0

@dataclass
class PipelineResult:
    image_path: str
    image_size: tuple
    detections: List[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0
    annotated_image_path: str = ""
    caption_path: str = ""

class RiceDiseasePipeline:
    CLASS_COLORS = {
        "bacterial_leaf_blight": (0, 165, 255),
        "brown_spot": (42, 42, 165),
        "healthy": (0, 200, 0),
        "leaf_blast": (0, 0, 220),
        "leaf_scald": (0, 200, 200),
        "sheath_blight": (255, 160, 0),
    }

    def __init__(
        self,
        yolo_weights: str = str(YOLO_WEIGHTS_TRAINED),
        effnet_weights: str = str(EFFICIENTNET_WEIGHTS),
        vlm_mode: str = "template",
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Pipeline device: {self.device}")

        print("Loading YOLO...")
        try:
            from ultralytics import YOLO
            self.yolo = YOLO(yolo_weights)
            print(f"YOLO loaded: {yolo_weights}")
        except Exception as e:
            print(f"Failed to load YOLO: {e}")
            self.yolo = None

        print("Loading EfficientNet...")
        try:
            self.effnet = build_model(
                num_classes=len(ALL_CLASSES),
                pretrained=False,
                weights_path=effnet_weights,
                device=self.device,
            )
            self.effnet.eval()
            self.transform = get_transforms("val")
            print(f"EfficientNet loaded: {effnet_weights}")
        except Exception as e:
            print(f"Failed to load EfficientNet: {e}")
            self.effnet = None

        print(f"Loading VLM: {vlm_mode}")
        self.vlm_mode = vlm_mode
        self.captioner = get_captioner(vlm_mode) if vlm_mode != "template" else None

    def detect(self, image_bgr: np.ndarray) -> List[Dict]:
        if self.yolo is None:
            return []

        results = self.yolo.predict(
            source=image_bgr,
            conf=YOLO_CONF_THRESHOLD,
            iou=YOLO_IOU_THRESHOLD,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "yolo_class": cls_id,
                "yolo_conf": conf,
            })
        return detections

    @torch.no_grad()
    def classify(self, image_bgr: np.ndarray, bbox: List[int]) -> tuple:
        if self.effnet is None:
            return ALL_CLASSES[0], 0.0

        x1, y1, x2, y2 = bbox
        H, W = image_bgr.shape[:2]

        x1p = max(0, x1 - CROP_PADDING)
        y1p = max(0, y1 - CROP_PADDING)
        x2p = min(W, x2 + CROP_PADDING)
        y2p = min(H, y2 + CROP_PADDING)

        crop_bgr = image_bgr[y1p:y2p, x1p:x2p]
        if crop_bgr.size == 0:
            return ALL_CLASSES[0], 0.0

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)
        tensor = self.transform(crop_pil).unsqueeze(0).to(self.device)
        logits = self.effnet(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        conf, pred_idx = probs.max(0)
        return ALL_CLASSES[pred_idx.item()], conf.item()

    def caption(self, image_bgr: np.ndarray, bbox: List[int], class_name: str, conf: float) -> str:
        H, W = image_bgr.shape[:2]
        bbox_pos, area_ratio = get_bbox_position(bbox, W, H)

        if self.vlm_mode == "template" or self.captioner is None:
            return template_caption(class_name, conf, bbox_pos, area_ratio)

        x1, y1, x2, y2 = bbox
        crop = image_bgr[y1:y2, x1:x2]
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        return self.captioner.caption(crop_pil, class_name)

    def visualize(self, image_bgr: np.ndarray, detections: List[Detection]) -> np.ndarray:
        out = image_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = self.CLASS_COLORS.get(det.class_name, (255, 255, 255))
            label_vi = CLASS_LABELS_VI.get(det.class_name, det.class_name)
            severity = SEVERITY_LEVEL.get(det.class_name, "")

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{label_vi} {det.classify_conf * 100:.1f}% [{severity}]"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            ty = max(y1 - th - baseline - 4, 0)
            cv2.rectangle(out, (x1, ty), (x1 + tw + 4, ty + th + baseline + 4), color, -1)

            text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
            cv2.putText(
                out, label, (x1 + 2, ty + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA
            )

        cv2.putText(
            out,
            f"Rice Disease Detection Pipeline - {len(detections)} detection(s)",
            (10, out.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        return out

    def run(self, image_path: str, save_output: bool = True, vlm_caption: bool = True) -> PipelineResult:
        t0 = time.time()
        image_path = Path(image_path)

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        H, W = image_bgr.shape[:2]
        result = PipelineResult(image_path=str(image_path), image_size=(W, H))

        raw_detections = self.detect(image_bgr)
        print(f"YOLO detections: {len(raw_detections)}")

        for raw in raw_detections:
            bbox = raw["bbox"]
            det = Detection(
                bbox=bbox,
                yolo_class=raw["yolo_class"],
                yolo_conf=raw["yolo_conf"],
            )

            class_name, conf = self.classify(image_bgr, bbox)
            det.class_name = class_name
            det.classify_conf = conf
            det.bbox_position, det.area_ratio = get_bbox_position(bbox, W, H)

            label_vi = CLASS_LABELS_VI.get(class_name, class_name)
            print(f"bbox {bbox} -> {label_vi} ({conf * 100:.1f}%)")

            if vlm_caption:
                det.caption = self.caption(image_bgr, bbox, class_name, conf)

            result.detections.append(det)

        result.inference_time_ms = (time.time() - t0) * 1000

        if save_output:
            OUTPUT_VIZ.mkdir(parents=True, exist_ok=True)
            OUTPUT_CAPTIONS.mkdir(parents=True, exist_ok=True)

            annotated = self.visualize(image_bgr, result.detections)
            out_img_path = OUTPUT_VIZ / f"annotated_{image_path.name}"
            cv2.imwrite(str(out_img_path), annotated)
            result.annotated_image_path = str(out_img_path)

            if vlm_caption and result.detections:
                cap_path = OUTPUT_CAPTIONS / f"{image_path.stem}_captions.txt"
                with open(cap_path, "w", encoding="utf-8") as f:
                    f.write(f"File: {image_path.name}\n")
                    f.write(f"Processing time: {result.inference_time_ms:.1f} ms\n")
                    f.write(f"Number of disease regions: {len(result.detections)}\n\n")
                    for i, det in enumerate(result.detections, 1):
                        f.write(f"--- Region {i} ---\n")
                        f.write(f"Bbox: {det.bbox}\n")
                        f.write(f"Class: {CLASS_LABELS_VI.get(det.class_name, det.class_name)}\n")
                        f.write(f"Confidence: {det.classify_conf * 100:.1f}%\n")
                        f.write(f"Caption: {det.caption}\n\n")
                result.caption_path = str(cap_path)

            print(f"Processing time: {result.inference_time_ms:.1f} ms")
            if result.annotated_image_path:
                print(f"Annotated image: {result.annotated_image_path}")
            if result.caption_path:
                print(f"Caption file: {result.caption_path}")

        return result

    def run_folder(self, folder_path: str, save_json: bool = False) -> List[PipelineResult]:
        folder = Path(folder_path)
        image_paths = sorted(
            list(folder.glob("*.jpg")) +
            list(folder.glob("*.jpeg")) +
            list(folder.glob("*.png")) +
            list(folder.glob("*.JPG")) +
            list(folder.glob("*.JPEG")) +
            list(folder.glob("*.PNG"))
        )
        print(f"Processing folder: {folder} ({len(image_paths)} images)")

        results = []
        for i, img_path in enumerate(image_paths, 1):
            print(f"[{i}/{len(image_paths)}] {img_path.name}")
            try:
                r = self.run(str(img_path))
                results.append(r)
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")

        if save_json:
            out_json = OUTPUT_VIZ / "batch_results.json"
            serializable = []
            for r in results:
                serializable.append({
                    "image": r.image_path,
                    "num_detections": len(r.detections),
                    "time_ms": r.inference_time_ms,
                    "detections": [
                        {
                            "class": d.class_name,
                            "class_vi": CLASS_LABELS_VI.get(d.class_name, ""),
                            "conf": round(d.classify_conf, 4),
                            "bbox": d.bbox,
                            "caption": d.caption,
                        }
                        for d in r.detections
                    ],
                })
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)
            print(f"JSON results saved to: {out_json}")

        return results

def parse_args():
    p = argparse.ArgumentParser(description="Rice Disease Pipeline - 3-stage inference")
    p.add_argument("--image", type=str, help="Path to a single image")
    p.add_argument("--folder", type=str, help="Path to an image folder")
    p.add_argument("--vlm_mode", default="template", choices=["template", "blip2", "llava"])
    p.add_argument("--save_json", action="store_true")
    p.add_argument("--no_caption", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument('--model', default='/content/rice_disease_project/models/yolo/yolo9c.pt')
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    pipeline = RiceDiseasePipeline(
        yolo_weights=args.model,
        vlm_mode=args.vlm_mode,
        device=args.device,
)
    if args.image:
        pipeline.run(args.image, vlm_caption=not args.no_caption)
    elif args.folder:
        pipeline.run_folder(args.folder, save_json=args.save_json)
    else:
        print("Provide --image or --folder")
