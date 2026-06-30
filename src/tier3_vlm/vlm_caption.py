import argparse
import sys
from pathlib import Path
from typing import Optional

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from configs.config import (
    VLM_MODEL_ID, VLM_DEVICE_MAP, VLM_LOAD_4BIT,
    CLASS_LABELS_VI, SEVERITY_LEVEL, ALL_CLASSES,
)

# CLASS_LABELS_VI in config already maps to English strings
# (e.g. "Bacterial leaf blight"), so it is used here as the display label.

# ── Treatment recommendations (English) ───────────────────────
RECOMMENDATION = {
    "bacterial_leaf_blight": "Remove heavily infected leaves and apply an appropriate bactericide under expert guidance.",
    "brown_spot": "Improve field nutrition and manage moisture stress to reduce disease pressure.",
    "healthy": "No treatment needed.",
    "leaf_blast": "Monitor closely and apply a fungicide if conditions favor spread.",
    "leaf_scald": "Maintain field hygiene and reduce excessive humidity where possible.",
    "sheath_blight": "Remove infected debris and consider fungicide management if severity increases.",
}


def template_caption(disease_name: str, conf: float, bbox_pos: str, area_ratio: float) -> str:
    """Rule-based caption (English only, no model required)."""
    label = CLASS_LABELS_VI.get(disease_name, disease_name)
    severity = SEVERITY_LEVEL.get(disease_name, "Unknown")
    rec = RECOMMENDATION.get(disease_name, "No recommendation available.")
    return (
        f"Detected disease: {label}. "
        f"Confidence: {conf * 100:.1f}%. "
        f"Location: {bbox_pos}. "
        f"Approximate area ratio: {area_ratio:.3f}. "
        f"Severity: {severity}. "
        f"Recommendation: {rec}"
    )


def get_bbox_position(bbox, img_w, img_h):
    """Return a human-readable position label (English) and the area ratio."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0 / max(img_w, 1)
    cy = (y1 + y2) / 2.0 / max(img_h, 1)
    area_ratio = ((x2 - x1) * (y2 - y1)) / max(img_w * img_h, 1)

    if cx < 0.33:
        x_pos = "left"
    elif cx > 0.66:
        x_pos = "right"
    else:
        x_pos = "center"

    if cy < 0.33:
        y_pos = "top"
    elif cy > 0.66:
        y_pos = "bottom"
    else:
        y_pos = "middle"

    return f"{y_pos}-{x_pos}", float(area_ratio)


# ── BLIP-2 Caption ────────────────────────────────────────────
class BLIP2Captioner:
    """
    BLIP-2 VLM — runs on GPU.
    Requires: pip install transformers accelerate bitsandbytes
    VRAM: ~6-8 GB with 4-bit quantization.
    """

    def __init__(
        self,
        model_id: str = VLM_MODEL_ID,
        load_4bit: bool = VLM_LOAD_4BIT,
        device_map: str = VLM_DEVICE_MAP,
    ):
        print(f"Loading BLIP-2 ({model_id})...")
        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            import torch
        except ImportError:
            raise ImportError("pip install transformers accelerate bitsandbytes")

        self.processor = Blip2Processor.from_pretrained(model_id)

        load_kwargs = {"device_map": device_map}
        if load_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            load_kwargs["torch_dtype"] = torch.float16

        self.model = Blip2ForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
        print("BLIP-2 loaded")

    def caption(
        self,
        image: Image.Image,
        disease_name: str,
        prompt: str = None,
    ) -> str:
        """Generate an English caption from BLIP-2 plus label/severity/recommendation."""
        if prompt is None:
            disease_en = disease_name.replace("_", " ").title()
            prompt = (
                f"Question: This rice leaf image shows {disease_en} disease. "
                f"Describe the visual symptoms you observe on the leaf in detail. "
                f"Answer:"
            )

        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.model.device)
        import torch
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=150,
                num_beams=5,
                temperature=0.7,
                do_sample=True,
            )
        en_caption = self.processor.decode(output[0], skip_special_tokens=True)

        label = CLASS_LABELS_VI.get(disease_name, disease_name)
        rec = RECOMMENDATION.get(disease_name, "")
        severity = SEVERITY_LEVEL.get(disease_name, "Unknown")

        return (
            f"[VLM] {en_caption}\n"
            f"[Disease] {label} | Severity: {severity}\n"
            f"[Recommendation] {rec}"
        )


# ── LLaVA Caption ─────────────────────────────────────────────
class LLaVACaptioner:
    """
    LLaVA-1.5-7B — runs on GPU (~8GB VRAM with 4-bit).
    Higher caption quality than BLIP-2.
    Requires: pip install transformers accelerate bitsandbytes
    """

    def __init__(
        self,
        model_id: str = "llava-hf/llava-1.5-7b-hf",
        load_4bit: bool = True,
    ):
        print(f"Loading LLaVA ({model_id})...")
        try:
            from transformers import LlavaForConditionalGeneration, AutoProcessor
            import torch
        except ImportError:
            raise ImportError("pip install transformers accelerate bitsandbytes")

        self.processor = AutoProcessor.from_pretrained(model_id)

        load_kwargs = {"device_map": "auto"}
        if load_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        self.model = LlavaForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
        print("LLaVA loaded")

    def caption(self, image: Image.Image, disease_name: str) -> str:
        disease_en = disease_name.replace("_", " ").title()
        prompt = (
            f"USER: <image>\n"
            f"This is a rice leaf affected by {disease_en}. "
            f"Describe the visual symptoms in detail: location, color, shape, and severity. "
            f"ASSISTANT:"
        )
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.model.device)
        import torch
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
        raw = self.processor.decode(output[0][2:], skip_special_tokens=True)

        label = CLASS_LABELS_VI.get(disease_name, disease_name)
        rec = RECOMMENDATION.get(disease_name, "")
        return f"[LLaVA] {raw}\n[Disease] {label}\n[Recommendation] {rec}"


def get_captioner(mode: str = "template"):
    """Factory: return a captioner instance for the given mode."""
    if mode == "blip2":
        return BLIP2Captioner()
    elif mode == "llava":
        return LLaVACaptioner()
    elif mode == "template":
        return None  # template_caption() is used directly, no object needed
    else:
        raise ValueError("mode must be one of: template | blip2 | llava")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image",    required=True, help="Path to the crop image")
    p.add_argument("--disease",  required=True, choices=ALL_CLASSES)
    p.add_argument("--conf",     type=float, default=0.9)
    p.add_argument("--mode",     default="template", choices=["template", "blip2", "llava"])
    p.add_argument("--bbox",     nargs=4, type=int, metavar=("x1", "y1", "x2", "y2"), default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    img = Image.open(args.image).convert("RGB")
    w, h = img.size

    bbox_pos, area_ratio = "middle-center", 0.0
    if args.bbox:
        bbox_pos, area_ratio = get_bbox_position(args.bbox, w, h)

    if args.mode == "template":
        caption = template_caption(args.disease, args.conf, bbox_pos, area_ratio)
    else:
        captioner = get_captioner(args.mode)
        caption = captioner.caption(img, args.disease)

    print("\nCAPTION:")
    print(caption)
