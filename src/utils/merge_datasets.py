import shutil
import yaml
from pathlib import Path
from typing import Dict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from configs.config import ROOT_DIR, RAW_DATASETS, YOLO_DATASET_DIR, ALL_CLASSES, YOLO_DATA_YAML

def read_roboflow_yaml(dataset_path: Path) -> Dict:
    yaml_files = list(dataset_path.glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No YAML file found in {dataset_path}")
    with open(yaml_files[0], "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_class_mapping(src_yaml: Dict, disease_name: str) -> Dict[int, int]:
    if disease_name not in ALL_CLASSES:
        raise ValueError(f"{disease_name!r} is not in ALL_CLASSES")
    target_idx = ALL_CLASSES.index(disease_name)
    src_classes = src_yaml.get("names", [])
    return {src_idx: target_idx for src_idx in range(len(src_classes))}

def remap_label_file(src_label: Path, dst_label: Path, class_mapping: Dict[int, int]) -> None:
    lines_out = []
    with open(src_label, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            old_cls = int(parts[0])
            new_cls = class_mapping.get(old_cls, old_cls)
            lines_out.append(f"{new_cls} {' '.join(parts[1:])}")

    dst_label.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_label, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out) + ("\n" if lines_out else ""))

def copy_split(src_dataset: Path, disease_name: str, class_mapping: Dict[int, int], split: str, counter: Dict[str, int]) -> int:
    src_images = src_dataset / split / "images"
    src_labels = src_dataset / split / "labels"

    dst_split = "val" if split == "valid" else split
    dst_images = YOLO_DATASET_DIR / "images" / dst_split
    dst_labels = YOLO_DATASET_DIR / "labels" / dst_split
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    if not src_images.exists():
        return 0

    copied = 0
    for img_path in src_images.iterdir():
        if not img_path.is_file():
            continue

        count = counter.get(dst_split, 0)
        new_stem = f"{disease_name}_{count:05d}_{img_path.stem}"
        counter[dst_split] = count + 1

        dst_img = dst_images / (new_stem + img_path.suffix)
        shutil.copy2(img_path, dst_img)

        label_path = src_labels / (img_path.stem + ".txt")
        dst_lbl = dst_labels / (new_stem + ".txt")
        if label_path.exists():
            remap_label_file(label_path, dst_lbl, class_mapping)
        else:
            dst_lbl.touch()

        copied += 1

    return copied

def generate_data_yaml() -> None:
    data = {
        "path": str(YOLO_DATASET_DIR),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(ALL_CLASSES),
        "names": ALL_CLASSES,
    }
    with open(YOLO_DATA_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

def merge_all_datasets() -> None:
    print("=" * 60)
    print("STARTING DATASET MERGE")
    print("=" * 60)

    if YOLO_DATASET_DIR.exists():
        for split in ["train", "val", "test"]:
            for sub in ["images", "labels"]:
                d = YOLO_DATASET_DIR / sub / split
                if d.exists():
                    shutil.rmtree(d)

    counter: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    stats: Dict[str, Dict[str, int]] = {}

    for disease_name, dataset_path in RAW_DATASETS.items():
        if not dataset_path.exists():
            print(f"SKIP {disease_name}: missing {dataset_path}")
            continue

        print(f"Processing: {disease_name}")
        src_yaml = read_roboflow_yaml(dataset_path)
        class_mapping = get_class_mapping(src_yaml, disease_name)

        stats[disease_name] = {}
        for split in ["train", "valid", "test"]:
            dst_split = "val" if split == "valid" else split
            before = counter.get(dst_split, 0)
            copied = copy_split(dataset_path, disease_name, class_mapping, split, counter)
            after = counter.get(dst_split, 0)
            stats[disease_name][dst_split] = after - before
            print(f"  {dst_split}: +{copied} images")

    generate_data_yaml()

    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    for disease, s in stats.items():
        total = sum(s.values())
        print(
            f"{disease:<30} train={s.get('train',0):4d} "
            f"val={s.get('val',0):4d} test={s.get('test',0):4d} total={total:4d}"
        )

    total_train = counter["train"]
    total_val = counter["val"]
    total_test = counter["test"]
    print(
        f"{'TOTAL':<30} train={total_train:4d} val={total_val:4d} "
        f"test={total_test:4d} total={total_train + total_val + total_test:4d}"
    )
    print("MERGE COMPLETE")

if __name__ == "__main__":
    merge_all_datasets()