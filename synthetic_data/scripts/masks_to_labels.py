import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import yaml


def contours_to_yolo(mask_gray: np.ndarray, class_id: int, min_area: float) -> list[str]:
    binary = (mask_gray > 0).astype(np.uint8) * 255
    h, w = binary.shape

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue

        epsilon = 0.005 * cv2.arcLength(cnt, closed=True)
        approx  = cv2.approxPolyDP(cnt, epsilon, closed=True).reshape(-1, 2)

        if len(approx) < 3:
            continue

        coords = " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in approx)
        lines.append(f"{class_id} {coords}")

    return lines


def process_dataset(data_dir: Path, val_split: float, min_area: float, seed: int = 42) -> None:
    images_dir = data_dir / "images"
    masks_dir  = data_dir / "masks"
    labels_dir = data_dir / "labels"
    labels_dir.mkdir(exist_ok=True)

    # Sorted so class IDs stay stable across runs
    class_names = sorted(p.name for p in masks_dir.iterdir() if p.is_dir())
    if not class_names:
        raise RuntimeError(f"No class subdirectories found in {masks_dir}")
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    print(f"Classes: {class_to_id}")

    image_paths = sorted(images_dir.glob("*.png"))
    if not image_paths:
        raise RuntimeError(f"No PNG images found in {images_dir}")

    processed = []
    for img_path in image_paths:
        stem = img_path.stem
        label_lines = []

        for cls_name, cls_id in class_to_id.items():
            mask_path = masks_dir / cls_name / f"{stem.replace('image', 'mask_' + cls_name)}.png"
            if not mask_path.exists():
                continue
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            label_lines.extend(contours_to_yolo(mask, cls_id, min_area))

        if not any((masks_dir / cls).glob(f"{stem.replace('image', 'mask_*')}.png")
                   for cls in class_names):
            print(f"  [skip] no masks found for {img_path.name}")
            continue

        label_path = labels_dir / f"{stem}.txt"
        label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""))
        processed.append(img_path)
        print(f"  {img_path.name} → {len(label_lines)} instance(s)")

    if not processed:
        raise RuntimeError("No images were processed.")

    random.seed(seed)
    shuffled = processed.copy()
    random.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_split)) if len(shuffled) > 1 else 0
    val_imgs   = shuffled[:n_val]
    train_imgs = shuffled[n_val:]

    (data_dir / "train.txt").write_text("\n".join(str(p.resolve()) for p in train_imgs) + "\n")
    (data_dir / "val.txt").write_text("\n".join(str(p.resolve()) for p in val_imgs) + "\n")

    yaml_data = {
        "path": str(data_dir.resolve()),
        "train": "train.txt",
        "val": "val.txt",
        "nc": len(class_names),
        "names": {idx: name for idx, name in enumerate(class_names)},
    }
    with open(data_dir / "data.yaml", "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    print(
        f"\nDone. {len(processed)} image(s) — "
        f"{len(train_imgs)} train / {len(val_imgs)} val.\n"
        f"Config: {data_dir / 'data.yaml'}"
    )


def main() -> None:
    default_data_dir = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",  type=Path, default=default_data_dir)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--min-area",  type=float, default=10.0)
    args = parser.parse_args()

    process_dataset(args.data_dir, args.val_split, args.min_area)


if __name__ == "__main__":
    main()
