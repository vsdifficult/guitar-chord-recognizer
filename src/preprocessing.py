import os
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import imagehash
from PIL import Image
from tqdm import tqdm

MIN_SIZE = 300
BLUR_THRESHOLD = 80
DUPLICATE_HASH_DISTANCE = 5
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def remove_corrupted(path: Path) -> bool:
    try:
        img = Image.open(path)
        img.verify()
        return True
    except Exception:
        return False


def is_blurry(img) -> bool:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score < BLUR_THRESHOLD


def clean_dataset(raw_dir="data/raw", clean_dir="data/clean"):
    raw_path = Path(raw_dir)
    clean_path = Path(clean_dir)
    duplicate_hashes = []
    stats = defaultdict(int)

    for chord_dir in sorted(raw_path.iterdir()):
        if not chord_dir.is_dir():
            continue

        target_dir = clean_path / chord_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(list(chord_dir.iterdir()), desc=f"Cleaning {chord_dir.name}"):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                stats["unsupported_ext"] += 1
                continue

            if not remove_corrupted(img_path):
                stats["corrupted"] += 1
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                stats["read_fail"] += 1
                continue

            h, w, _ = img.shape
            if min(h, w) < MIN_SIZE:
                stats["too_small"] += 1
                continue

            if is_blurry(img):
                stats["blurry"] += 1
                continue

            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img_hash = imagehash.phash(pil_img)

            if any(abs(img_hash - seen_hash) <= DUPLICATE_HASH_DISTANCE for seen_hash in duplicate_hashes):
                stats["duplicate"] += 1
                continue

            duplicate_hashes.append(img_hash)
            save_path = target_dir / img_path.name
            cv2.imwrite(str(save_path), img)
            stats["saved"] += 1

    print("Cleaning stats:")
    for key, value in sorted(stats.items()):
        print(f"  {key}: {value}")


def split_dataset(clean_dir="data/clean", output_dir="data", train_ratio=0.75, val_ratio=0.15, seed=42):
    rng = __import__("random").Random(seed)
    clean_path = Path(clean_dir)
    output_path = Path(output_dir)

    for split in ["train", "val", "test"]:
        split_dir = output_path / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)

    for chord_dir in sorted(clean_path.iterdir()):
        if not chord_dir.is_dir():
            continue

        images = [p for p in chord_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        rng.shuffle(images)
        total = len(images)

        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        split_map = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }

        for split, split_images in split_map.items():
            class_out = output_path / split / chord_dir.name
            class_out.mkdir(parents=True, exist_ok=True)
            for img_path in split_images:
                shutil.copy2(img_path, class_out / img_path.name)

        print(
            f"[{chord_dir.name}] total={total}, train={len(split_map['train'])}, "
            f"val={len(split_map['val'])}, test={len(split_map['test'])}"
        )


if __name__ == "__main__":
    clean_dataset()
    split_dataset()
