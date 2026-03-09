from collections import defaultdict
from pathlib import Path
import random
import shutil

import cv2
import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class CLIPFilter:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.device = "cpu"
        self.model = None
        self.preprocess = None
        self.clip = None

        if not enabled:
            return
        try:
            import torch
            import clip

            self.clip = clip
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            self.model.eval()
            self.torch = torch
            prompts = [
                "a real photo of a person playing a guitar fretboard",
                "a guitar chord chart or tab diagram",
                "an illustration or drawing",
            ]
            self.text = clip.tokenize(prompts).to(self.device)
        except Exception as exc:
            print(f"CLIP unavailable, fallback to heuristic filter only: {exc}")
            self.enabled = False

    def keep(self, image_bgr: np.ndarray) -> bool:
        if not self.enabled:
            return True

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        with self.torch.no_grad():
            x = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            logits_per_image, _ = self.model(x, self.text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

        # keep if real-photo probability dominates diagram/illustration
        return probs[0] > max(probs[1], probs[2])


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def blur_score(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def has_fretboard_edges(image_bgr: np.ndarray) -> bool:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=80, maxLineGap=10)
    return lines is not None and len(lines) >= 5


def clean_dataset(raw_dir="data/raw", clean_dir="data/clean", min_size=256, blur_threshold=70.0, dedup_distance=6, use_clip=True):
    raw_root = Path(raw_dir)
    clean_root = Path(clean_dir)
    clean_root.mkdir(parents=True, exist_ok=True)

    clip_filter = CLIPFilter(enabled=use_clip)
    seen_hashes = []
    stats = defaultdict(int)

    for class_dir in sorted(raw_root.iterdir()):
        if not class_dir.is_dir():
            continue
        out_dir = clean_root / class_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        for path in tqdm(list(class_dir.iterdir()), desc=f"filter:{class_dir.name}"):
            if path.suffix.lower() not in IMAGE_EXTS:
                stats["bad_ext"] += 1
                continue
            if not is_valid_image(path):
                stats["corrupted"] += 1
                continue

            image = cv2.imread(str(path))
            if image is None:
                stats["cv2_fail"] += 1
                continue

            h, w = image.shape[:2]
            if min(h, w) < min_size:
                stats["too_small"] += 1
                continue
            if blur_score(image) < blur_threshold:
                stats["blurry"] += 1
                continue
            if not has_fretboard_edges(image):
                stats["no_fretboard"] += 1
                continue
            if not clip_filter.keep(image):
                stats["clip_reject"] += 1
                continue

            ph = imagehash.phash(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
            if any(abs(ph - p) <= dedup_distance for p in seen_hashes):
                stats["duplicate"] += 1
                continue
            seen_hashes.append(ph)

            cv2.imwrite(str(out_dir / path.name), image)
            stats["saved"] += 1

    print("Cleaning summary:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")


def split_dataset(clean_dir="data/clean", output_dir="data", train_ratio=0.75, val_ratio=0.15, seed=42):
    rng = random.Random(seed)
    clean_root = Path(clean_dir)
    out_root = Path(output_dir)

    for split in ["train", "val", "test"]:
        split_dir = out_root / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)

    for class_dir in sorted(clean_root.iterdir()):
        if not class_dir.is_dir():
            continue

        images = [p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        rng.shuffle(images)
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        packs = {
            "train": images[:n_train],
            "val": images[n_train : n_train + n_val],
            "test": images[n_train + n_val :],
        }

        for split, subset in packs.items():
            class_out = out_root / split / class_dir.name
            class_out.mkdir(parents=True, exist_ok=True)
            for src in subset:
                shutil.copy2(src, class_out / src.name)

        print(f"[{class_dir.name}] total={n} train={len(packs['train'])} val={len(packs['val'])} test={len(packs['test'])}")


if __name__ == "__main__":
    clean_dataset()
    split_dataset()
