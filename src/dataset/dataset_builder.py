from pathlib import Path
import random

from icrawler.builtin import BingImageCrawler

CHORD_QUERIES = {
    "C": ["guitar C major chord fretboard close up", "left hand C chord on guitar neck", "acoustic C chord fingering real photo"],
    "G": ["guitar G major chord fretboard close up", "left hand G chord on guitar neck", "acoustic G chord fingering real photo"],
    "Am": ["guitar A minor chord fretboard close up", "left hand Am chord on guitar neck", "acoustic Am chord fingering real photo"],
    "D": ["guitar D major chord fretboard close up", "left hand D chord on guitar neck", "acoustic D chord fingering real photo"],
    "Em": ["guitar E minor chord fretboard close up", "left hand Em chord on guitar neck", "acoustic Em chord fingering real photo"],
    "F": ["guitar F major barre chord fretboard close up", "left hand F barre chord on guitar neck", "acoustic F chord fingering real photo"],
}


def count_files(folder: Path) -> int:
    return sum(1 for p in folder.iterdir() if p.is_file()) if folder.exists() else 0


def download_balanced_dataset(raw_dir="data/raw", per_class=600, seed=42):
    random.seed(seed)
    root = Path(raw_dir)
    root.mkdir(parents=True, exist_ok=True)

    for chord, queries in CHORD_QUERIES.items():
        class_dir = root / chord
        class_dir.mkdir(parents=True, exist_ok=True)

        if count_files(class_dir) >= per_class:
            print(f"[{chord}] already balanced")
            continue

        order = queries[:]
        random.shuffle(order)
        for q in order:
            remaining = max(0, per_class - count_files(class_dir))
            if remaining == 0:
                break
            print(f"[{chord}] query='{q}' remaining={remaining}")
            crawler = BingImageCrawler(storage={"root_dir": str(class_dir)})
            crawler.crawl(keyword=q, max_num=remaining)

        print(f"[{chord}] final_count={count_files(class_dir)}")


if __name__ == "__main__":
    download_balanced_dataset()
