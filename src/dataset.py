import os
import random
from pathlib import Path

from icrawler.builtin import BingImageCrawler

CHORDS = ["C", "G", "Am", "D", "Em", "F"]

QUERIES = {
    "C": [
        "guitar C chord fretboard close up",
        "acoustic guitar C major chord fingers",
        "left hand C chord on guitar neck",
    ],
    "G": [
        "guitar G chord fretboard close up",
        "acoustic guitar G major chord fingers",
        "left hand G chord on guitar neck",
    ],
    "Am": [
        "guitar A minor chord fretboard close up",
        "acoustic guitar Am chord fingers",
        "left hand Am chord on guitar neck",
    ],
    "D": [
        "guitar D chord fretboard close up",
        "acoustic guitar D major chord fingers",
        "left hand D chord on guitar neck",
    ],
    "Em": [
        "guitar E minor chord fretboard close up",
        "acoustic guitar Em chord fingers",
        "left hand Em chord on guitar neck",
    ],
    "F": [
        "guitar F chord barre close up",
        "acoustic guitar F major chord fingers",
        "left hand F barre chord on guitar neck",
    ],
}


def count_images(folder: Path) -> int:
    return len([p for p in folder.iterdir() if p.is_file()]) if folder.exists() else 0


def download_dataset(base_dir="data/raw", per_chord=500, seed=42):
    random.seed(seed)
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    for chord in CHORDS:
        chord_dir = base_path / chord
        chord_dir.mkdir(parents=True, exist_ok=True)

        existing_count = count_images(chord_dir)
        if existing_count >= per_chord:
            print(f"[{chord}] already has {existing_count} images, skipping.")
            continue

        query_pool = QUERIES[chord][:]
        random.shuffle(query_pool)

        print(f"[{chord}] current: {existing_count}, target: {per_chord}")
        for query in query_pool:
            remaining = per_chord - count_images(chord_dir)
            if remaining <= 0:
                break

            print(f"  -> query='{query}' (need ~{remaining} more)")
            crawler = BingImageCrawler(storage={"root_dir": str(chord_dir)})
            crawler.crawl(keyword=query, max_num=remaining)

        final_count = count_images(chord_dir)
        print(f"[{chord}] downloaded total: {final_count}")


if __name__ == "__main__":
    download_dataset()
