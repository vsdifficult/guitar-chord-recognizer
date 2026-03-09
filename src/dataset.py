import os
from icrawler.builtin import BingImageCrawler

CHORDS = ["C","G","Am","D","Em","F"]

queries = {
"C":["guitar C chord fretboard close up","acoustic guitar C chord fingers close up"],
"G":["guitar G chord fretboard close up","acoustic guitar G chord fingers close up"],
"Am":["guitar Am chord fretboard close up"],
"D":["guitar D chord fretboard close up"],
"Em":["guitar Em chord fretboard close up"],
"F":["guitar F chord barre close up"]
}

def download_dataset(base_dir="data/raw", per_query=200):

    os.makedirs(base_dir, exist_ok=True)

    for chord in CHORDS:
        chord_dir = os.path.join(base_dir, chord)
        os.makedirs(chord_dir, exist_ok=True)

        for q in queries[chord]:
            crawler = BingImageCrawler(storage={"root_dir": chord_dir})
            crawler.crawl(keyword=q, max_num=per_query)

if __name__ == "__main__":
    download_dataset()