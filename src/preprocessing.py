import os
import cv2
import imagehash
from PIL import Image
from tqdm import tqdm

MIN_SIZE = 300

def remove_corrupted(path):
    try:
        img = Image.open(path)
        img.verify()
        return True
    except:
        return False

def is_blurry(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score < 80

def clean_dataset(raw_dir="data/raw", clean_dir="data/clean"):
    hashes = set()

    for chord in os.listdir(raw_dir):
        os.makedirs(f"{clean_dir}/{chord}", exist_ok=True)
        folder = f"{raw_dir}/{chord}"

        for img_name in tqdm(os.listdir(folder)):
            path = os.path.join(folder,img_name)

            if not remove_corrupted(path):
                continue

            img = cv2.imread(path)
            if img is None:
                continue

            h,w,_ = img.shape
            if min(h,w) < MIN_SIZE:
                continue

            if is_blurry(img):
                continue

            pil_img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            h = imagehash.phash(pil_img)

            if h in hashes:
                continue

            hashes.add(h)
            save_path = f"{clean_dir}/{chord}/{img_name}"
            cv2.imwrite(save_path,img)