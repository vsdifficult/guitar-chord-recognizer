# Guitar Chord AI

AI system for recognizing guitar chords from fretboard images.

Supported chords: `C`, `G`, `Am`, `D`, `Em`, `F`.

## Updated pipeline
1. **Collect** balanced raw images per chord with multiple queries.
2. **Clean** data (corrupted / blurry / too small / near-duplicate removal).
3. **Split** into stratified train/val/test sets.
4. **Train** optimized CNN (`ChordNet`) with augmentations and callbacks.
5. **Evaluate** quality on the test set.

## Install
```bash
pip install -r requirements.txt
```

## Run
1. Download dataset (balanced):
```bash
python src/dataset.py
```

2. Clean and split dataset:
```bash
python src/preprocessing.py
```

3. Train model:
```bash
python src/train.py
```

4. Evaluate model:
```bash
python src/evaluate.py
```

Model artifacts are saved to `models/`.
