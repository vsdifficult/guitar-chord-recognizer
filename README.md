# Guitar Chord AI (Research-Grade CNN Pipeline)

Production-style project for guitar chord recognition from fretboard images using transfer learning and advanced CNN engineering.

## Key improvements
- **Backbone**: EfficientNetV2B0 transfer learning.
- **Architecture**: SE channel-attention, spatial attention, FPN-like feature fusion, LayerNorm + BatchNorm.
- **Objective**: multiclass classification with focal cross-entropy and label smoothing.
- **Optimization**: AdamW + weight decay, warmup+cosine LR schedule, gradient clipping, staged fine-tuning.
- **Data**: balanced scraping, CLIP-assisted filtering, pHash deduplication, blur/edge filtering, deterministic splits.
- **Interpretability**: Grad-CAM and Grad-CAM++ overlays.
- **Evaluation**: Accuracy, Precision, Recall, F1, confusion matrix, ROC-AUC per class, calibration curves.

## Project structure
```text
src/
  dataset/
    dataset_builder.py
    dataset_filtering.py
  models/
    cnn_model.py
    losses.py
  training/
    trainer.py
    scheduler.py
  evaluation/
    metrics.py
    evaluate.py
  inference/
    predict.py
  interpretability/
    gradcam.py
  utils/
    config.py
```
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

## End-to-end run
```bash
python src/dataset.py        # scraping
python src/preprocessing.py  # filtering + split
python src/train.py          # two-stage training
python src/evaluate.py       # full metrics
```

## Mathematical setup
Given image \(x\) and one-hot label \(y\), model predicts \(p_\theta(y\mid x)\).

- Base objective: \(\mathcal{L}_{CE} = -\sum_c y_c\log p_c\)
- Focal modulation: \(\mathcal{L}_{focal} = -\sum_c \alpha (1-p_c)^\gamma y_c\log p_c\)
- Label smoothing regularizes hard targets.
- Weight decay regularizes \(\|\theta\|_2^2\).

Total objective combines data loss + regularization.

## Colab/GPU notes
- Pipeline works with `tf.keras` and GPU acceleration.
- CLIP filtering is optional and gracefully degrades to heuristic filtering if unavailable.


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
