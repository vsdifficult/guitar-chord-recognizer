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

## Install
```bash
pip install -r requirements.txt
```

## End-to-end run
### Unified entrypoint (recommended)
```bash
python main.py full
```

### Step-by-step commands
```bash
python main.py dataset       # scraping
python main.py preprocess    # filtering + split
python main.py train         # two-stage training
python main.py evaluate      # full metrics
python main.py predict --image sample.jpg
python main.py gradcam --image sample.jpg --method gradcam++
```

### Legacy wrappers (still supported)
```bash
python src/dataset.py
python src/preprocessing.py
python src/train.py
python src/evaluate.py
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
