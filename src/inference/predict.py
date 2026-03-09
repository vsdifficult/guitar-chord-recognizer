import cv2
import numpy as np
import tensorflow as tf

from utils.config import ProjectConfig


def predict(image_path, model_path="models/guitar_chord_model.keras"):
    cfg = ProjectConfig()
    model = tf.keras.models.load_model(model_path, compile=False)

    image = cv2.imread(image_path)
    image = cv2.resize(image, (cfg.data.image_size, cfg.data.image_size))
    x = np.expand_dims(image.astype(np.float32), axis=0)

    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    chord = cfg.chords[idx]
    return chord, float(probs[idx]), probs


if __name__ == "__main__":
    chord, conf, _ = predict("sample.jpg")
    print(f"Prediction: {chord} ({conf:.3f})")
