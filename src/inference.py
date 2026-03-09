import tensorflow as tf
import cv2
import numpy as np

CLASSES = ["C","G","Am","D","Em","F"]

model = tf.keras.models.load_model("models/guitar_chord_model.keras")

def predict(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img,(224,224))
    img = img/255.0
    img = np.expand_dims(img,0)

    pred = model.predict(img)[0]

    chord = CLASSES[np.argmax(pred)]
    confidence = np.max(pred)

    return chord, confidence