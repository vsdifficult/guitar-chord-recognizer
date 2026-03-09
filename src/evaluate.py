import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

model = tf.keras.models.load_model("models/guitar_chord_model.keras")

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
"data/test",
image_size=(224,224)
)

y_true = []
y_pred = []

for x,y in test_ds:
    pred = model.predict(x)
    y_true.extend(y.numpy())
    y_pred.extend(np.argmax(pred,axis=1))

print(classification_report(y_true,y_pred))
print(confusion_matrix(y_true,y_pred))