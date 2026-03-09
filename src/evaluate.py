import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = 224
BATCH = 32


test_ds = tf.keras.utils.image_dataset_from_directory(
    "data/test",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    shuffle=False,
)

class_names = test_ds.class_names
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

y_true = []
y_pred = []

for x, y in test_ds:
    pred = model.predict(x, verbose=0)
    y_true.extend(y.numpy())
    y_pred.extend(np.argmax(pred, axis=1))

print(classification_report(y_true, y_pred, target_names=class_names))
print(confusion_matrix(y_true, y_pred))
