import tensorflow as tf
from model import build_model

IMG_SIZE = 224
BATCH = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
"data/train",
image_size=(IMG_SIZE,IMG_SIZE),
batch_size=BATCH
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
"data/val",
image_size=(IMG_SIZE,IMG_SIZE),
batch_size=BATCH
)

model = build_model()

model.compile(
optimizer=tf.keras.optimizers.Adam(1e-3),
loss="sparse_categorical_crossentropy",
metrics=["accuracy"]
)

model.fit(train_ds, validation_data=val_ds, epochs=10)

model.layers[0].trainable = True

model.compile(
optimizer=tf.keras.optimizers.Adam(1e-5),
loss="sparse_categorical_crossentropy",
metrics=["accuracy"]
)

model.fit(train_ds, validation_data=val_ds, epochs=15)

model.save("models/guitar_chord_model.keras")