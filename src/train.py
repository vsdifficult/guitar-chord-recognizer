import tensorflow as tf

from model import build_model

IMG_SIZE = 224
BATCH = 32
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE


def make_dataset(path, training=False):
    ds = tf.keras.utils.image_dataset_from_directory(
        path,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH,
        shuffle=training,
        seed=SEED,
    )

    if training:
        augmenter = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.08),
                tf.keras.layers.RandomZoom(0.12),
                tf.keras.layers.RandomContrast(0.1),
            ]
        )
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=AUTOTUNE)

    return ds.cache().prefetch(AUTOTUNE)


train_ds = make_dataset("data/train", training=True)
val_ds = make_dataset("data/val")

model = build_model()

model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "models/best_guitar_chord_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=40,
    callbacks=callbacks,
)

model.save("models/guitar_chord_model.keras")
print("Training finished. Best val_accuracy:", max(history.history["val_accuracy"]))
