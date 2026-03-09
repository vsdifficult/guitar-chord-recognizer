import tensorflow as tf
from tensorflow.keras import layers

NUM_CLASSES = 6

def build_model():
    base = tf.keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(224,224,3)
    )

    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256,activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    output = layers.Dense(NUM_CLASSES,activation="softmax")(x)

    model = tf.keras.Model(base.input,output)

    return model