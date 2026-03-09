from models.cnn_model import build_model

NUM_CLASSES = 6


def conv_bn_act(x, filters, kernel_size=3, strides=1, dropout=0.0):
    x = layers.SeparableConv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    return x


def residual_block(x, filters, strides=1, dropout=0.0):
    shortcut = x

    x = conv_bn_act(x, filters, strides=strides, dropout=dropout)
    x = conv_bn_act(x, filters, dropout=dropout)

    if shortcut.shape[-1] != filters or strides != 1:
        shortcut = layers.Conv2D(filters, 1, strides=strides, padding="same", use_bias=False)(
            shortcut
        )
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    return layers.Activation("swish")(x)


def build_model(input_shape=(224, 224, 3), num_classes=NUM_CLASSES):
    inputs = layers.Input(shape=input_shape)

    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)

    x = residual_block(x, 64, strides=2, dropout=0.05)
    x = residual_block(x, 64, dropout=0.05)

    x = residual_block(x, 128, strides=2, dropout=0.1)
    x = residual_block(x, 128, dropout=0.1)

    x = residual_block(x, 256, strides=2, dropout=0.1)
    x = residual_block(x, 256, dropout=0.1)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="swish")(x)
    x = layers.Dropout(0.35)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs, name="ChordNet")
