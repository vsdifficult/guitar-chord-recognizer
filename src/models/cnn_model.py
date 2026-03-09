import tensorflow as tf
from tensorflow.keras import layers


def squeeze_excitation(x, ratio=0.25, name="se"):
    channels = x.shape[-1]
    se = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    se = layers.Dense(max(8, int(channels * ratio)), activation="swish", name=f"{name}_fc1")(se)
    se = layers.Dense(channels, activation="sigmoid", name=f"{name}_fc2")(se)
    se = layers.Reshape((1, 1, channels), name=f"{name}_reshape")(se)
    return layers.Multiply(name=f"{name}_scale")([x, se])


def spatial_attention(x, name="spatial_attn"):
    avg = layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True), name=f"{name}_avg")(x)
    mx = layers.Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True), name=f"{name}_max")(x)
    feat = layers.Concatenate(axis=-1, name=f"{name}_concat")([avg, mx])
    attn = layers.Conv2D(1, 7, padding="same", activation="sigmoid", name=f"{name}_conv")(feat)
    return layers.Multiply(name=f"{name}_mul")([x, attn])


class TemperatureSoftmax(layers.Layer):
    def __init__(self, temperature=1.0, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def call(self, logits):
        return tf.nn.softmax(logits / self.temperature)


def build_model(num_classes=6, input_shape=(260, 260, 3), dropout_rate=0.3, temperature=1.0):
    inputs = layers.Input(shape=input_shape)

    backbone = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
    )

    x = backbone.output
    x = squeeze_excitation(x, ratio=0.25, name="neck_se")
    x = spatial_attention(x, name="neck_sa")

    p2 = layers.Conv2D(128, 1, padding="same", use_bias=False, name="p2_conv")(x)
    p2 = layers.BatchNormalization(name="p2_bn")(p2)
    p2 = layers.Activation("swish", name="p2_act")(p2)

    p1 = layers.GlobalAveragePooling2D(name="gap")(x)
    p2_pool = layers.GlobalAveragePooling2D(name="p2_gap")(p2)
    fused = layers.Concatenate(name="fpn_like_fusion")([p1, p2_pool])

    fused = layers.LayerNormalization(name="head_ln")(fused)
    fused = layers.Dense(512, activation="swish", kernel_regularizer=tf.keras.regularizers.l2(1e-4), name="head_fc1")(fused)
    fused = layers.Dropout(dropout_rate, name="head_dropout")(fused)
    logits = layers.Dense(num_classes, name="classifier_logits")(fused)
    outputs = TemperatureSoftmax(temperature=temperature, name="temperature_softmax")(logits)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ChordRecognizerV2")
    return model, backbone
