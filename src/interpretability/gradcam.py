from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


def _find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if len(getattr(layer, "output_shape", [])) == 4:
            return layer.name
    raise ValueError("No convolutional layer found for Grad-CAM")


def _overlay_heatmap(image, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 1 - alpha, color, alpha, 0)


def gradcam(model, image_array, class_index=None, layer_name=None):
    if layer_name is None:
        layer_name = _find_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(image_array)
        if class_index is None:
            class_index = tf.argmax(preds[0])
        score = preds[:, class_index]

    grads = tape.gradient(score, conv_outputs)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.reduce_sum(weights * conv_outputs, axis=-1)[0]
    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy()


def gradcam_pp(model, image_array, class_index=None, layer_name=None):
    if layer_name is None:
        layer_name = _find_last_conv_layer(model)

    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            conv_outputs, preds = grad_model(image_array)
            if class_index is None:
                class_index = tf.argmax(preds[0])
            score = preds[:, class_index]
        first = tape2.gradient(score, conv_outputs)
    second = tape1.gradient(first, conv_outputs)

    alpha_num = second
    alpha_den = 2.0 * second + tf.reduce_sum(conv_outputs * second, axis=(1, 2), keepdims=True)
    alpha = alpha_num / (alpha_den + 1e-8)
    weights = tf.reduce_sum(alpha * tf.nn.relu(first), axis=(1, 2), keepdims=True)

    cam = tf.reduce_sum(weights * conv_outputs, axis=-1)[0]
    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy()


def save_overlay(model_path, image_path, output_path="models/gradcam_overlay.jpg", method="gradcam"):
    model = tf.keras.models.load_model(model_path, compile=False)
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (260, 260))
    batch = np.expand_dims(image_resized.astype(np.float32), axis=0)

    if method == "gradcam++":
        heatmap = gradcam_pp(model, batch)
    else:
        heatmap = gradcam(model, batch)

    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    overlay = _overlay_heatmap(image, heatmap)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), overlay)
    return str(out)
