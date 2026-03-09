import tensorflow as tf


class CategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.75, label_smoothing=0.0, name="categorical_focal_loss"):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        if self.label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1.0 - self.label_smoothing) + self.label_smoothing / num_classes

        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        ce = -y_true * tf.math.log(y_pred)
        focal_weight = self.alpha * tf.pow(1.0 - y_pred, self.gamma)
        return tf.reduce_sum(focal_weight * ce, axis=-1)


def objective_description() -> str:
    return (
        "Objective: minimize multiclass cross-entropy over p_theta(y|x) with regularized parameters. "
        "Optional focal modulation improves hard-example weighting under class imbalance."
    )
