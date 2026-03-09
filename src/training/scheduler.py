import math
import tensorflow as tf


class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, total_steps, min_lr=1e-6, name="warmup_cosine"):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.min_lr = min_lr
        self.name = name

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)

        warmup_lr = self.initial_lr * (step / warmup_steps)
        progress = tf.clip_by_value((step - warmup_steps) / (total_steps - warmup_steps), 0.0, 1.0)
        cosine = 0.5 * (1 + tf.cos(math.pi * progress))
        decayed = self.min_lr + (self.initial_lr - self.min_lr) * cosine
        return tf.where(step < warmup_steps, warmup_lr, decayed)

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
            "name": self.name,
        }
