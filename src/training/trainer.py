from pathlib import Path
import tensorflow as tf

from models.cnn_model import build_model
from models.losses import CategoricalFocalLoss, objective_description
from training.scheduler import WarmupCosineDecay
from utils.config import ProjectConfig


def make_tfdata(path, cfg: ProjectConfig, training=False):
    ds = tf.keras.utils.image_dataset_from_directory(
        path,
        image_size=(cfg.data.image_size, cfg.data.image_size),
        batch_size=cfg.data.batch_size,
        label_mode="categorical",
        shuffle=training,
        seed=cfg.data.seed,
    )

    augment = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomContrast(0.2),
            tf.keras.layers.RandomBrightness(0.15),
            tf.keras.layers.RandomTranslation(height_factor=0.08, width_factor=0.08),
            tf.keras.layers.RandomCrop(cfg.data.image_size - 8, cfg.data.image_size - 8),
            tf.keras.layers.Resizing(cfg.data.image_size, cfg.data.image_size),
        ],
        name="augment_pipeline",
    )

    if training:
        ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    options = tf.data.Options()
    options.deterministic = True
    ds = ds.with_options(options)
    return ds.cache().prefetch(tf.data.AUTOTUNE)


def set_backbone_trainable(backbone, ratio):
    n = len(backbone.layers)
    cut = int(n * (1.0 - ratio))
    for i, layer in enumerate(backbone.layers):
        layer.trainable = i >= cut


def build_optimizer(lr_schedule, cfg):
    return tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=cfg.train.weight_decay,
        clipnorm=cfg.train.gradient_clip_norm,
    )


def train():
    cfg = ProjectConfig()
    cfg.train.model_dir.mkdir(parents=True, exist_ok=True)
    cfg.train.logs_dir.mkdir(parents=True, exist_ok=True)

    print(objective_description())

    train_ds = make_tfdata(cfg.data.train_dir, cfg, training=True)
    val_ds = make_tfdata(cfg.data.val_dir, cfg, training=False)

    model, backbone = build_model(
        num_classes=cfg.train.num_classes,
        input_shape=(cfg.data.image_size, cfg.data.image_size, 3),
        dropout_rate=cfg.train.dropout_head_stage1,
        temperature=1.15,
    )

    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    total_steps_stage1 = max(1, steps_per_epoch * cfg.train.stage1_epochs)
    warmup_steps = max(1, steps_per_epoch * cfg.train.warmup_epochs)

    stage1_lr = WarmupCosineDecay(cfg.train.stage1_lr, warmup_steps, total_steps_stage1, min_lr=1e-5)

    backbone.trainable = False
    model.compile(
        optimizer=build_optimizer(stage1_lr, cfg),
        loss=CategoricalFocalLoss(
            gamma=cfg.train.focal_gamma,
            alpha=cfg.train.focal_alpha,
            label_smoothing=cfg.train.label_smoothing,
        ),
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(cfg.train.model_dir / "best_stage1.keras"), monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        tf.keras.callbacks.TensorBoard(log_dir=str(cfg.train.logs_dir / "stage1")),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=cfg.train.stage1_epochs, callbacks=callbacks)

    set_backbone_trainable(backbone, cfg.train.trainable_backbone_ratio)
    for layer in model.layers:
        if layer.name == "head_dropout":
            layer.rate = cfg.train.dropout_head_stage2

    total_steps_stage2 = max(1, steps_per_epoch * cfg.train.stage2_epochs)
    stage2_lr = WarmupCosineDecay(cfg.train.stage2_lr, warmup_steps, total_steps_stage2, min_lr=1e-6)

    model.compile(
        optimizer=build_optimizer(stage2_lr, cfg),
        loss=CategoricalFocalLoss(
            gamma=cfg.train.focal_gamma,
            alpha=cfg.train.focal_alpha,
            label_smoothing=cfg.train.label_smoothing / 2,
        ),
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
    )

    callbacks2 = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(cfg.train.model_dir / "best_guitar_chord_model.keras"), monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
        tf.keras.callbacks.TensorBoard(log_dir=str(cfg.train.logs_dir / "stage2")),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=cfg.train.stage2_epochs, callbacks=callbacks2)
    model.save(cfg.train.model_dir / "guitar_chord_model.keras")


if __name__ == "__main__":
    train()
