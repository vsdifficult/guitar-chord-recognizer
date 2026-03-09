from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    raw_dir: Path = Path("data/raw")
    clean_dir: Path = Path("data/clean")
    train_dir: Path = Path("data/train")
    val_dir: Path = Path("data/val")
    test_dir: Path = Path("data/test")
    image_size: int = 260
    batch_size: int = 32
    seed: int = 42


@dataclass
class TrainConfig:
    num_classes: int = 6
    label_smoothing: float = 0.1
    focal_gamma: float = 1.5
    focal_alpha: float = 0.75
    warmup_epochs: int = 3
    stage1_epochs: int = 12
    stage2_epochs: int = 20
    stage1_lr: float = 1e-3
    stage2_lr: float = 2e-5
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    dropout_head_stage1: float = 0.25
    dropout_head_stage2: float = 0.45
    trainable_backbone_ratio: float = 0.25
    model_dir: Path = Path("models")
    logs_dir: Path = Path("logs")


@dataclass
class ProjectConfig:
    chords: list[str] = field(default_factory=lambda: ["C", "G", "Am", "D", "Em", "F"])
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
