"""Configuration for neural tabular experiments."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from fraud_detection.config import FEATURE_STRATEGIES, FeatureStrategy


DeepModelName = Literal["ft_transformer", "tabm"]
LossName = Literal["weighted_bce", "focal", "weighted_bce+deviation"]


@dataclass(frozen=True)
class DeepTrainingConfig:
    model_name: DeepModelName
    data_dir: Path
    feature_strategy: FeatureStrategy = "all"
    loss_name: LossName = "weighted_bce"
    validation_fraction: float = 0.2
    batch_size: int = 2048
    max_epochs: int = 30
    patience: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    d_token: int = 64
    n_layers: int = 3
    n_heads: int = 8
    dropout: float = 0.1
    tabm_k: int = 16
    focal_alpha: float = 0.95
    focal_gamma: float = 2.0
    deviation_weight: float = 1.0
    deviation_margin: float = 5.0
    deviation_reference_samples: int = 5_000
    num_workers: int = 0
    random_state: int = 42
    output_dir: Path = Path("experiments")

    def validate(self) -> None:
        if self.model_name not in ("ft_transformer", "tabm"):
            raise ValueError(f"Unknown deep model: {self.model_name}")
        if self.loss_name not in ("weighted_bce", "focal", "weighted_bce+deviation"):
            raise ValueError(f"Unknown loss: {self.loss_name}")
        if self.feature_strategy not in FEATURE_STRATEGIES:
            raise ValueError(f"Unknown feature strategy: {self.feature_strategy}")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1.")
        for name in ("batch_size", "max_epochs", "patience", "d_token", "n_layers"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1.")
        if self.d_token % self.n_heads:
            raise ValueError("d_token must be divisible by n_heads.")
        if not 0 < self.focal_alpha < 1:
            raise ValueError("focal_alpha must be between 0 and 1.")
        if self.deviation_weight < 0:
            raise ValueError("deviation_weight cannot be negative.")
        if self.deviation_margin <= 0:
            raise ValueError("deviation_margin must be positive.")
        if self.deviation_reference_samples < 2:
            raise ValueError("deviation_reference_samples must be at least 2.")

    def serializable(self) -> dict[str, Any]:
        values = asdict(self)
        values["data_dir"] = str(self.data_dir)
        values["output_dir"] = str(self.output_dir)
        return values
