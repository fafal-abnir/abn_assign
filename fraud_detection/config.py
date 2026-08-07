
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


ModelName = Literal["xgboost", "random_forest", "catboost", "lightgbm"]
SamplingStrategy = Literal["none", "random_under", "random_over", "smote"]
FeatureStrategy = Literal["base", "time", "geo", "behavioral", "all"]
MODEL_NAMES: tuple[ModelName, ...] = (
    "xgboost",
    "random_forest",
    "catboost",
    "lightgbm",
)
SAMPLING_STRATEGIES: tuple[SamplingStrategy, ...] = (
    "none",
    "random_under",
    "random_over",
    "smote",
)
FEATURE_STRATEGIES: tuple[FeatureStrategy, ...] = (
    "base",
    "time",
    "geo",
    "behavioral",
    "all",
)

TARGET = "is_fraud"
TIME_COLUMN = "trans_date_trans_time"
BASE_CATEGORICAL_COLUMNS = ("merchant", "category", "gender", "state", "job")
BASE_NUMERIC_COLUMNS = (
    "amt",
    "city_pop",
    "age",
    "log_amount",
    "is_round_amount",
)
TIME_NUMERIC_COLUMNS = (
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
)
GEO_NUMERIC_COLUMNS = (
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "distance_km",
    "likely_different_state",
    "distance_from_previous_transaction_km",
)
BEHAVIORAL_NUMERIC_COLUMNS = (
    "card_transaction_number",
    "card_age_days",
    "hours_since_previous_transaction",
    "amount_vs_previous_ratio",
    "amount_vs_historical_mean_ratio",
)


def feature_columns(strategy: FeatureStrategy) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return categorical and numeric columns for an experiment strategy."""
    groups = {
        "base": (),
        "time": TIME_NUMERIC_COLUMNS,
        "geo": GEO_NUMERIC_COLUMNS,
        "behavioral": BEHAVIORAL_NUMERIC_COLUMNS,
        "all": TIME_NUMERIC_COLUMNS + GEO_NUMERIC_COLUMNS + BEHAVIORAL_NUMERIC_COLUMNS,
    }
    if strategy not in groups:
        raise ValueError(f"Unknown feature strategy {strategy!r}")
    return BASE_CATEGORICAL_COLUMNS, BASE_NUMERIC_COLUMNS + groups[strategy]


@dataclass(frozen=True)
class TrainingConfig:
    """All user-controlled settings for one reproducible training run."""

    model_name: ModelName
    data_dir: Path
    output_dir: Path = Path("experiments")
    sampling_strategy: SamplingStrategy = "none"
    feature_strategy: FeatureStrategy = "all"
    shap: bool = False
    feature_importance: bool = False
    validation_fraction: float = 0.2
    n_iter: int = 12
    random_state: int = 42

    def validate(self) -> None:
        if self.model_name not in MODEL_NAMES:
            raise ValueError(f"Unknown model {self.model_name!r}; choose from {MODEL_NAMES}")
        if self.sampling_strategy not in SAMPLING_STRATEGIES:
            raise ValueError(f"Unknown sampling strategy {self.sampling_strategy!r}")
        if self.feature_strategy not in FEATURE_STRATEGIES:
            raise ValueError(f"Unknown feature strategy {self.feature_strategy!r}")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1.")
        if self.n_iter < 1:
            raise ValueError("n_iter must be at least 1.")
