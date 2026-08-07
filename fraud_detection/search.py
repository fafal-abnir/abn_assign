import json
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

from fraud_detection.config import (
    FEATURE_STRATEGIES,
    MODEL_NAMES,
    SAMPLING_STRATEGIES,
    TARGET,
    TIME_COLUMN,
    FeatureStrategy,
    ModelName,
    SamplingStrategy,
    TrainingConfig,
    feature_columns,
)
from fraud_detection.data import build_preprocessor, load_data
from fraud_detection.explainability import save_explanations
from fraud_detection.models import create_model
from fraud_detection.training import create_sampler, make_latest_holdout


@dataclass(frozen=True)
class SearchConfig:
    models: tuple[ModelName, ...]
    feature_strategies: tuple[FeatureStrategy, ...]
    sampling_strategies: tuple[SamplingStrategy, ...]
    data_dir: Path
    output_dir: Path = Path("experiments")
    validation_fraction: float = 0.2
    n_iter: int = 10
    random_state: int = 42
    shap: bool = False
    feature_importance: bool = False

    def validate(self) -> None:
        _validate_choices("model", self.models, MODEL_NAMES)
        _validate_choices("feature strategy", self.feature_strategies, FEATURE_STRATEGIES)
        _validate_choices(
            "sampling strategy", self.sampling_strategies, SAMPLING_STRATEGIES
        )
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1.")
        if self.n_iter < 1:
            raise ValueError("n_iter must be at least 1.")


def search_and_evaluate(config: SearchConfig) -> dict[str, Any]:
    """Select the complete configuration on validation, then test it once."""
    config.validate()
    train_df, test_df = load_data(config.data_dir)
    time_split, split_index = make_latest_holdout(
        len(train_df), config.validation_fraction
    )
    candidate_values = list(
        product(
            config.models, config.feature_strategies, config.sampling_strategies
        )
    )
    print(
        f"Searching {len(candidate_values)} complete configurations; "
        f"{config.n_iter} hyperparameter combinations each.\n"
        f"Validation begins at {train_df.loc[split_index, TIME_COLUMN]}. "
        "The final test set will not be read by an estimator until selection is complete."
    )

    leaderboard_rows: list[dict[str, Any]] = []
    best_search: RandomizedSearchCV | None = None
    best_columns: tuple[str, ...] | None = None
    best_score = -np.inf

    for number, (model_name, feature_strategy, sampling_strategy) in enumerate(
        candidate_values, start=1
    ):
        print(
            f"\n[{number}/{len(candidate_values)}] model={model_name}, "
            f"features={feature_strategy}, sampling={sampling_strategy}"
        )
        categorical_columns, numeric_columns = feature_columns(feature_strategy)
        selected_columns = categorical_columns + numeric_columns
        X_train = train_df[list(selected_columns)]
        y_train = train_df[TARGET]
        estimator, parameter_space = create_model(
            model_name,
            y_train.iloc[:split_index],
            config.random_state,
            sampling_strategy,
        )
        candidate_config = TrainingConfig(
            model_name=model_name,
            data_dir=config.data_dir,
            sampling_strategy=sampling_strategy,
            feature_strategy=feature_strategy,
            random_state=config.random_state,
        )
        pipeline = Pipeline(
            [
                (
                    "preprocessor",
                    build_preprocessor(categorical_columns, numeric_columns),
                ),
                ("sampler", create_sampler(candidate_config)),
                ("model", estimator),
            ]
        )
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=parameter_space,
            n_iter=config.n_iter,
            scoring="average_precision",
            cv=time_split,
            refit=True,
            random_state=config.random_state,
            n_jobs=1,
            verbose=0,
            return_train_score=False,
        )
        search.fit(X_train, y_train)
        row = {
            "model": model_name,
            "feature_strategy": feature_strategy,
            "sampling_strategy": sampling_strategy,
            "validation_average_precision": float(search.best_score_),
            "best_params": json.dumps(search.best_params_, sort_keys=True),
        }
        leaderboard_rows.append(row)
        print(f"Validation average precision: {search.best_score_:.6f}")
        if search.best_score_ > best_score:
            best_score = float(search.best_score_)
            best_search = search
            best_columns = selected_columns

    if best_search is None or best_columns is None:
        raise RuntimeError("No candidate configuration completed.")

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        "validation_average_precision", ascending=False
    )
    winner = leaderboard.iloc[0]
    experiment_dir = _create_search_directory(config, str(winner["model"]))
    leaderboard.to_csv(experiment_dir.parent / "validation_leaderboard.csv", index=False)

    # RandomizedSearchCV refitted this winning pipeline on all fraudTrain rows.
    X_test = test_df[list(best_columns)]
    y_test = test_df[TARGET]
    probabilities = best_search.best_estimator_.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "winning_model": winner["model"],
        "winning_feature_strategy": winner["feature_strategy"],
        "winning_sampling_strategy": winner["sampling_strategy"],
        "best_validation_average_precision": best_score,
        "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
        "test_average_precision": float(average_precision_score(y_test, probabilities)),
        "best_params": best_search.best_params_,
        "validation_start": train_df.loc[split_index, TIME_COLUMN].isoformat(),
        "test_start": test_df[TIME_COLUMN].min().isoformat(),
        "candidate_count": len(candidate_values),
    }
    (experiment_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    if config.shap or config.feature_importance:
        save_explanations(
            best_search.best_estimator_,
            X_test,
            experiment_dir,
            include_shap=config.shap,
            include_feature_importance=config.feature_importance,
        )

    print("\nWinning configuration:")
    print(json.dumps(metrics, indent=2))
    print("\n", classification_report(y_test, predictions, digits=4))
    print(f"Search results: {experiment_dir.parent}")
    return metrics


def _create_search_directory(config: SearchConfig, model_name: str) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f_%z")
    directory = config.output_dir / "search" / timestamp / model_name
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _validate_choices(name: str, values: tuple[Any, ...], allowed: tuple[Any, ...]) -> None:
    if not values:
        raise ValueError(f"At least one {name} is required.")
    invalid = set(values) - set(allowed)
    if invalid:
        raise ValueError(f"Unknown {name}(s): {sorted(invalid)}")
