import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV

from fraud_detection.config import (
    TARGET,
    TIME_COLUMN,
    TrainingConfig,
    feature_columns,
)
from fraud_detection.data import build_preprocessor, load_data
from fraud_detection.explainability import save_explanations
from fraud_detection.models import create_model


def train_and_evaluate(config: TrainingConfig) -> dict[str, Any]:
    config.validate()
    train_df, test_df = load_data(config.data_dir)
    categorical_columns, numeric_columns = feature_columns(config.feature_strategy)
    selected_columns = categorical_columns + numeric_columns
    X_train, y_train = train_df[list(selected_columns)], train_df[TARGET]
    X_test, y_test = test_df[list(selected_columns)], test_df[TARGET]

    time_split, split_index = make_latest_holdout(
        len(train_df), config.validation_fraction
    )
    estimator, search_space = create_model(
        config.model_name,
        y_train.iloc[:split_index],
        config.random_state,
        config.sampling_strategy,
    )
    pipeline = Pipeline(
        [
            (
                "preprocessor",
                build_preprocessor(categorical_columns, numeric_columns),
            ),
            ("sampler", create_sampler(config)),
            ("model", estimator),
        ]
    )
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=search_space,
        n_iter=config.n_iter,
        scoring="average_precision",
        cv=time_split,
        refit=True,
        random_state=config.random_state,
        n_jobs=1,
        verbose=2,
        return_train_score=False,
    )

    validation_start = train_df.loc[split_index, TIME_COLUMN]
    _print_split_summary(config, train_df, test_df, split_index, validation_start)
    search.fit(X_train, y_train)

    probabilities = search.best_estimator_.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = _build_metrics(
        config, search, y_test, probabilities, validation_start, test_df
    )
    experiment_dir = create_experiment_directory(config)
    _save_outputs(
        experiment_dir,
        metrics,
    )
    if config.shap or config.feature_importance:
        save_explanations(
            search.best_estimator_,
            X_test,
            experiment_dir,
            include_shap=config.shap,
            include_feature_importance=config.feature_importance,
        )
    _print_results(metrics, y_test, predictions)
    print(f"Experiment directory: {experiment_dir}")
    return metrics


def create_sampler(config: TrainingConfig) -> Any:
    samplers = {
        "none": "passthrough",
        "random_under": RandomUnderSampler(random_state=config.random_state),
        "random_over": RandomOverSampler(random_state=config.random_state),
        "smote": SMOTE(random_state=config.random_state),
    }
    return samplers[config.sampling_strategy]


def create_experiment_directory(config: TrainingConfig) -> Path:
    """Create experiments/<model>/<local timestamp>/ for this run."""
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f_%z")
    experiment_dir = config.output_dir / config.model_name / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=False)
    return experiment_dir


def make_latest_holdout(
    n_rows: int, validation_fraction: float
) -> tuple[PredefinedSplit, int]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    split_index = int(n_rows * (1 - validation_fraction))
    if split_index < 1 or split_index >= n_rows:
        raise ValueError("validation_fraction leaves an empty train or validation set.")
    fold = np.full(n_rows, -1, dtype=int)
    fold[split_index:] = 0
    return PredefinedSplit(fold), split_index


def _build_metrics(
    config: TrainingConfig,
    search: RandomizedSearchCV,
    actual: pd.Series,
    probabilities: np.ndarray,
    validation_start: pd.Timestamp,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "model": config.model_name,
        "sampling_strategy": config.sampling_strategy,
        "feature_strategy": config.feature_strategy,
        "shap": config.shap,
        "feature_importance": config.feature_importance,
        "best_validation_average_precision": float(search.best_score_),
        "test_roc_auc": float(roc_auc_score(actual, probabilities)),
        "test_average_precision": float(average_precision_score(actual, probabilities)),
        "best_params": search.best_params_,
        "validation_start": validation_start.isoformat(),
        "test_start": test_df[TIME_COLUMN].min().isoformat(),
    }


def _save_outputs(
    experiment_dir: Path,
    metrics: dict[str, Any],
) -> None:
    (experiment_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )


def _print_split_summary(
    config: TrainingConfig,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_index: int,
    validation_start: pd.Timestamp,
) -> None:
    print(
        f"Model: {config.model_name}\n"
        f"Features: {config.feature_strategy} | Sampling: {config.sampling_strategy}\n"
        f"Tuning train: {split_index:,} rows\n"
        f"Tuning validation: {len(train_df) - split_index:,} latest rows "
        f"(starting {validation_start})\n"
        f"Final test: {len(test_df):,} untouched rows"
    )


def _print_results(
    metrics: dict[str, Any], actual: pd.Series, predictions: np.ndarray
) -> None:
    print("\nBest parameters:", json.dumps(metrics["best_params"], indent=2))
    print("Test ROC-AUC:", f"{metrics['test_roc_auc']:.6f}")
    print("Test average precision:", f"{metrics['test_average_precision']:.6f}")
    print("\n", classification_report(actual, predictions, digits=4))
