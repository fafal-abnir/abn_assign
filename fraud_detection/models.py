from typing import Any

import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from fraud_detection.config import MODEL_NAMES, ModelName, SamplingStrategy


SearchSpace = dict[str, list[Any]]


def create_model(
    model_name: ModelName,
    y_train: pd.Series,
    random_state: int,
    sampling_strategy: SamplingStrategy = "none",
) -> tuple[Any, SearchSpace]:
    positive_weight = (
        _positive_class_weight(y_train) if sampling_strategy == "none" else 1.0
    )
    common_space: SearchSpace = {
        "model__n_estimators": [200, 400, 700],
        "model__max_depth": [4, 6, 8, 12],
    }

    if model_name == "random_forest":
        return RandomForestClassifier(
            class_weight="balanced" if sampling_strategy == "none" else None,
            random_state=random_state,
            n_jobs=-1,
        ), {
            **common_space,
            "model__max_features": ["sqrt", "log2", 0.5],
            "model__min_samples_leaf": [1, 2, 5, 10],
        }
    if model_name == "xgboost":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=positive_weight,
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
        ), {
            **common_space,
            "model__learning_rate": [0.02, 0.05, 0.1],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
        }
    if model_name == "catboost":
        return CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="PRAUC",
            auto_class_weights="Balanced",
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        ), {
            **common_space,
            "model__learning_rate": [0.02, 0.05, 0.1],
            "model__l2_leaf_reg": [1, 3, 5, 9],
        }
    if model_name == "lightgbm":
        return LGBMClassifier(
            objective="binary",
            scale_pos_weight=positive_weight,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        ), {
            **common_space,
            "model__learning_rate": [0.02, 0.05, 0.1],
            "model__num_leaves": [15, 31, 63],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
        }
    raise ValueError(f"Unknown model {model_name!r}; choose from {MODEL_NAMES}")


def _positive_class_weight(target: pd.Series) -> float:
    positive_count = int(target.sum())
    if positive_count == 0:
        raise ValueError("Training data contains no positive fraud examples.")
    return float((target == 0).sum() / positive_count)
