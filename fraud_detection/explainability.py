from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import shap
from scipy import sparse

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_explanations(
    fitted_pipeline: Any,
    features: pd.DataFrame,
    output_dir: Path,
    max_shap_rows: int = 1_000,
    include_shap: bool = True,
    include_feature_importance: bool = True,
) -> None:
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    model = fitted_pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    if include_feature_importance:
        native_importance = _native_feature_importance(model)
        if native_importance is None:
            raise TypeError(
                f"{type(model).__name__} does not expose feature_importances_"
            )
        native_frame = _importance_frame(feature_names, native_importance)
        native_frame.to_csv(output_dir / "feature_importance.csv", index=False)
        _save_bar_plot(
            native_frame,
            output_dir / "feature_importance.png",
            "Model feature importance",
        )

    if not include_shap:
        return
    sample = features.sample(n=min(max_shap_rows, len(features)), random_state=42)
    transformed = preprocessor.transform(sample)
    dense_sample = transformed.toarray() if sparse.issparse(transformed) else transformed
    explanation = shap.TreeExplainer(model)(dense_sample)
    shap_values = _binary_class_values(np.asarray(explanation.values))

    shap_frame = _importance_frame(feature_names, np.abs(shap_values).mean(axis=0))
    shap_frame.to_csv(output_dir / "shap_importance.csv", index=False)
    shap.summary_plot(
        shap_values,
        dense_sample,
        feature_names=feature_names,
        max_display=25,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()


def _native_feature_importance(model: Any) -> np.ndarray | None:
    importance = getattr(model, "feature_importances_", None)
    if importance is not None:
        return np.asarray(importance)

    ensemble_importances = []
    for estimator in getattr(model, "estimators_", []):
        fitted_classifier = getattr(estimator, "named_steps", {}).get("classifier")
        estimator_importance = getattr(
            fitted_classifier, "feature_importances_", None
        )
        if estimator_importance is not None:
            ensemble_importances.append(np.asarray(estimator_importance))
    if ensemble_importances:
        return np.mean(ensemble_importances, axis=0)
    return None


def _binary_class_values(values: np.ndarray) -> np.ndarray:
    if values.ndim == 3:
        return values[:, :, -1]
    if values.ndim != 2:
        raise ValueError(f"Unexpected SHAP value shape: {values.shape}")
    return values


def _importance_frame(names: np.ndarray, values: np.ndarray) -> pd.DataFrame:
    return (
        pd.DataFrame({"feature": names, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _save_bar_plot(frame: pd.DataFrame, path: Path, title: str) -> None:
    top = frame.head(25).sort_values("importance")
    figure, axis = plt.subplots(figsize=(10, 8))
    axis.barh(top["feature"], top["importance"])
    axis.set_title(title)
    axis.set_xlabel("Importance")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
