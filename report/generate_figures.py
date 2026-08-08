"""Generate report figures from timestamped experiment metrics."""

import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
FIGURES = Path(__file__).resolve().parent / "figures"
SCORE = "validation_average_precision"
MODEL_LABELS = {
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "lightgbm": "LightGBM",
    "random_forest": "Random Forest",
    "easyensemble": "EasyEnsemble",
    "rusboost": "RUSBoost",
    "ft_transformer": "FT-Transformer",
    "tabm": "TabM",
}


def load_results() -> pd.DataFrame:
    rows = []
    for path in EXPERIMENTS.rglob("metrics.json"):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        relative = path.relative_to(EXPERIMENTS)
        rows.append(
            {
                "family": "Deep learning" if relative.parts[0] == "deep" else "Ensemble",
                "model": metrics.get("model", metrics.get("winning_model")),
                "feature_strategy": metrics.get(
                    "feature_strategy", metrics.get("winning_feature_strategy")
                ),
                "sampling_strategy": metrics.get(
                    "sampling_strategy", metrics.get("winning_sampling_strategy")
                ),
                "loss": metrics.get("loss"),
                SCORE: metrics.get("best_validation_average_precision"),
                "test_average_precision": metrics.get("test_average_precision"),
                "test_roc_auc": metrics.get("test_roc_auc"),
            }
        )
    frame = pd.DataFrame(rows).dropna(subset=[SCORE])
    frame["model_label"] = frame["model"].map(MODEL_LABELS).fillna(frame["model"])
    return frame


def box_and_points(
    axis: plt.Axes,
    frame: pd.DataFrame,
    factor: str,
    order: list[str],
    title: str,
) -> None:
    sns.boxplot(
        data=frame,
        x=factor,
        y=SCORE,
        order=order,
        color="#c9dff2",
        showfliers=False,
        ax=axis,
    )
    sns.stripplot(
        data=frame,
        x=factor,
        y=SCORE,
        order=order,
        hue="model_label",
        jitter=0.16,
        alpha=0.82,
        size=5,
        ax=axis,
    )
    axis.set_title(title, weight="bold")
    axis.set_xlabel(factor.replace("_", " ").title())
    axis.set_ylabel("Validation average precision")
    axis.tick_params(axis="x", rotation=18)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="Model", fontsize=8, title_fontsize=8)


def ensemble_influence(results: pd.DataFrame) -> None:
    ensemble = results.loc[results["family"] == "Ensemble"]
    external = ensemble.loc[
        ~ensemble["model"].isin(["easyensemble", "rusboost"])
    ]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    box_and_points(
        axes[0],
        external.dropna(subset=["sampling_strategy"]),
        "sampling_strategy",
        ["none", "random_under", "random_over", "smote"],
        "External sampling strategy",
    )
    box_and_points(
        axes[1],
        ensemble.dropna(subset=["feature_strategy"]),
        "feature_strategy",
        ["base", "time", "geo", "behavioral", "all"],
        "Feature-engineering strategy",
    )
    figure.tight_layout()
    figure.savefig(FIGURES / "ensemble_influence.pdf", bbox_inches="tight")
    figure.savefig(FIGURES / "ensemble_influence.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def deep_influence(results: pd.DataFrame) -> None:
    deep = results.loc[results["family"] == "Deep learning"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    box_and_points(
        axes[0],
        deep,
        "feature_strategy",
        ["base", "time", "geo", "behavioral", "all"],
        "Deep models: feature strategy",
    )
    box_and_points(
        axes[1],
        deep,
        "loss",
        ["weighted_bce", "focal", "weighted_bce+deviation"],
        "Deep models: loss function",
    )
    figure.tight_layout()
    figure.savefig(FIGURES / "deep_influence.pdf", bbox_inches="tight")
    figure.savefig(FIGURES / "deep_influence.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def best_model_comparison(results: pd.DataFrame) -> None:
    best = (
        results.sort_values(SCORE, ascending=False)
        .drop_duplicates("model")
        .sort_values(SCORE, ascending=False)
    )
    long = best.melt(
        id_vars=["model_label", "family"],
        value_vars=[SCORE, "test_average_precision", "test_roc_auc"],
        var_name="metric",
        value_name="score",
    )
    labels = {
        SCORE: "Validation AP",
        "test_average_precision": "Test AP",
        "test_roc_auc": "Test ROC-AUC",
    }
    long["metric"] = long["metric"].map(labels)
    figure, axis = plt.subplots(figsize=(12.5, 5.8))
    sns.barplot(
        data=long,
        x="model_label",
        y="score",
        hue="metric",
        palette=["#457b9d", "#e76f51", "#2a9d8f"],
        ax=axis,
    )
    axis.set_ylim(0.55, 1.01)
    axis.set_xlabel("Best validation-selected configuration per model")
    axis.set_ylabel("Score")
    axis.tick_params(axis="x", rotation=18)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="")
    figure.tight_layout()
    figure.savefig(FIGURES / "best_model_comparison.pdf", bbox_inches="tight")
    figure.savefig(FIGURES / "best_model_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def validation_test_relationship(results: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 6))
    sns.scatterplot(
        data=results,
        x=SCORE,
        y="test_average_precision",
        hue="family",
        style="family",
        s=70,
        alpha=0.8,
        ax=axis,
    )
    lower = min(results[SCORE].min(), results["test_average_precision"].min())
    axis.plot([lower, 1], [lower, 1], linestyle="--", color="gray", linewidth=1)
    axis.set_xlabel("Validation average precision")
    axis.set_ylabel("Test average precision")
    axis.set_title("Validation-to-test relationship", weight="bold")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURES / "validation_test_relationship.pdf", bbox_inches="tight")
    figure.savefig(FIGURES / "validation_test_relationship.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def copy_explainability_assets() -> None:
    """Copy SHAP/native importance plots from the strongest explained run."""
    explained_runs = []
    for metrics_path in EXPERIMENTS.rglob("metrics.json"):
        run_dir = metrics_path.parent
        if not all(
            (run_dir / name).is_file()
            for name in ("shap_summary.png", "feature_importance.png")
        ):
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        explained_runs.append(
            (metrics.get("best_validation_average_precision", float("-inf")), run_dir)
        )
    if not explained_runs:
        print("No completed SHAP/native-importance experiment found; skipping.")
        return
    _, best_run = max(explained_runs, key=lambda item: item[0])
    shutil.copy2(best_run / "shap_summary.png", FIGURES / "lightgbm_shap_summary.png")
    shutil.copy2(
        best_run / "feature_importance.png",
        FIGURES / "lightgbm_feature_importance.png",
    )
    print(f"Copied explainability assets from {best_run.relative_to(ROOT)}")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    results = load_results()
    ensemble_influence(results)
    deep_influence(results)
    best_model_comparison(results)
    validation_test_relationship(results)
    copy_explainability_assets()
    print(f"Generated report figures from {len(results)} experiments in {FIGURES}")


if __name__ == "__main__":
    main()
