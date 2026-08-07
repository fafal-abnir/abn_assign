import argparse
from pathlib import Path
from typing import Sequence, cast

from fraud_detection.config import (
    FEATURE_STRATEGIES,
    MODEL_NAMES,
    SAMPLING_STRATEGIES,
    FeatureStrategy,
    ModelName,
    SamplingStrategy,
    TrainingConfig,
)
from fraud_detection.training import train_and_evaluate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune and train a fraud-detection ensemble with a time holdout."
    )
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing fraudTrain.csv and fraudTest.csv.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Latest fraction used for tuning (default: 0.2).",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=10,
        help="Number of random hyperparameter combinations (default: 12).",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--sampling",
        choices=SAMPLING_STRATEGIES,
        default="none",
        help="Class sampling applied only to training folds (default: none).",
    )
    parser.add_argument(
        "--feature-strategy",
        choices=FEATURE_STRATEGIES,
        default="base",
        help="Feature group: base, or base plus the selected group (default: all).",
    )
    parser.add_argument(
        "--shap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write SHAP importance and summary plot.",
    )
    parser.add_argument(
        "--feature-importance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write native model feature importance and plot.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"))
    return parser


def parse_config(arguments: Sequence[str] | None = None) -> TrainingConfig:
    args = build_parser().parse_args(arguments)
    return TrainingConfig(
        model_name=cast(ModelName, args.model),
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sampling_strategy=cast(SamplingStrategy, args.sampling),
        feature_strategy=cast(FeatureStrategy, args.feature_strategy),
        shap=args.shap,
        feature_importance=args.feature_importance,
        validation_fraction=args.validation_fraction,
        n_iter=args.n_iter,
        random_state=args.random_state,
    )


def main() -> None:
    train_and_evaluate(parse_config())


if __name__ == "__main__":
    main()
