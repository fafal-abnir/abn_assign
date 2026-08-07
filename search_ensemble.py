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
)
from fraud_detection.search import SearchConfig, search_and_evaluate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Search models, feature groups, sampling methods, and hyperparameters "
            "on a time-based validation set; evaluate only the winner on test data."
        )
    )
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES)
    )
    parser.add_argument(
        "--feature-strategies",
        nargs="+",
        choices=FEATURE_STRATEGIES,
        default=list(FEATURE_STRATEGIES),
    )
    parser.add_argument(
        "--sampling-strategies",
        nargs="+",
        choices=SAMPLING_STRATEGIES,
        default=list(SAMPLING_STRATEGIES),
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--n-iter",
        type=int,
        default=10,
        help="Hyperparameter combinations tried per complete configuration.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--shap", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--feature-importance",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"))
    return parser


def parse_config(arguments: Sequence[str] | None = None) -> SearchConfig:
    args = build_parser().parse_args(arguments)
    return SearchConfig(
        models=tuple(cast(list[ModelName], args.models)),
        feature_strategies=tuple(
            cast(list[FeatureStrategy], args.feature_strategies)
        ),
        sampling_strategies=tuple(
            cast(list[SamplingStrategy], args.sampling_strategies)
        ),
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        validation_fraction=args.validation_fraction,
        n_iter=args.n_iter,
        random_state=args.random_state,
        shap=args.shap,
        feature_importance=args.feature_importance,
    )


def main() -> None:
    search_and_evaluate(parse_config())


if __name__ == "__main__":
    main()
