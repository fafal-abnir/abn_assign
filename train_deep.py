import argparse
from pathlib import Path
from typing import Sequence, cast

from deep_learning.config import DeepModelName, DeepTrainingConfig, LossName
from fraud_detection.config import FEATURE_STRATEGIES, FeatureStrategy
from deep_learning.training import train_and_evaluate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a time-validated deep tabular fraud classifier."
    )
    parser.add_argument(
        "--model", required=True, choices=("ft_transformer", "tabm")
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument(
        "--feature-strategy", choices=FEATURE_STRATEGIES, default="all"
    )
    parser.add_argument(
        "--loss",
        choices=("weighted_bce", "focal", "weighted_bce+deviation"),
        default="weighted_bce",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--d-token", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--tabm-k", type=int, default=16)
    parser.add_argument("--focal-alpha", type=float, default=0.95)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--deviation-weight",
        type=float,
        default=1.0,
        help="Lambda multiplying deviation loss in the combined objective.",
    )
    parser.add_argument("--deviation-margin", type=float, default=5.0)
    parser.add_argument("--deviation-reference-samples", type=int, default=5_000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"))
    return parser


def parse_config(arguments: Sequence[str] | None = None) -> DeepTrainingConfig:
    args = build_parser().parse_args(arguments)
    return DeepTrainingConfig(
        model_name=cast(DeepModelName, args.model),
        data_dir=args.data_dir,
        feature_strategy=cast(FeatureStrategy, args.feature_strategy),
        loss_name=cast(LossName, args.loss),
        validation_fraction=args.validation_fraction,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        d_token=args.d_token,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        tabm_k=args.tabm_k,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        deviation_weight=args.deviation_weight,
        deviation_margin=args.deviation_margin,
        deviation_reference_samples=args.deviation_reference_samples,
        num_workers=args.num_workers,
        random_state=args.random_state,
        output_dir=args.output_dir,
    )


def main() -> None:
    train_and_evaluate(parse_config())


if __name__ == "__main__":
    main()
