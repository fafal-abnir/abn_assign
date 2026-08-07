
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import lightning as L
import pandas as pd
from lightning.pytorch.callbacks import Callback, EarlyStopping

from deep_learning.config import DeepTrainingConfig
from deep_learning.data import DeepPreprocessor, FraudDataModule
from deep_learning.models import create_deep_model
from deep_learning.module import FraudLightningModule
from fraud_detection.config import TARGET, TIME_COLUMN, feature_columns
from fraud_detection.data import load_data


class ValidationEpochSelector(Callback):

    def __init__(self) -> None:
        self.best_score = float("-inf")
        self.best_epoch = 0

    def on_validation_epoch_end(self, trainer: L.Trainer, module: L.LightningModule) -> None:
        metric = trainer.callback_metrics.get("val_ap")
        if metric is not None and float(metric) > self.best_score:
            self.best_score = float(metric)
            self.best_epoch = trainer.current_epoch + 1


def train_and_evaluate(config: DeepTrainingConfig) -> dict[str, Any]:
    """Select epoch count on validation, retrain on all train, then test once."""
    config.validate()
    L.seed_everything(config.random_state, workers=True)
    train_df, test_df = load_data(config.data_dir)
    categorical_columns, numeric_columns = feature_columns(config.feature_strategy)
    split_index = int(len(train_df) * (1 - config.validation_fraction))
    fitting_df = train_df.iloc[:split_index]
    validation_df = train_df.iloc[split_index:]

    selection_preprocessor = DeepPreprocessor(
        categorical_columns, numeric_columns
    ).fit(fitting_df)
    selection_data = FraudDataModule(
        selection_preprocessor.transform(fitting_df, TARGET),
        selection_preprocessor.transform(validation_df, TARGET),
        None,
        config.batch_size,
        config.num_workers,
    )
    positive_weight = _positive_weight(fitting_df[TARGET])
    selection_module = _make_module(
        config, selection_preprocessor, positive_weight
    )
    best_state = ValidationEpochSelector()
    early_stopping = EarlyStopping(
        monitor="val_ap", mode="max", patience=config.patience
    )
    selection_trainer = _trainer(
        config, config.max_epochs, [best_state, early_stopping]
    )
    selection_trainer.fit(selection_module, datamodule=selection_data)
    if best_state.best_epoch == 0:
        raise RuntimeError("Training completed without a validation checkpoint.")

    # Refit preprocessing and a fresh model on every fraudTrain row for the
    # validation-selected number of epochs. The test set is still untouched.
    final_preprocessor = DeepPreprocessor(
        categorical_columns, numeric_columns
    ).fit(train_df)
    final_data = FraudDataModule(
        final_preprocessor.transform(train_df, TARGET),
        None,
        final_preprocessor.transform(test_df, TARGET),
        config.batch_size,
        config.num_workers,
    )
    L.seed_everything(config.random_state, workers=True)
    final_module = _make_module(
        config, final_preprocessor, _positive_weight(train_df[TARGET])
    )
    final_trainer = _trainer(config, max(1, best_state.best_epoch), [])
    final_trainer.fit(final_module, datamodule=final_data)
    test_results = final_trainer.test(final_module, datamodule=final_data, verbose=False)[0]

    metrics = {
        "model": config.model_name,
        "feature_strategy": config.feature_strategy,
        "loss": config.loss_name,
        "best_validation_average_precision": best_state.best_score,
        "selected_epochs": best_state.best_epoch,
        "test_average_precision": float(test_results["test_ap"]),
        "test_roc_auc": float(test_results["test_auc"]),
        "validation_start": train_df.loc[split_index, TIME_COLUMN].isoformat(),
        "test_start": test_df[TIME_COLUMN].min().isoformat(),
        "config": config.serializable(),
    }
    output_dir = _experiment_directory(config)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"Experiment directory: {output_dir}")
    return metrics


def _make_module(
    config: DeepTrainingConfig,
    preprocessor: DeepPreprocessor,
    positive_weight: float,
) -> FraudLightningModule:
    model = create_deep_model(
        config,
        len(preprocessor.numeric_columns),
        preprocessor.cardinalities,
    )
    return FraudLightningModule(model, config, positive_weight)


def _trainer(
    config: DeepTrainingConfig, max_epochs: int, callbacks: list[Callback]
) -> L.Trainer:
    return L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        deterministic=True,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        logger=False,
        log_every_n_steps=50,
    )


def _positive_weight(target: pd.Series) -> float:
    positives = int(target.sum())
    if positives == 0:
        raise ValueError("The training period contains no fraud observations.")
    return float((target == 0).sum() / positives)


def _experiment_directory(config: DeepTrainingConfig) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f_%z")
    directory = config.output_dir / "deep" / config.model_name / timestamp
    directory.mkdir(parents=True, exist_ok=False)
    return directory
