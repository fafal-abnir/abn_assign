"""Leakage-safe encoding and Lightning data loaders for deep models."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class EncodedData:
    numeric: np.ndarray
    categorical: np.ndarray
    target: np.ndarray

    def dataset(self) -> TensorDataset:
        return TensorDataset(
            torch.from_numpy(self.numeric),
            torch.from_numpy(self.categorical),
            torch.from_numpy(self.target),
        )


class DeepPreprocessor:
    """Train-fitted numeric standardizer and categorical vocabulary encoder."""

    def __init__(
        self, categorical_columns: Sequence[str], numeric_columns: Sequence[str]
    ) -> None:
        self.categorical_columns = tuple(categorical_columns)
        self.numeric_columns = tuple(numeric_columns)
        self.category_maps: dict[str, dict[str, int]] = {}
        self.numeric_mean: pd.Series | None = None
        self.numeric_std: pd.Series | None = None

    def fit(self, frame: pd.DataFrame) -> "DeepPreprocessor":
        numeric = frame[list(self.numeric_columns)].astype(np.float32)
        self.numeric_mean = numeric.mean()
        self.numeric_std = numeric.std().replace(0, 1).fillna(1)
        self.category_maps = {
            column: {
                value: index + 1
                for index, value in enumerate(
                    sorted(frame[column].fillna("<missing>").astype(str).unique())
                )
            }
            for column in self.categorical_columns
        }
        return self

    @property
    def cardinalities(self) -> list[int]:
        return [len(self.category_maps[column]) + 1 for column in self.categorical_columns]

    def transform(self, frame: pd.DataFrame, target_column: str) -> EncodedData:
        if self.numeric_mean is None or self.numeric_std is None:
            raise RuntimeError("DeepPreprocessor must be fitted before transform().")
        numeric = (
            (frame[list(self.numeric_columns)] - self.numeric_mean) / self.numeric_std
        ).to_numpy(dtype=np.float32)
        categorical = np.column_stack(
            [
                frame[column]
                .fillna("<missing>")
                .astype(str)
                .map(self.category_maps[column])
                .fillna(0)
                .to_numpy(dtype=np.int64)
                for column in self.categorical_columns
            ]
        )
        target = frame[target_column].to_numpy(dtype=np.float32)
        return EncodedData(numeric, categorical, target)


class FraudDataModule(LightningDataModule):
    def __init__(
        self,
        train: EncodedData,
        validation: EncodedData | None,
        test: EncodedData | None,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.train_data = train
        self.validation_data = validation
        self.test_data = test
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _loader(self, data: EncodedData, shuffle: bool) -> DataLoader:
        return DataLoader(
            data.dataset(),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_data, shuffle=True)

    def val_dataloader(self) -> DataLoader | list[DataLoader]:
        return (
            self._loader(self.validation_data, shuffle=False)
            if self.validation_data is not None
            else []
        )

    def test_dataloader(self) -> DataLoader | list[DataLoader]:
        return (
            self._loader(self.test_data, shuffle=False)
            if self.test_data is not None
            else []
        )
