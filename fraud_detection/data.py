from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraud_detection.config import TIME_COLUMN


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create row-level features that do not depend on future transactions."""
    result = df.copy()
    transaction_time = pd.to_datetime(result[TIME_COLUMN], errors="raise")
    date_of_birth = pd.to_datetime(result["dob"], errors="raise")
    result[TIME_COLUMN] = transaction_time
    result["dob"] = date_of_birth

    birthday_not_reached = (
        (transaction_time.dt.month < date_of_birth.dt.month)
        | (
            (transaction_time.dt.month == date_of_birth.dt.month)
            & (transaction_time.dt.day < date_of_birth.dt.day)
        )
    )
    result["age"] = (
        transaction_time.dt.year
        - date_of_birth.dt.year
        - birthday_not_reached.astype(int)
    )
    result["hour"] = transaction_time.dt.hour
    result["day_of_week"] = transaction_time.dt.dayofweek
    result["month"] = transaction_time.dt.month
    result["is_weekend"] = (result["day_of_week"] >= 5).astype(np.int8)
    result["log_amount"] = np.log1p(result["amt"])
    result["is_round_amount"] = np.isclose(result["amt"] % 1, 0).astype(np.int8)
    result["distance_km"] = _haversine_distance(result)
    result["likely_different_state"] = (result["distance_km"] > 321.869).astype(
        np.int8
    )
    return result


def _haversine_distance(df: pd.DataFrame) -> pd.Series:
    """Calculate customer-to-merchant distance in kilometres."""
    customer_latitude = np.radians(df["lat"])
    merchant_latitude = np.radians(df["merch_lat"])
    latitude_delta = merchant_latitude - customer_latitude
    longitude_delta = np.radians(df["merch_long"] - df["long"])
    haversine = (
        np.sin(latitude_delta / 2) ** 2
        + np.cos(customer_latitude)
        * np.cos(merchant_latitude)
        * np.sin(longitude_delta / 2) ** 2
    )
    return 2 * 6371.0088 * np.arcsin(np.sqrt(haversine))


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load chronological train/test files and enforce their time boundary."""
    train_path = data_dir / "fraudTrain.csv"
    test_path = data_dir / "fraudTest.csv"
    missing = [str(path) for path in (train_path, test_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing dataset file(s): {', '.join(missing)}")

    train_df = _read_dataset(train_path)
    test_df = _read_dataset(test_path)
    if train_df[TIME_COLUMN].max() >= test_df[TIME_COLUMN].min():
        raise ValueError(
            "The final test data must start after the training data; the files overlap."
        )
    return add_historical_features(train_df, test_df)


def _read_dataset(path: Path) -> pd.DataFrame:
    frame = add_features(pd.read_csv(path, index_col=0))
    return frame.sort_values(TIME_COLUMN).reset_index(drop=True)


def add_historical_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add causal cardholder features using only each transaction's past.

    Train and test are joined only so later test transactions can use history
    that would exist in an online system. No target values are used.
    """
    train = train_df.assign(_dataset="train")
    test = test_df.assign(_dataset="test")
    combined = (
        pd.concat([train, test], ignore_index=True)
        .sort_values(["cc_num", TIME_COLUMN])
        .reset_index(drop=True)
    )
    by_card = combined.groupby("cc_num", sort=False)

    previous_time = by_card[TIME_COLUMN].shift()
    previous_amount = by_card["amt"].shift()
    previous_latitude = by_card["merch_lat"].shift()
    previous_longitude = by_card["merch_long"].shift()
    first_time = by_card[TIME_COLUMN].transform("min")

    combined["card_transaction_number"] = by_card.cumcount()
    combined["card_age_days"] = (
        (combined[TIME_COLUMN] - first_time).dt.total_seconds() / 86_400
    )
    combined["hours_since_previous_transaction"] = (
        (combined[TIME_COLUMN] - previous_time).dt.total_seconds() / 3_600
    ).fillna(-1)
    combined["distance_from_previous_transaction_km"] = _distance_between_points(
        previous_latitude,
        previous_longitude,
        combined["merch_lat"],
        combined["merch_long"],
    ).fillna(0)
    combined["amount_vs_previous_ratio"] = (
        combined["amt"] / previous_amount.clip(lower=0.01)
    ).replace([np.inf, -np.inf], np.nan).fillna(1)

    prior_count = combined["card_transaction_number"]
    prior_amount_sum = by_card["amt"].cumsum() - combined["amt"]
    prior_amount_mean = prior_amount_sum / prior_count.replace(0, np.nan)
    combined["amount_vs_historical_mean_ratio"] = (
        combined["amt"] / prior_amount_mean.clip(lower=0.01)
    ).replace([np.inf, -np.inf], np.nan).fillna(1)

    combined = combined.sort_values(TIME_COLUMN).reset_index(drop=True)
    train_out = combined.loc[combined["_dataset"] == "train"].drop(columns="_dataset")
    test_out = combined.loc[combined["_dataset"] == "test"].drop(columns="_dataset")
    return train_out.reset_index(drop=True), test_out.reset_index(drop=True)


def _distance_between_points(
    latitude_1: pd.Series,
    longitude_1: pd.Series,
    latitude_2: pd.Series,
    longitude_2: pd.Series,
) -> pd.Series:
    lat1 = np.radians(latitude_1)
    lat2 = np.radians(latitude_2)
    latitude_delta = lat2 - lat1
    longitude_delta = np.radians(longitude_2 - longitude_1)
    haversine = (
        np.sin(latitude_delta / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(longitude_delta / 2) ** 2
    )
    return 2 * 6371.0088 * np.arcsin(np.sqrt(haversine))


def build_preprocessor(
    categorical_columns: Sequence[str], numeric_columns: Sequence[str]
) -> ColumnTransformer:
    """Build preprocessing fitted inside CV to prevent validation leakage."""
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", dtype=np.float32),
                list(categorical_columns),
            ),
            ("numeric", StandardScaler(), list(numeric_columns)),
        ],
        remainder="drop",
    )
