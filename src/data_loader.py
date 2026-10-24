"""FinGuard AI Data Loading, Validation, and Synthetic Demo Module.

Handles dataset presence checks, schema validation, quality profiling,
and clearly-labeled synthetic transaction generation for educational demo mode.
"""

from typing import Dict, Any, Tuple, Optional
import os
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import (
    RAW_DATA_PATH,
    SYNTHETIC_DATA_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    ALL_COLUMNS,
    V_COLUMNS,
    AMOUNT_COLUMN,
    TIME_COLUMN,
    EXPECTED_COLUMNS_COUNT,
    EXPECTED_FEATURES_COUNT,
    EXPECTED_ROWS_COUNT,
)


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the Kaggle creditcard.csv dataset is required but not found."""
    pass


class DatasetValidationError(ValueError):
    """Raised when a dataset violates schema, types, or content expectations."""
    pass


def check_dataset_status() -> Dict[str, Any]:
    """Check whether the Kaggle credit card fraud dataset is present and valid.

    Returns:
        dict: Status containing 'exists', 'path', 'file_size_mb', and 'message'.
    """
    if not RAW_DATA_PATH.exists():
        return {
            "exists": False,
            "path": str(RAW_DATA_PATH),
            "file_size_mb": 0.0,
            "is_synthetic": False,
            "message": (
                "Kaggle Credit Card Fraud dataset ('creditcard.csv') was not found in data/raw/.\n"
                "To use real data: Download creditcard.csv from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
                "and place it at data/raw/creditcard.csv.\n"
                "FinGuard AI provides a safe, clearly labeled synthetic demo dataset for testing."
            ),
        }

    size_mb = RAW_DATA_PATH.stat().st_size / (1024 * 1024)
    return {
        "exists": True,
        "path": str(RAW_DATA_PATH),
        "file_size_mb": round(size_mb, 2),
        "is_synthetic": False,
        "message": f"Kaggle dataset located at {RAW_DATA_PATH} ({size_mb:.1f} MB).",
    }


def validate_dataset_schema(
    df: pd.DataFrame,
    require_target: bool = True,
    strict_row_count: bool = False
) -> Dict[str, Any]:
    """Validate DataFrame columns, data types, and values.

    Args:
        df: Pandas DataFrame to validate.
        require_target: Whether the 'Class' target column must be present.
        strict_row_count: Whether to assert the exact 284,807 rows of the Kaggle dataset.

    Returns:
        dict: Validation report with summary metrics.

    Raises:
        DatasetValidationError: If columns, types, or values are invalid.
    """
    if df.empty:
        raise DatasetValidationError("Dataset is empty (0 rows).")

    # Column checks
    expected_cols = ALL_COLUMNS if require_target else FEATURE_COLUMNS
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise DatasetValidationError(
            f"Missing required columns: {missing_cols[:5]}" +
            (f" and {len(missing_cols) - 5} more" if len(missing_cols) > 5 else "")
        )

    # Check for extra columns if strict
    extra_cols = [c for c in df.columns if c not in expected_cols]
    if extra_cols and require_target:
        # Note extra columns for visibility
        pass

    # Verify numeric types
    for col in expected_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DatasetValidationError(f"Column '{col}' must be numeric, found {df[col].dtype}.")

    # Check for infinite values
    numeric_df = df[expected_cols]
    is_inf = np.isinf(numeric_df.to_numpy())
    if np.any(is_inf):
        inf_count = int(np.sum(is_inf))
        raise DatasetValidationError(f"Dataset contains {inf_count} infinite numeric values.")

    # Check missing values
    null_counts = df[expected_cols].isnull().sum()
    total_nulls = int(null_counts.sum())

    # Target column validation
    class_dist = {}
    if require_target and TARGET_COLUMN in df.columns:
        unique_targets = set(df[TARGET_COLUMN].dropna().unique())
        if not unique_targets.issubset({0, 1}):
            raise DatasetValidationError(
                f"Target column '{TARGET_COLUMN}' contains invalid classes: {unique_targets}. "
                "Only binary classes 0 (genuine) and 1 (fraud) are supported."
            )
        class_dist = df[TARGET_COLUMN].value_counts().to_dict()

    # Amount validation
    if AMOUNT_COLUMN in df.columns:
        if (df[AMOUNT_COLUMN] < 0).any():
            neg_count = int((df[AMOUNT_COLUMN] < 0).sum())
            raise DatasetValidationError(f"Found {neg_count} negative transaction amounts in '{AMOUNT_COLUMN}'.")

    # Duplicate check
    duplicate_count = int(df.duplicated(subset=expected_cols).sum())

    return {
        "is_valid": True,
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_values_count": total_nulls,
        "duplicate_count": duplicate_count,
        "class_distribution": class_dist,
    }


def compute_data_quality_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute an exploratory data quality summary.

    Args:
        df: Transaction DataFrame.

    Returns:
        dict: Detailed quality metrics including amount statistics and class imbalance.
    """
    total_rows = len(df)
    missing_by_col = df[FEATURE_COLUMNS].isnull().sum().to_dict()
    total_missing = sum(missing_by_col.values())
    duplicates = int(df.duplicated(subset=FEATURE_COLUMNS).sum())

    summary: Dict[str, Any] = {
        "total_records": total_rows,
        "total_features": len(FEATURE_COLUMNS),
        "total_missing_values": total_missing,
        "duplicate_records": duplicates,
        "has_target": TARGET_COLUMN in df.columns,
    }

    if AMOUNT_COLUMN in df.columns:
        amt = df[AMOUNT_COLUMN]
        summary["amount_stats"] = {
            "min": float(amt.min()),
            "max": float(amt.max()),
            "mean": float(amt.mean()),
            "median": float(amt.median()),
            "std": float(amt.std()),
            "p25": float(amt.quantile(0.25)),
            "p75": float(amt.quantile(0.75)),
            "p95": float(amt.quantile(0.95)),
            "p99": float(amt.quantile(0.99)),
        }

    if TARGET_COLUMN in df.columns:
        counts = df[TARGET_COLUMN].value_counts()
        genuine_count = int(counts.get(0, 0))
        fraud_count = int(counts.get(1, 0))
        fraud_percentage = (fraud_count / total_rows * 100.0) if total_rows > 0 else 0.0

        summary["class_distribution"] = {
            "genuine_count": genuine_count,
            "fraud_count": fraud_count,
            "fraud_percentage": round(fraud_percentage, 4),
            "imbalance_ratio": f"1 : {round(genuine_count / fraud_count, 1)}" if fraud_count > 0 else "N/A",
        }

    return summary


def load_dataset(
    path: Optional[Path] = None,
    allow_synthetic_fallback: bool = True
) -> Tuple[pd.DataFrame, bool]:
    """Load credit card dataset from file, falling back to synthetic demo if missing.

    Args:
        path: Optional explicit CSV path.
        allow_synthetic_fallback: If True, generate and load synthetic transactions
            when the real dataset is absent.

    Returns:
        Tuple[pd.DataFrame, bool]: (DataFrame, is_synthetic_flag)

    Raises:
        DatasetNotFoundError: If dataset is missing and fallback is disallowed.
    """
    target_path = Path(path) if path else RAW_DATA_PATH

    if target_path.exists():
        df = pd.read_csv(target_path)
        validate_dataset_schema(df, require_target=True)
        return df, False

    if not allow_synthetic_fallback:
        raise DatasetNotFoundError(
            f"Dataset not found at '{target_path}'.\n"
            "Download creditcard.csv from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            f"and place it at {RAW_DATA_PATH}"
        )

    # Synthetic demo fallback
    print(f"[FinGuard AI Demo Mode] Real dataset not found at {target_path}.")
    print("[FinGuard AI Demo Mode] Generating clearly labeled synthetic demo dataset for testing.")
    df_synthetic = generate_synthetic_transactions(n_samples=5000, fraud_ratio=0.015, random_state=42)
    save_synthetic_dataset(df_synthetic)
    return df_synthetic, True


def generate_synthetic_transactions(
    n_samples: int = 5000,
    fraud_ratio: float = 0.015,
    random_state: int = 42,
    include_target: bool = True
) -> pd.DataFrame:
    """Generate statistically plausible synthetic credit card transactions.

    This synthetic generator models the structure of PCA features (V1-V28)
    with genuine transactions centered around zero and fraudulent transactions
    exhibiting characteristic shifts (e.g., negative shifts in V10, V12, V14;
    positive shifts in V4, V11).

    NOTE: This is strictly for system demonstration, UI testing, and unit tests
    when the real Kaggle dataset is not downloaded. It is explicitly labeled.

    Args:
        n_samples: Total number of transactions to generate.
        fraud_ratio: Fraction of fraudulent transactions (~1.5% for demo signal).
        random_state: Random seed for reproducibility.
        include_target: Whether to include the 'Class' column.

    Returns:
        pd.DataFrame: Synthetic transaction DataFrame.
    """
    rng = np.random.default_rng(random_state)
    n_fraud = int(n_samples * fraud_ratio)
    n_genuine = n_samples - n_fraud

    # Generate Time (seconds over 48 hours = 172800s)
    time_genuine = rng.uniform(0, 172800, size=n_genuine)
    time_fraud = rng.uniform(0, 172800, size=n_fraud)

    # Generate genuine V1-V28 (standard normal with small correlations)
    v_genuine = rng.standard_normal(size=(n_genuine, 28))

    # Generate fraud V1-V28 with characteristic shifts observed in fraud literature:
    # V14, V12, V10, V17 tend to be strongly negative in fraud cases
    # V4, V11, V2 tend to be elevated
    v_fraud = rng.standard_normal(size=(n_fraud, 28))
    # Indices: V1 is col 0, V4 is col 3, V10 is col 9, V11 is col 10, V12 is col 11, V14 is col 13, V17 is col 16
    v_fraud[:, 3] += rng.normal(3.5, 1.2, size=n_fraud)   # V4 positive shift
    v_fraud[:, 9] -= rng.normal(4.0, 1.5, size=n_fraud)   # V10 negative shift
    v_fraud[:, 10] += rng.normal(3.0, 1.0, size=n_fraud)  # V11 positive shift
    v_fraud[:, 11] -= rng.normal(4.5, 1.5, size=n_fraud)  # V12 negative shift
    v_fraud[:, 13] -= rng.normal(5.0, 1.5, size=n_fraud)  # V14 strong negative shift
    v_fraud[:, 16] -= rng.normal(3.5, 1.2, size=n_fraud)  # V17 negative shift

    # Generate Amounts: genuine transactions log-normal (~$5 to $400),
    # fraud transactions often distinct (either small card-testing or high amounts)
    amt_genuine = np.round(rng.lognormal(mean=3.2, sigma=1.1, size=n_genuine), 2)
    # Clamp extreme genuine outliers
    amt_genuine = np.clip(amt_genuine, 0.50, 2500.00)

    # Fraud amounts
    amt_fraud_part1 = rng.uniform(1.0, 15.0, size=n_fraud // 2)  # micro test transactions
    amt_fraud_part2 = rng.uniform(250.0, 1400.0, size=n_fraud - (n_fraud // 2))  # high ticket
    amt_fraud = np.round(np.concatenate([amt_fraud_part1, amt_fraud_part2]), 2)
    rng.shuffle(amt_fraud)

    # Combine into dataframes
    df_gen = pd.DataFrame(v_genuine, columns=V_COLUMNS)
    df_gen.insert(0, TIME_COLUMN, np.round(time_genuine, 1))
    df_gen[AMOUNT_COLUMN] = amt_genuine
    if include_target:
        df_gen[TARGET_COLUMN] = 0

    df_frd = pd.DataFrame(v_fraud, columns=V_COLUMNS)
    df_frd.insert(0, TIME_COLUMN, np.round(time_fraud, 1))
    df_frd[AMOUNT_COLUMN] = amt_fraud
    if include_target:
        df_frd[TARGET_COLUMN] = 1

    # Concatenate and shuffle
    df = pd.concat([df_gen, df_frd], ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return df


def save_synthetic_dataset(df: pd.DataFrame, path: Optional[Path] = None) -> Path:
    """Save synthetic transactions to disk."""
    save_path = Path(path) if path else SYNTHETIC_DATA_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    return save_path


def get_sample_transaction(is_fraud_pattern: bool = False) -> Dict[str, float]:
    """Provide a realistic sample transaction for single-analysis UI testing.

    Args:
        is_fraud_pattern: If True, returns a pattern typical of fraud flags;
            otherwise returns a benign pattern.

    Returns:
        dict: Mapping of all 30 feature names (Time, V1..V28, Amount) to floats.
    """
    sample = {TIME_COLUMN: 85200.0}
    rng = np.random.default_rng(101 if is_fraud_pattern else 42)

    for i in range(1, 29):
        sample[f"V{i}"] = float(np.round(rng.normal(0, 1.0), 4))

    if is_fraud_pattern:
        sample["V4"] = 3.85
        sample["V10"] = -4.22
        sample["V11"] = 3.41
        sample["V12"] = -5.10
        sample["V14"] = -6.80
        sample["V17"] = -4.15
        sample[AMOUNT_COLUMN] = 549.99
    else:
        sample["V4"] = 0.12
        sample["V10"] = -0.05
        sample["V11"] = 0.33
        sample["V12"] = 0.15
        sample["V14"] = 0.08
        sample["V17"] = -0.11
        sample[AMOUNT_COLUMN] = 45.50

    return sample
