"""Unit tests for dataset loading, validation, and synthetic generation."""

import pytest
import numpy as np
import pandas as pd

from src.data_loader import (
    validate_dataset_schema,
    compute_data_quality_summary,
    generate_synthetic_transactions,
    DatasetValidationError,
    check_dataset_status,
)
from src.config import FEATURE_COLUMNS, ALL_COLUMNS, TARGET_COLUMN, AMOUNT_COLUMN


def test_validate_dataset_schema_success(small_synthetic_df):
    """Test schema validation succeeds on valid synthetic data."""
    report = validate_dataset_schema(small_synthetic_df, require_target=True)
    assert report["is_valid"] is True
    assert report["row_count"] == 200
    assert report["missing_values_count"] == 0
    assert 0 in report["class_distribution"]
    assert 1 in report["class_distribution"]


def test_validate_empty_dataset_fails():
    """Empty dataframe must raise DatasetValidationError."""
    empty_df = pd.DataFrame()
    with pytest.raises(DatasetValidationError, match="empty"):
        validate_dataset_schema(empty_df)


def test_validate_missing_columns_fails(small_synthetic_df):
    """Dropping a feature must trigger validation failure."""
    df_missing = small_synthetic_df.drop(columns=["V14"])
    with pytest.raises(DatasetValidationError, match="Missing required columns"):
        validate_dataset_schema(df_missing)


def test_validate_non_numeric_column_fails(small_synthetic_df):
    """String values in a numeric feature must raise error."""
    df_bad = small_synthetic_df.copy()
    df_bad["V1"] = "invalid_string"
    with pytest.raises(DatasetValidationError, match="must be numeric"):
        validate_dataset_schema(df_bad)


def test_validate_infinite_values_fails(small_synthetic_df):
    """Infinite values must be rejected."""
    df_inf = small_synthetic_df.copy()
    df_inf.loc[0, "V5"] = np.inf
    with pytest.raises(DatasetValidationError, match="infinite numeric values"):
        validate_dataset_schema(df_inf)


def test_validate_negative_amount_fails(small_synthetic_df):
    """Negative transaction amounts must be rejected."""
    df_neg = small_synthetic_df.copy()
    df_neg.loc[0, AMOUNT_COLUMN] = -12.50
    with pytest.raises(DatasetValidationError, match="negative transaction amounts"):
        validate_dataset_schema(df_neg)


def test_validate_invalid_target_classes_fails(small_synthetic_df):
    """Target classes other than 0 and 1 must fail."""
    df_bad_target = small_synthetic_df.copy()
    df_bad_target.loc[0, TARGET_COLUMN] = 99
    with pytest.raises(DatasetValidationError, match="invalid classes"):
        validate_dataset_schema(df_bad_target, require_target=True)


def test_compute_data_quality_summary(small_synthetic_df):
    """Test data quality summary outputs expected fields and amount stats."""
    summary = compute_data_quality_summary(small_synthetic_df)
    assert summary["total_records"] == 200
    assert summary["total_features"] == 30
    assert "amount_stats" in summary
    assert "class_distribution" in summary
    assert summary["class_distribution"]["genuine_count"] > 0
    assert summary["class_distribution"]["fraud_count"] > 0


def test_check_dataset_status_structure():
    """Dataset status returns dictionary with required keys."""
    status = check_dataset_status()
    assert "exists" in status
    assert "path" in status
    assert "message" in status
