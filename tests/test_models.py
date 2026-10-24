"""Unit tests for ML modeling pipeline and inference engine."""

import pytest
import numpy as np
import pandas as pd

from src.train import split_data, find_optimal_threshold
from src.predict import (
    FraudPredictionEngine,
    prediction_engine,
    FeatureValidationError,
)
from src.config import FEATURE_COLUMNS, MODEL_FILE, METADATA_FILE


def test_split_data_proportions(small_synthetic_df):
    """Test 60/20/20 stratified data splitting."""
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        small_synthetic_df, test_size=0.20, val_size=0.20, random_state=42
    )

    total = len(small_synthetic_df)
    assert len(X_test) == int(total * 0.20)
    assert len(X_val) == int(total * 0.20)
    assert len(X_train) == total - len(X_test) - len(X_val)

    # Both train and test should contain both classes
    assert len(np.unique(y_train)) == 2
    assert len(np.unique(y_test)) == 2


def test_find_optimal_threshold():
    """Test threshold optimization selects threshold with highest F1."""
    y_true = pd.Series([0, 0, 0, 1, 1, 0, 1, 0, 0, 1])
    # Scores that correlate with labels
    y_scores = np.array([0.1, 0.2, 0.15, 0.85, 0.78, 0.35, 0.90, 0.05, 0.12, 0.82])

    thresh, metrics = find_optimal_threshold(y_true, y_scores)
    assert 0.05 <= thresh <= 0.95
    assert "val_f1" in metrics
    assert metrics["val_f1"] > 0.80


def test_prediction_engine_is_loaded():
    """Verify that global prediction_engine loaded the trained model artifact."""
    assert prediction_engine.is_loaded() is True
    assert prediction_engine.pipeline is not None
    assert prediction_engine.model_name in ["LogisticRegression", "RandomForest"]
    assert 0.0 < prediction_engine.decision_threshold < 1.0


def test_prediction_engine_single_prediction(valid_single_transaction):
    """Test scoring a valid single transaction."""
    res = prediction_engine.predict_single(valid_single_transaction)
    assert "risk_score" in res
    assert 0.0 <= res["risk_score"] <= 1.0
    assert "threshold" in res
    assert res["prediction_status"] in ["Flag for Review", "Lower Predicted Risk"]
    assert "recommended_action" in res


def test_prediction_engine_missing_feature_raises(valid_single_transaction):
    """Omitting any of the 30 features must raise FeatureValidationError."""
    corrupted = valid_single_transaction.copy()
    del corrupted["V1"]
    with pytest.raises(FeatureValidationError, match="Missing required feature"):
        prediction_engine.predict_single(corrupted)


def test_prediction_engine_non_numeric_raises(valid_single_transaction):
    """Non-numeric values in features must raise FeatureValidationError."""
    corrupted = valid_single_transaction.copy()
    corrupted["V28"] = "abc_bad_val"
    with pytest.raises(FeatureValidationError, match="numeric"):
        prediction_engine.predict_single(corrupted)


def test_prediction_engine_batch_prediction(small_synthetic_df):
    """Test batch prediction produces arrays of risk scores and status."""
    scores, statuses, summary = prediction_engine.predict_batch(small_synthetic_df)
    assert len(scores) == len(small_synthetic_df)
    assert len(statuses) == len(small_synthetic_df)
    assert summary["total_transactions"] == len(small_synthetic_df)
    assert summary["flagged_for_review_count"] + summary["lower_risk_count"] == len(small_synthetic_df)
