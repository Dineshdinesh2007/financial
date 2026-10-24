"""FinGuard AI Risk Scoring and Prediction Engine.

Loads trained model artifacts and metadata, enforces feature schema,
executes single and batch risk scoring, and evaluates transactions against
the active decision threshold.
"""

from typing import Dict, Any, List, Optional, Tuple
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.config import (
    MODEL_FILE,
    METADATA_FILE,
    FEATURE_COLUMNS,
    DEFAULT_DECISION_THRESHOLD,
    MODEL_VERSION,
)
from src.schemas import PredictionStatusEnum


class ModelNotTrainedError(FileNotFoundError):
    """Raised when predictions are requested before the model pipeline is trained."""
    pass


class FeatureValidationError(ValueError):
    """Raised when an input transaction violates the required feature schema."""
    pass


class FraudPredictionEngine:
    """Thread-safe prediction service maintaining the trained model and decision threshold."""

    def __init__(self, model_path: Path = MODEL_FILE, metadata_path: Path = METADATA_FILE):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.pipeline = None
        self.metadata: Dict[str, Any] = {}
        self.decision_threshold: float = DEFAULT_DECISION_THRESHOLD
        self.model_version: str = MODEL_VERSION
        self.model_name: str = "Unknown"

        self.load_artifacts()

    def is_loaded(self) -> bool:
        """Return True if model pipeline and metadata are successfully loaded."""
        return self.pipeline is not None

    def load_artifacts(self) -> bool:
        """Load the model and metadata from disk.

        Returns:
            bool: True if loaded, False if files are missing.
        """
        if not self.model_path.exists() or not self.metadata_path.exists():
            return False

        try:
            self.pipeline = joblib.load(self.model_path)
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

            self.model_name = self.metadata.get("model_name", "TrainedClassifier")
            self.model_version = self.metadata.get("model_version", MODEL_VERSION)
            self.decision_threshold = float(self.metadata.get("selected_threshold", DEFAULT_DECISION_THRESHOLD))
            return True
        except Exception as e:
            print(f"[FinGuard AI Warning] Error loading model artifacts: {e}")
            self.pipeline = None
            return False

    def set_threshold(self, new_threshold: float) -> float:
        """Update the active decision threshold.

        Args:
            new_threshold: Float between 0.01 and 0.99.

        Returns:
            float: Updated decision threshold.
        """
        if not (0.01 <= new_threshold <= 0.99):
            raise ValueError("Threshold must be between 0.01 and 0.99")
        self.decision_threshold = round(float(new_threshold), 4)
        return self.decision_threshold

    def _ensure_model_ready(self) -> None:
        """Verify model readiness or raise ModelNotTrainedError with clear instructions."""
        if not self.is_loaded():
            loaded = self.load_artifacts()
            if not loaded:
                raise ModelNotTrainedError(
                    "Model artifact not found. Please train the model first by executing: "
                    "'python -m src.train' from the project root."
                )

    def validate_features(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Validate and format a raw dictionary into a single-row DataFrame.

        Args:
            data: Feature dictionary.

        Returns:
            pd.DataFrame: Formatted DataFrame matching FEATURE_COLUMNS order.
        """
        missing = [col for col in FEATURE_COLUMNS if col not in data]
        if missing:
            raise FeatureValidationError(f"Missing required feature(s): {missing[:5]}")

        row_dict = {}
        for col in FEATURE_COLUMNS:
            val = data[col]
            try:
                f_val = float(val)
                if np.isnan(f_val) or np.isinf(f_val):
                    raise FeatureValidationError(f"Feature '{col}' contains non-finite value ({val}).")
                row_dict[col] = f_val
            except (ValueError, TypeError):
                raise FeatureValidationError(f"Feature '{col}' must be a numeric value, got '{val}'.")

        if row_dict["Amount"] < 0:
            raise FeatureValidationError("Transaction Amount cannot be negative.")

        return pd.DataFrame([row_dict], columns=FEATURE_COLUMNS)

    def predict_single(
        self,
        transaction_data: Dict[str, Any],
        custom_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Score risk for a single transaction.

        Args:
            transaction_data: Dictionary containing Time, V1-V28, and Amount.
            custom_threshold: Optional threshold override.

        Returns:
            dict: Scored prediction results and recommendations.
        """
        self._ensure_model_ready()
        df_row = self.validate_features(transaction_data)

        # Predict probability for positive fraud class
        risk_score = float(self.pipeline.predict_proba(df_row)[0, 1])
        threshold_to_use = custom_threshold if custom_threshold is not None else self.decision_threshold

        is_flagged = risk_score >= threshold_to_use
        status = (
            PredictionStatusEnum.FLAG_FOR_REVIEW.value
            if is_flagged
            else PredictionStatusEnum.LOWER_PREDICTED_RISK.value
        )

        recommended_action = (
            "Manual review recommended. Route to fraud team for transaction verification and secondary checks."
            if is_flagged
            else "Continue normal transaction settlement. Transaction matches standard non-fraud profile."
        )

        return {
            "risk_score": round(risk_score, 4),
            "threshold": round(threshold_to_use, 4),
            "prediction_status": status,
            "recommended_action": recommended_action,
            "amount": float(transaction_data.get("Amount", 0.0)),
            "model_version": self.model_version,
            "features": {col: float(transaction_data[col]) for col in FEATURE_COLUMNS},
        }

    def predict_batch(
        self,
        df: pd.DataFrame,
        custom_threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """Perform vectorized batch risk scoring on a DataFrame.

        Args:
            df: DataFrame containing at least FEATURE_COLUMNS.
            custom_threshold: Optional threshold override.

        Returns:
            Tuple[np.ndarray, List[str], dict]: (risk_scores, status_list, summary_metrics)
        """
        self._ensure_model_ready()

        # Check required columns
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise FeatureValidationError(f"Batch dataset missing required columns: {missing[:5]}")

        # Ensure finite numeric values
        X = df[FEATURE_COLUMNS].copy()
        if X.isnull().any().any():
            raise FeatureValidationError("Batch contains null/missing values in feature columns.")

        X_num = X.to_numpy(dtype=float)
        if np.isinf(X_num).any():
            raise FeatureValidationError("Batch contains infinite numeric values.")

        if (df["Amount"] < 0).any():
            raise FeatureValidationError("Batch contains negative Amount values.")

        # Batch probability inference
        risk_scores = self.pipeline.predict_proba(X)[:, 1]
        threshold_to_use = custom_threshold if custom_threshold is not None else self.decision_threshold

        is_flagged = risk_scores >= threshold_to_use
        status_list = [
            PredictionStatusEnum.FLAG_FOR_REVIEW.value if f else PredictionStatusEnum.LOWER_PREDICTED_RISK.value
            for f in is_flagged
        ]

        flagged_count = int(np.sum(is_flagged))
        total = len(df)
        lower_risk_count = total - flagged_count
        flag_rate = round(float(flagged_count / total * 100.0), 2) if total > 0 else 0.0
        avg_risk = round(float(np.mean(risk_scores)), 4) if total > 0 else 0.0

        summary = {
            "total_transactions": total,
            "flagged_for_review_count": flagged_count,
            "lower_risk_count": lower_risk_count,
            "flag_rate": flag_rate,
            "average_risk_score": avg_risk,
            "threshold_used": round(threshold_to_use, 4),
            "model_version": self.model_version,
        }

        return risk_scores, status_list, summary


# Global singleton instance for API and internal calls
prediction_engine = FraudPredictionEngine()
