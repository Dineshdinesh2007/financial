"""FinGuard AI Pydantic Validation Schemas.

Defines strict input and output models for single and batch predictions,
transaction history, review updates, health, and model metadata.
"""

from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime
import math
from pydantic import BaseModel, Field, field_validator, model_validator

from src.config import V_COLUMNS, FEATURE_COLUMNS, DEFAULT_DECISION_THRESHOLD, MAX_BATCH_ROWS


class PredictionStatusEnum(str, Enum):
    FLAG_FOR_REVIEW = "Flag for Review"
    LOWER_PREDICTED_RISK = "Lower Predicted Risk"


class ReviewStatusEnum(str, Enum):
    PENDING_REVIEW = "Pending Review"
    REVIEWED = "Reviewed"
    CONFIRMED_FRAUD = "Confirmed Fraud"
    LEGITIMATE = "Legitimate"


class SingleTransactionInput(BaseModel):
    """Schema for a single financial transaction.

    Features include Time (seconds), V1 through V28 (PCA transformed features),
    and Amount (transaction currency amount).
    """
    Time: float = Field(..., description="Seconds elapsed since the reference transaction.")
    Amount: float = Field(..., ge=0.0, description="Transaction amount (must be non-negative).")

    # V1 through V28
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    # Optional client-provided transaction reference identifier
    transaction_ref: Optional[str] = Field(None, max_length=100, description="Optional external transaction reference.")

    @model_validator(mode="before")
    @classmethod
    def check_finite_and_required(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values

        # Check all required features
        for col in FEATURE_COLUMNS:
            if col not in values:
                raise ValueError(f"Missing required feature: '{col}'")
            val = values[col]
            if val is None:
                raise ValueError(f"Feature '{col}' cannot be null.")
            try:
                f_val = float(val)
                if math.isnan(f_val) or math.isinf(f_val):
                    raise ValueError(f"Feature '{col}' must be a finite number.")
            except (TypeError, ValueError) as e:
                raise ValueError(f"Feature '{col}' must be a valid numeric float: {e}")

        # Check Amount >= 0
        if float(values["Amount"]) < 0.0:
            raise ValueError("Feature 'Amount' must be non-negative (>= 0.0).")

        return values


class PredictionResponse(BaseModel):
    """Response returned for a transaction risk scoring request."""
    transaction_id: int
    transaction_ref: Optional[str] = None
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Estimated fraud risk score.")
    threshold: float = Field(..., description="Decision threshold applied for classification.")
    prediction_status: PredictionStatusEnum
    recommended_action: str
    amount: float
    timestamp: str
    model_version: str
    disclaimer: str = (
        "Educational Screening Prototype: This score represents an estimated risk index and "
        "is not proof of criminal fraud. Flagged transactions should be evaluated by a human fraud analyst."
    )


class BatchTransactionItem(BaseModel):
    """Single item in a batch transaction payload."""
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    transaction_ref: Optional[str] = None


class BatchPredictionRequest(BaseModel):
    """Batch prediction request containing multiple transactions."""
    transactions: List[BatchTransactionItem] = Field(..., min_length=1)

    @field_validator("transactions")
    @classmethod
    def validate_batch_size(cls, v: List[BatchTransactionItem]) -> List[BatchTransactionItem]:
        if len(v) > MAX_BATCH_ROWS:
            raise ValueError(f"Batch size exceeds maximum limit of {MAX_BATCH_ROWS} transactions.")
        return v


class BatchPredictionResponse(BaseModel):
    """Response returned for a batch prediction request."""
    total_transactions: int
    flagged_for_review_count: int
    lower_risk_count: int
    flag_rate: float
    average_risk_score: float
    threshold_used: float
    model_version: str
    predictions: List[PredictionResponse]


class ReviewUpdateRequest(BaseModel):
    """Payload to update the human fraud review outcome."""
    review_status: ReviewStatusEnum
    reviewer_notes: Optional[str] = Field(None, max_length=1000)


class ReviewResponse(BaseModel):
    id: int
    prediction_id: int
    review_status: ReviewStatusEnum
    reviewer_notes: Optional[str] = None
    updated_at: str


class TransactionDetailResponse(BaseModel):
    id: int
    transaction_ref: Optional[str]
    timestamp: str
    model_version: str
    risk_score: float
    threshold: float
    prediction_status: str
    amount: float
    features: Optional[Dict[str, float]] = None
    review: Optional[ReviewResponse] = None


class PaginatedTransactionsResponse(BaseModel):
    items: List[TransactionDetailResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    decision_threshold: Optional[float] = None
    dataset_status: Dict[str, Any]
    database_connected: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    created_at: str
    dataset_source: str
    is_synthetic: bool
    features_count: int
    feature_columns: List[str]
    selected_threshold: float
    candidate_models_comparison: Dict[str, Any]
    feature_importances: List[Dict[str, Any]]
    training_config: Dict[str, Any]
    evaluation_metrics: Dict[str, Any]


class ThresholdUpdateRequest(BaseModel):
    threshold: float = Field(..., ge=0.01, le=0.99, description="New decision threshold (between 0.01 and 0.99).")
