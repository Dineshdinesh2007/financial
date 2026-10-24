"""FinGuard AI Database Persistence Module.

SQLAlchemy ORM models for predictions, human fraud reviews, and model evaluations.
Provides automatic table initialization, session management, and history clearing.
"""

from typing import Generator, Optional, Dict, Any, List
from datetime import datetime, timezone
import json
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Text,
    ForeignKey,
    desc,
    func,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session

from src.config import DATABASE_URL, APP_VERSION

Base = declarative_base()


class PredictionRecord(Base):
    """Database record storing risk prediction outputs and transaction metadata."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    transaction_ref = Column(String(100), nullable=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    model_version = Column(String(50), nullable=False)
    risk_score = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    prediction_status = Column(String(50), nullable=False, index=True)  # 'Flag for Review' or 'Lower Predicted Risk'
    amount = Column(Float, nullable=False)
    features_json = Column(Text, nullable=True)  # JSON-encoded dictionary of input features

    # 1-to-1 or 1-to-many relationship with ReviewRecord
    reviews = relationship("ReviewRecord", back_populates="prediction", cascade="all, delete-orphan")


class ReviewRecord(Base):
    """Database record storing human fraud analyst decisions."""
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False, index=True)
    review_status = Column(String(50), nullable=False, default="Pending Review", index=True)
    reviewer_notes = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    prediction = relationship("PredictionRecord", back_populates="reviews")


class EvaluationAuditRecord(Base):
    """Audit table storing model evaluation snapshots."""
    __tablename__ = "model_evaluations"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    model_version = Column(String(50), nullable=False)
    dataset_description = Column(String(200), nullable=False)
    metrics_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)


# Initialize Engine and SessionMaker
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Needed for SQLite in multi-threaded FastAPI/Streamlit
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create all tables on first startup if they do not exist."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency helper yielding an active SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def store_prediction(
    db: Session,
    risk_score: float,
    threshold: float,
    prediction_status: str,
    amount: float,
    model_version: str,
    features: Optional[Dict[str, float]] = None,
    transaction_ref: Optional[str] = None,
) -> PredictionRecord:
    """Persist a single prediction and create a corresponding initial review record.

    Args:
        db: Active DB session.
        risk_score: Float risk score (0.0 to 1.0).
        threshold: Decision threshold used.
        prediction_status: 'Flag for Review' or 'Lower Predicted Risk'.
        amount: Transaction amount.
        model_version: Model identifier string.
        features: Optional raw feature dictionary.
        transaction_ref: Optional client transaction ID.

    Returns:
        PredictionRecord: The newly persisted DB record.
    """
    rec = PredictionRecord(
        transaction_ref=transaction_ref,
        timestamp=datetime.now(timezone.utc),
        model_version=model_version,
        risk_score=float(risk_score),
        threshold=float(threshold),
        prediction_status=prediction_status,
        amount=float(amount),
        features_json=json.dumps(features) if features else None,
    )
    db.add(rec)
    db.flush()  # Populates rec.id

    # Create initial review item: Pending Review if flagged, otherwise Legitimate or Pending Review
    initial_status = "Pending Review" if prediction_status == "Flag for Review" else "Legitimate"
    rev = ReviewRecord(
        prediction_id=rec.id,
        review_status=initial_status,
        reviewer_notes="Initial automated triage" if prediction_status != "Flag for Review" else "Auto-flagged for analyst review",
        updated_at=datetime.now(timezone.utc),
    )
    db.add(rev)
    db.commit()
    db.refresh(rec)
    return rec


def update_review_status(
    db: Session,
    prediction_id: int,
    status: str,
    notes: Optional[str] = None
) -> Optional[ReviewRecord]:
    """Update or create the review record for a given prediction."""
    rev = db.query(ReviewRecord).filter(ReviewRecord.prediction_id == prediction_id).first()
    if not rev:
        # Check if prediction exists
        pred = db.query(PredictionRecord).filter(PredictionRecord.id == prediction_id).first()
        if not pred:
            return None
        rev = ReviewRecord(prediction_id=prediction_id)
        db.add(rev)

    rev.review_status = status
    if notes is not None:
        rev.reviewer_notes = notes
    rev.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(rev)
    return rev


def clear_prediction_history(db: Session) -> int:
    """Clear all prediction and review history for demo resets.

    Returns:
        int: Number of deleted prediction records.
    """
    count = db.query(PredictionRecord).count()
    db.query(ReviewRecord).delete()
    db.query(PredictionRecord).delete()
    db.commit()
    return count
