"""FinGuard AI FastAPI Backend Application.

Provides RESTful endpoints for transaction fraud risk inference, batch scoring,
prediction history query and pagination, human review workflow, model metadata,
and system health checks.
"""

from typing import Optional, Dict, Any, List
import logging
from datetime import datetime, timezone
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import desc

from src.config import APP_VERSION, DEFAULT_DECISION_THRESHOLD, MAX_BATCH_ROWS
from src.data_loader import check_dataset_status
from src.database import (
    init_db,
    get_db,
    store_prediction,
    update_review_status,
    clear_prediction_history,
    PredictionRecord,
    ReviewRecord,
)
from src.predict import (
    prediction_engine,
    ModelNotTrainedError,
    FeatureValidationError,
)
from src.schemas import (
    SingleTransactionInput,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ReviewUpdateRequest,
    ReviewResponse,
    TransactionDetailResponse,
    PaginatedTransactionsResponse,
    HealthResponse,
    ModelInfoResponse,
    ThresholdUpdateRequest,
    PredictionStatusEnum,
)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("finguard-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler initializing DB tables and verifying model artifacts."""
    logger.info("Initializing FinGuard AI database tables...")
    init_db()

    if prediction_engine.is_loaded():
        logger.info(
            f"Model '{prediction_engine.model_name}' ({prediction_engine.model_version}) "
            f"ready with active threshold {prediction_engine.decision_threshold:.3f}"
        )
    else:
        logger.warning("Model artifacts not yet found on disk. Run 'python -m src.train' to build them.")
    yield
    logger.info("FinGuard AI API shutting down.")


app = FastAPI(
    title="FinGuard AI API",
    description=(
        "Intelligent Financial Transaction Fraud Detection System API.\n"
        "Screening prototype providing fraud risk estimation, review workflow, "
        "and transaction audit history."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)

# Enable CORS for frontend dashboard communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System Health"])
def get_health(db: Session = Depends(get_db)):
    """Return health status, database connectivity, dataset presence, and model readiness."""
    dataset_status = check_dataset_status()
    db_connected = True
    try:
        db.query(PredictionRecord).first()
    except Exception as e:
        logger.error(f"Database connectivity check failed: {e}")
        db_connected = False

    return HealthResponse(
        status="healthy" if (prediction_engine.is_loaded() and db_connected) else "degraded",
        model_loaded=prediction_engine.is_loaded(),
        model_name=prediction_engine.model_name if prediction_engine.is_loaded() else None,
        model_version=prediction_engine.model_version if prediction_engine.is_loaded() else None,
        decision_threshold=prediction_engine.decision_threshold if prediction_engine.is_loaded() else None,
        dataset_status=dataset_status,
        database_connected=db_connected,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model Governance"])
def get_model_info():
    """Return metadata for the currently active machine learning model."""
    if not prediction_engine.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts not found. Please train the model first by running 'python -m src.train'.",
        )

    meta = prediction_engine.metadata
    return ModelInfoResponse(
        model_name=meta.get("model_name", prediction_engine.model_name),
        model_version=meta.get("model_version", prediction_engine.model_version),
        created_at=meta.get("created_at", ""),
        dataset_source=meta.get("dataset_source", "unknown"),
        is_synthetic=meta.get("is_synthetic", False),
        features_count=meta.get("features_count", 30),
        feature_columns=meta.get("feature_columns", []),
        selected_threshold=prediction_engine.decision_threshold,
        candidate_models_comparison=meta.get("candidate_models_comparison", {}),
        feature_importances=meta.get("feature_importances", []),
        training_config=meta.get("training_config", {}),
        evaluation_metrics=meta.get("evaluation_metrics", {}),
    )


@app.get("/metrics", tags=["Model Governance"])
def get_evaluation_metrics():
    """Return comprehensive held-out test evaluation metrics."""
    if not prediction_engine.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not trained. Execute 'python -m src.train' to compute evaluation metrics.",
        )

    eval_metrics = prediction_engine.metadata.get("evaluation_metrics", {})
    return {
        "model_name": prediction_engine.model_name,
        "model_version": prediction_engine.model_version,
        "is_synthetic": prediction_engine.metadata.get("is_synthetic", False),
        "dataset_source": prediction_engine.metadata.get("dataset_source", "unknown"),
        "metrics": eval_metrics,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Risk Scoring"])
def predict_single_transaction(
    payload: SingleTransactionInput,
    db: Session = Depends(get_db),
):
    """Score fraud risk for a single transaction and persist the record in SQLite."""
    try:
        data = payload.model_dump()
        tx_ref = data.pop("transaction_ref", None)

        res = prediction_engine.predict_single(data)

        # Store in SQLite
        db_rec = store_prediction(
            db=db,
            risk_score=res["risk_score"],
            threshold=res["threshold"],
            prediction_status=res["prediction_status"],
            amount=res["amount"],
            model_version=res["model_version"],
            features=res["features"],
            transaction_ref=tx_ref,
        )

        logger.info(
            f"Scored transaction #{db_rec.id} (ref={tx_ref}): "
            f"Amount=${res['amount']:.2f}, Risk={res['risk_score']:.4f}, Status={res['prediction_status']}"
        )

        return PredictionResponse(
            transaction_id=db_rec.id,
            transaction_ref=db_rec.transaction_ref,
            risk_score=res["risk_score"],
            threshold=res["threshold"],
            prediction_status=PredictionStatusEnum(res["prediction_status"]),
            recommended_action=res["recommended_action"],
            amount=res["amount"],
            timestamp=db_rec.timestamp.isoformat(),
            model_version=res["model_version"],
        )
    except ModelNotTrainedError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except FeatureValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal inference failure.")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Risk Scoring"])
def predict_batch_transactions(
    payload: BatchPredictionRequest,
    db: Session = Depends(get_db),
):
    """Score a batch of transactions and persist them in SQLite."""
    try:
        import pandas as pd
        raw_items = [item.model_dump() for item in payload.transactions]
        df_batch = pd.DataFrame(raw_items)

        tx_refs = df_batch["transaction_ref"].tolist() if "transaction_ref" in df_batch.columns else [None] * len(df_batch)

        risk_scores, status_list, summary = prediction_engine.predict_batch(df_batch)

        predictions_out: List[PredictionResponse] = []
        # Batch insert to DB
        for i in range(len(df_batch)):
            amt = float(df_batch.iloc[i]["Amount"])
            score = float(risk_scores[i])
            st = status_list[i]
            ref = tx_refs[i] if tx_refs[i] is not None else None

            rec = store_prediction(
                db=db,
                risk_score=score,
                threshold=summary["threshold_used"],
                prediction_status=st,
                amount=amt,
                model_version=summary["model_version"],
                features=None,  # Omit raw feature JSON for huge batches to conserve DB space
                transaction_ref=ref,
            )

            rec_action = (
                "Manual review recommended. Route to fraud team."
                if st == PredictionStatusEnum.FLAG_FOR_REVIEW.value
                else "Continue normal transaction settlement."
            )

            predictions_out.append(
                PredictionResponse(
                    transaction_id=rec.id,
                    transaction_ref=rec.transaction_ref,
                    risk_score=score,
                    threshold=summary["threshold_used"],
                    prediction_status=PredictionStatusEnum(st),
                    recommended_action=rec_action,
                    amount=amt,
                    timestamp=rec.timestamp.isoformat(),
                    model_version=summary["model_version"],
                )
            )

        logger.info(
            f"Batch processed {summary['total_transactions']} transactions: "
            f"{summary['flagged_for_review_count']} flagged ({summary['flag_rate']}%)"
        )

        return BatchPredictionResponse(
            total_transactions=summary["total_transactions"],
            flagged_for_review_count=summary["flagged_for_review_count"],
            lower_risk_count=summary["lower_risk_count"],
            flag_rate=summary["flag_rate"],
            average_risk_score=summary["average_risk_score"],
            threshold_used=summary["threshold_used"],
            model_version=summary["model_version"],
            predictions=predictions_out,
        )
    except ModelNotTrainedError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except FeatureValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Batch prediction failure: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Batch inference error.")


@app.get("/transactions", response_model=PaginatedTransactionsResponse, tags=["Transaction History"])
def get_transactions_history(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    status_filter: Optional[str] = Query(None, description="Filter by prediction_status"),
    review_filter: Optional[str] = Query(None, description="Filter by review_status"),
    search: Optional[str] = Query(None, description="Search by transaction reference or ID"),
    db: Session = Depends(get_db),
):
    """Retrieve paginated prediction records with human review status."""
    query = db.query(PredictionRecord).outerjoin(ReviewRecord)

    if status_filter:
        query = query.filter(PredictionRecord.prediction_status == status_filter)

    if review_filter:
        query = query.filter(ReviewRecord.review_status == review_filter)

    if search:
        search_term = f"%{search}%"
        if search.isdigit():
            query = query.filter((PredictionRecord.id == int(search)) | (PredictionRecord.transaction_ref.ilike(search_term)))
        else:
            query = query.filter(PredictionRecord.transaction_ref.ilike(search_term))

    total = query.count()
    total_pages = max(1, (total + page_size - 1) // page_size)

    records = (
        query.order_by(desc(PredictionRecord.id))
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    items = []
    for r in records:
        rev_item = None
        if r.reviews:
            latest_rev = r.reviews[-1]
            rev_item = ReviewResponse(
                id=latest_rev.id,
                prediction_id=r.id,
                review_status=latest_rev.review_status,
                reviewer_notes=latest_rev.reviewer_notes,
                updated_at=latest_rev.updated_at.isoformat(),
            )

        features_dict = json.loads(r.features_json) if r.features_json else None

        items.append(
            TransactionDetailResponse(
                id=r.id,
                transaction_ref=r.transaction_ref,
                timestamp=r.timestamp.isoformat(),
                model_version=r.model_version,
                risk_score=r.risk_score,
                threshold=r.threshold,
                prediction_status=r.prediction_status,
                amount=r.amount,
                features=features_dict,
                review=rev_item,
            )
        )

    return PaginatedTransactionsResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@app.get("/transactions/{transaction_id}", response_model=TransactionDetailResponse, tags=["Transaction History"])
def get_transaction_detail(transaction_id: int, db: Session = Depends(get_db)):
    """Fetch full details and features of a single stored transaction."""
    r = db.query(PredictionRecord).filter(PredictionRecord.id == transaction_id).first()
    if not r:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction with ID #{transaction_id} was not found."
        )

    rev_item = None
    if r.reviews:
        latest_rev = r.reviews[-1]
        rev_item = ReviewResponse(
            id=latest_rev.id,
            prediction_id=r.id,
            review_status=latest_rev.review_status,
            reviewer_notes=latest_rev.reviewer_notes,
            updated_at=latest_rev.updated_at.isoformat(),
        )

    features_dict = json.loads(r.features_json) if r.features_json else None

    return TransactionDetailResponse(
        id=r.id,
        transaction_ref=r.transaction_ref,
        timestamp=r.timestamp.isoformat(),
        model_version=r.model_version,
        risk_score=r.risk_score,
        threshold=r.threshold,
        prediction_status=r.prediction_status,
        amount=r.amount,
        features=features_dict,
        review=rev_item,
    )


@app.patch("/transactions/{transaction_id}/review", response_model=ReviewResponse, tags=["Review Workflow"])
def update_transaction_review(
    transaction_id: int,
    payload: ReviewUpdateRequest,
    db: Session = Depends(get_db),
):
    """Update human fraud analyst review status ('Pending Review', 'Reviewed', 'Confirmed Fraud', 'Legitimate')."""
    rev = update_review_status(
        db=db,
        prediction_id=transaction_id,
        status=payload.review_status.value,
        notes=payload.reviewer_notes,
    )

    if not rev:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction with ID #{transaction_id} not found."
        )

    logger.info(f"Review updated for transaction #{transaction_id}: status='{payload.review_status.value}'")
    return ReviewResponse(
        id=rev.id,
        prediction_id=rev.prediction_id,
        review_status=rev.review_status,
        reviewer_notes=rev.reviewer_notes,
        updated_at=rev.updated_at.isoformat(),
    )


@app.patch("/settings/threshold", tags=["Settings"])
def update_decision_threshold(payload: ThresholdUpdateRequest):
    """Update active decision threshold for incoming predictions."""
    try:
        new_thresh = prediction_engine.set_threshold(payload.threshold)
        logger.info(f"Decision threshold updated to {new_thresh}")
        return {
            "status": "success",
            "message": f"Decision threshold updated to {new_thresh:.4f}.",
            "new_threshold": new_thresh,
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.delete("/history", tags=["Settings"])
def clear_history(confirm: bool = Query(False, description="Must be True to confirm reset"), db: Session = Depends(get_db)):
    """Reset demonstration history (requires explicit confirmation)."""
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reset confirmation parameter 'confirm=true' required."
        )
    deleted_count = clear_prediction_history(db)
    logger.info(f"Demo transaction history cleared ({deleted_count} records removed).")
    return {
        "status": "success",
        "message": f"Successfully cleared {deleted_count} stored prediction records.",
        "deleted_count": deleted_count,
    }
