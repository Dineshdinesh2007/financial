"""Integration and API contract tests for FastAPI backend."""

import pytest
from fastapi.testclient import TestClient
import numpy as np


def test_health_endpoint(client: TestClient):
    """GET /health should return 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "model_loaded" in data
    assert "dataset_status" in data
    assert "database_connected" in data


def test_model_info_endpoint(client: TestClient):
    """GET /model-info should return model metadata and feature columns."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "model_version" in data
    assert "feature_columns" in data
    assert len(data["feature_columns"]) == 30
    assert "selected_threshold" in data


def test_metrics_endpoint(client: TestClient):
    """GET /metrics should return held-out evaluation metrics."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    metrics = data["metrics"]
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "confusion_matrix" in metrics


def test_predict_single_success(client: TestClient, valid_single_transaction: dict):
    """POST /predict with valid transaction should score and return 200."""
    response = client.post("/predict", json=valid_single_transaction)
    assert response.status_code == 200
    data = response.json()
    assert "transaction_id" in data
    assert "risk_score" in data
    assert 0.0 <= data["risk_score"] <= 1.0
    assert data["prediction_status"] in ["Flag for Review", "Lower Predicted Risk"]
    assert "disclaimer" in data


def test_predict_single_missing_feature(client: TestClient, valid_single_transaction: dict):
    """POST /predict missing a feature must return 422."""
    bad_payload = valid_single_transaction.copy()
    del bad_payload["V10"]
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_single_negative_amount(client: TestClient, valid_single_transaction: dict):
    """POST /predict with negative Amount must return 422."""
    bad_payload = valid_single_transaction.copy()
    bad_payload["Amount"] = -50.0
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_batch_success(client: TestClient, valid_single_transaction: dict, fraudulent_single_transaction: dict):
    """POST /predict/batch should score multiple transactions."""
    payload = {
        "transactions": [
            valid_single_transaction,
            fraudulent_single_transaction,
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_transactions"] == 2
    assert "flagged_for_review_count" in data
    assert len(data["predictions"]) == 2


def test_predict_batch_empty_rejected(client: TestClient):
    """POST /predict/batch with empty list must return 422."""
    response = client.post("/predict/batch", json={"transactions": []})
    assert response.status_code == 422


def test_transactions_history_and_pagination(client: TestClient, valid_single_transaction: dict):
    """Ensure scored transactions appear in GET /transactions."""
    # First submit a prediction
    client.post("/predict", json=valid_single_transaction)

    response = client.get("/transactions?page=1&page_size=10")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert len(data["items"]) >= 1


def test_transaction_detail_and_not_found(client: TestClient, valid_single_transaction: dict):
    """GET /transactions/{id} succeeds for existing, returns 404 for invalid."""
    pred_res = client.post("/predict", json=valid_single_transaction).json()
    tx_id = pred_res["transaction_id"]

    res = client.get(f"/transactions/{tx_id}")
    assert res.status_code == 200
    assert res.json()["id"] == tx_id

    bad_res = client.get("/transactions/99999999")
    assert bad_res.status_code == 404


def test_review_status_workflow(client: TestClient, valid_single_transaction: dict):
    """PATCH /transactions/{id}/review should update review outcome."""
    pred_res = client.post("/predict", json=valid_single_transaction).json()
    tx_id = pred_res["transaction_id"]

    # Update to Confirmed Fraud with analyst notes
    update_res = client.patch(
        f"/transactions/{tx_id}/review",
        json={"review_status": "Confirmed Fraud", "reviewer_notes": "Identified compromised card pattern."}
    )
    assert update_res.status_code == 200
    assert update_res.json()["review_status"] == "Confirmed Fraud"
    assert update_res.json()["reviewer_notes"] == "Identified compromised card pattern."

    # Verify updated record via detail endpoint
    detail = client.get(f"/transactions/{tx_id}").json()
    assert detail["review"]["review_status"] == "Confirmed Fraud"


def test_threshold_settings_update(client: TestClient):
    """PATCH /settings/threshold updates the active threshold."""
    res = client.patch("/settings/threshold", json={"threshold": 0.42})
    assert res.status_code == 200
    assert res.json()["new_threshold"] == 0.42

    # Check health reflects updated threshold
    health = client.get("/health").json()
    assert health["decision_threshold"] == 0.42


def test_clear_history_requires_confirmation(client: TestClient):
    """DELETE /history without confirm=true must fail with 400."""
    res = client.delete("/history")
    assert res.status_code == 400

    # With confirmation
    res_confirmed = client.delete("/history?confirm=true")
    assert res_confirmed.status_code == 200
    assert "deleted_count" in res_confirmed.json()
