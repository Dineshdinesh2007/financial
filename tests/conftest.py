"""FinGuard AI Pytest Fixtures.

Provides deterministic transaction payloads, test database sessions,
and FastAPI TestClient instances.
"""

import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.api import app
from src.config import FEATURE_COLUMNS, ALL_COLUMNS, TIME_COLUMN, AMOUNT_COLUMN, TARGET_COLUMN
from src.data_loader import generate_synthetic_transactions, get_sample_transaction
from src.database import init_db, SessionLocal, Base, engine


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Ensure database tables exist for tests."""
    init_db()
    yield


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_single_transaction() -> dict:
    """Fixture returning a valid single transaction dictionary."""
    return get_sample_transaction(is_fraud_pattern=False)


@pytest.fixture
def fraudulent_single_transaction() -> dict:
    """Fixture returning a transaction with characteristic fraud shifts."""
    return get_sample_transaction(is_fraud_pattern=True)


@pytest.fixture
def small_synthetic_df() -> pd.DataFrame:
    """Fixture returning a small deterministic synthetic DataFrame."""
    return generate_synthetic_transactions(n_samples=200, fraud_ratio=0.05, random_state=123)
