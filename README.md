# 🛡️ FinGuard AI — Intelligent Financial Transaction Fraud Detection System

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Tests-29%20Passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Educational Fraud-Risk Screening Notice:**  
> FinGuard AI is an educational fraud-risk screening prototype. Predictions represent an estimated statistical risk score rather than conclusive proof of criminal misconduct. The system is designed to triage high-risk transactions for human analyst review and does not automatically freeze funds or decline real-world payments.

---

## 1. Problem Statement

Financial fraud costs the global economy tens of billions of dollars each year. Digital transaction volumes continue to surge, yet fraudulent transactions constitute a minute fraction (typically **0.1% to 0.2%**) of total volume. This extreme class imbalance creates critical challenges:

1. **The Accuracy Paradox:** A naive model predicting 100% genuine transactions achieves 99.8% accuracy while failing to detect a single fraudulent transaction.
2. **Alert Fatigue:** A system with low precision swamps fraud analysts with false alarms, wasting operational hours and delaying genuine emergency interventions.
3. **Data Leakage in Pipelines:** Naive preprocessing or oversampling applied prior to dataset splitting results in artificially inflated test scores that fail in production.

**FinGuard AI** addresses these challenges with a reproducible, leakage-free machine learning pipeline, validation average precision (PR-AUC) optimization, configurable decision thresholding, a production-conscious FastAPI backend with SQLite persistence, and an interactive Streamlit triage dashboard.

---

## 2. Key Features

- **Leakage-Free ML Pipeline:**
  - 60% Train / 20% Validation / 20% untouched Test stratified split.
  - Preprocessing (`RobustScaler`) fitted strictly on training data.
  - Class weighting (`balanced`) to handle extreme class imbalance without synthetic sampling on validation or test sets.
  - Model comparison between **Logistic Regression** and **Random Forest**.
  - Systematic threshold tuning on validation data maximizing minority fraud F1-score.
  - Final evaluation metrics calculated exclusively on the untouched hold-out test set.
- **Production-Ready FastAPI REST API:**
  - Pydantic v2 strict schema validation with numeric finiteness checks and negative amount rejection.
  - Single transaction inference (`POST /predict`) and high-throughput batch scoring (`POST /predict/batch`).
  - Structured logging omitting sensitive raw payloads.
  - OpenAPI Swagger documentation at `/docs`.
- **Relational Persistence (SQLite + SQLAlchemy):**
  - Stored audit trail of predictions, risk scores, and decision thresholds.
  - Human review workflow supporting statuses: `Pending Review`, `Reviewed`, `Confirmed Fraud`, and `Legitimate`.
  - Pagination, search, and filtering.
- **Fintech UI/UX Dashboard (Streamlit):**
  - **Overview:** Real-time KPI summary cards, risk distributions, and recent triage queues.
  - **Single Transaction Analysis:** Interactive parameter form, collapsible PCA features, sample loaders (Benign vs Suspicious), and risk gauges.
  - **Batch Detection:** CSV drag-and-drop screening, progress bars, threshold filtering, and downloadable scored CSV reports.
  - **Fraud Analytics:** Interactive Plotly distribution charts, risk severity bands, and amount breakdowns.
  - **Transaction History:** Paginated table, search query, and human review decision update workflow.
  - **Model Performance:** Real hold-out metrics (PR-AUC, ROC-AUC, F1, Precision, Recall), confusion matrix heatmap, and PR/ROC curves.
  - **Settings:** Dynamic threshold adjustment slider (0.05 - 0.95), dataset status monitor, and safe demonstration history reset.
  - **About Project:** Architecture breakdown, ethics, and governance guidelines.
- **Safe Labeled Demo Mode:**
  - Automatically verifies whether the Kaggle dataset exists. If missing, runs a clearly labeled synthetic generator to enable end-to-end testing without blocking execution.

---

## 3. Architecture Diagram

```
+-------------------------------------------------------------+
|                     Data Ingestion                          |
|  - data/raw/creditcard.csv (Kaggle ULB)                     |
|  - data/synthetic/demo_transactions.csv (Fallback Demo)     |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                  Machine Learning Pipeline                  |
|  - Stratified Split (60% Train, 20% Val, 20% Test)          |
|  - Pipeline: RobustScaler -> Balanced Class Weighting       |
|  - Candidates: Logistic Regression vs Random Forest         |
|  - Threshold Optimization on Validation Set                 |
|  - Evaluation on Untouched Test Set (Precision, Recall, AP) |
|  - Serialization: models/best_model.joblib + metadata.json  |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                  FastAPI Backend Server                     |
|  - Strict Pydantic Validation (Finite check, Amount >= 0)   |
|  - Prediction Service (Singleton Model Runner)              |
|  - SQLite Database (SQLAlchemy ORM)                         |
|    * predictions table                                      |
|    * reviews table                                          |
|  - Endpoints: /health, /model-info, /predict, /transactions |
+------------------------------+------------------------------+
                               ^
                               | HTTP / JSON Requests
                               v
+-------------------------------------------------------------+
|               Streamlit Fintech Dashboard                   |
|  - Overview Dashboard & KPI Summary Cards                   |
|  - Single Transaction Analysis & Risk Gauge                 |
|  - Batch Detection & Downloadable Scored CSV                |
|  - Fraud Analytics & Severity Distribution                  |
|  - Transaction History & Analyst Review Workflow            |
|  - Model Performance & Confusion Matrix Heatmap             |
|  - Settings & Dynamic Threshold Configuration               |
|  - About Project Demonstration Guide                        |
+-------------------------------------------------------------+
```

---

## 4. Technology Stack

| Layer | Technologies | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.11, 3.12, 3.14 | Core language runtime |
| **Machine Learning** | Scikit-Learn, Pandas, NumPy, Joblib | Pipeline modeling, scaling, threshold tuning, and serialization |
| **Backend API** | FastAPI, Uvicorn, Pydantic v2 | High-performance RESTful API and input validation |
| **Database** | SQLite, SQLAlchemy ORM | Audit trail and human review workflow persistence |
| **Frontend UI** | Streamlit, Plotly, HTML/CSS | Modern fintech dashboard and data visualizations |
| **Testing** | Pytest, FastAPI TestClient, HTTPX | Contract, schema, and unit testing |
| **Deployment** | Docker, Docker Compose | Multi-container reproducible runtime |

---

## 5. Dataset Information & Setup

### Dataset Source
The primary evaluation dataset is the **Credit Card Fraud Detection** dataset published by the Machine Learning Group (ULB) on Kaggle:
- **URL:** [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **File Name:** `creditcard.csv`
- **Dimensions:** 284,807 transactions, 31 columns.
- **Features:**
  - `Time`: Elapsed seconds from the initial recorded transaction.
  - `V1` to `V28`: PCA-transformed numeric components to protect cardholder privacy.
  - `Amount`: Transaction currency amount.
  - `Class`: Ground-truth label (`0` = Genuine, `1` = Fraudulent).

### Download Instructions
1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Place the file inside your project directory at:
   ```bash
   data/raw/creditcard.csv
   ```
3. Re-train the model on real data:
   ```bash
   python -m src.train
   ```

### Demo Mode (When Real Dataset is Not Downloaded)
If `data/raw/creditcard.csv` is not present, FinGuard AI automatically detects its absence and generates a clearly labeled synthetic dataset in `data/synthetic/demo_transactions.csv`.  
- **Labeling:** The UI and API clearly label synthetic mode.
- **Strict Separation:** Synthetic demo results are never mixed with real Kaggle dataset metrics.
- **Git Protection:** Raw datasets and SQLite databases are excluded via `.gitignore`.

---

## 6. Local Installation and Execution

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/finguard-ai.git
cd finguard-ai
```

### Step 2: Create and Activate Virtual Environment
```bash
# macOS / Linux:
python3 -m venv .venv
source .venv/bin/activate

# Windows (Command Prompt / PowerShell):
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 4: Train the Model Pipeline
```bash
python -m src.train
```
*Executes stratified splitting, trains Logistic Regression and Random Forest, tunes the decision threshold on validation data, evaluates on the untouched test set, and writes `models/best_model.joblib` and `models/metadata.json`.*

### Step 5: Start the FastAPI Backend
```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```
- API Base URL: `http://127.0.0.1:8000`
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`

### Step 6: Start the Streamlit Dashboard (In a separate terminal)
```bash
source .venv/bin/activate
streamlit run src/app.py
```
- Dashboard URL: `http://localhost:8501`

---

## 7. Automated Testing

FinGuard AI includes a 29-test comprehensive test suite verifying:
- Model artifact loading and schema matching.
- Missing feature rejection and negative amount checks.
- Finite numeric validation.
- Valid single-transaction and batch scoring.
- Database persistence and human review status updates.
- API endpoints (`/health`, `/model-info`, `/metrics`, `/predict`, `/transactions`).
- Demonstration history clearing with confirmation safeguards.

Execute tests with:
```bash
source .venv/bin/activate
python -m pytest -q
```

**Test Execution Output:**
```
.............................                                            [100%]
29 passed in 0.56s
```

---

## 8. API Endpoint Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Check service health, model readiness, dataset status, and DB connection |
| `GET` | `/model-info` | Return model architecture, version, features, and training configuration |
| `GET` | `/metrics` | Return evaluation metrics from the held-out untouched test set |
| `POST` | `/predict` | Score fraud risk for a single transaction (persists record in SQLite) |
| `POST` | `/predict/batch` | Batch score multiple transactions with triage metrics |
| `GET` | `/transactions` | Paginated transaction history with status and review filters |
| `GET` | `/transactions/{id}` | Detailed feature values and audit log for a single transaction |
| `PATCH`| `/transactions/{id}/review` | Update human fraud analyst decision (`Pending Review`, `Confirmed Fraud`, etc.) |
| `PATCH`| `/settings/threshold` | Update active classification decision threshold dynamically |
| `DELETE`| `/history?confirm=true` | Reset demonstration prediction records |

### Example API Request: Single Prediction
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 85200.0,
    "Amount": 149.50,
    "transaction_ref": "TX-SAMPLE-001",
    "V1": -1.25, "V2": 0.45, "V3": -0.80, "V4": 3.40, "V5": 0.10,
    "V6": -0.50, "V7": 0.20, "V8": 0.15, "V9": -1.10, "V10": -3.80,
    "V11": 2.90, "V12": -4.20, "V13": 0.05, "V14": -5.60, "V15": -0.20,
    "V16": -1.80, "V17": -3.20, "V18": -0.40, "V19": 0.30, "V20": 0.40,
    "V21": 0.25, "V22": -0.10, "V23": 0.05, "V24": 0.02, "V25": -0.15,
    "V26": 0.10, "V27": 0.20, "V28": 0.05
  }'
```

**Example JSON Response:**
```json
{
  "transaction_id": 1,
  "transaction_ref": "TX-SAMPLE-001",
  "risk_score": 0.8942,
  "threshold": 0.1800,
  "prediction_status": "Flag for Review",
  "recommended_action": "Manual review recommended. Route to fraud team for transaction verification and secondary checks.",
  "amount": 149.5,
  "timestamp": "2026-10-24T07:17:07.770000Z",
  "model_version": "v1.0.0",
  "disclaimer": "Educational Screening Prototype: This score represents an estimated risk index and is not proof of criminal fraud. Flagged transactions should be evaluated by a human fraud analyst."
}
```

---

## 9. Docker Deployment

Deploy FinGuard AI using Docker Compose:

```bash
docker-compose up --build
```

- FastAPI backend starts on `http://localhost:8000`
- Streamlit dashboard starts on `http://localhost:8501`

---

## 10. Ethical Considerations & Limitations

- **Human-in-the-Loop:** Automated fraud screening models must not make punitive, autonomous decisions (such as freezing bank accounts or cancelling cards) without human oversight.
- **Concept Drift:** Fraud patterns evolve rapidly. A static model trained on past transactions must be continuously retrained with updated ground truth.
- **PCA Interpretability:** The Kaggle dataset transforms features `V1` to `V28` using PCA to preserve privacy. In production environments, domain-specific features (e.g., device fingerprinting, IP reputation, velocity checks, merchant risk category) provide richer interpretability.
- **No Sensitive Data Storage:** FinGuard AI never collects or stores primary account numbers (PAN), CVVs, or account passwords.

---

## 11. Future Roadmap

- [ ] Chronological / Temporal data split benchmarking.
- [ ] Real-time event streaming with Apache Kafka / Redpanda.
- [ ] Tree-based SHAP explainability waterfall plots for single-transaction factor attribution.
- [ ] Multi-tenant role-based access control (RBAC) with JWT authentication.
