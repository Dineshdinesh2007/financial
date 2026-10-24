"""FinGuard AI — Intelligent Financial Transaction Fraud Detection System.

Streamlit Multi-Page Fintech Dashboard with modern UI, Plotly visualizations,
FastAPI backend communication, batch screening, human review workflow,
and model governance reporting.
"""

import io
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Ensure project root is in sys.path so 'src' package imports resolve correctly in Streamlit
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.config import (
    API_URL,
    APP_VERSION,
    FEATURE_COLUMNS,
    TIME_COLUMN,
    AMOUNT_COLUMN,
    V_COLUMNS,
    DEFAULT_DECISION_THRESHOLD,
    RAW_DATA_PATH,
    SYNTHETIC_DATA_PATH,
)
from src.data_loader import (
    get_sample_transaction,
    generate_synthetic_transactions,
    check_dataset_status,
)

# Set page configuration
st.set_page_config(
    page_title="FinGuard AI — Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Clean Fintech Styling
CUSTOM_CSS = """
<style>
    /* Global fonts and background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
    [data-testid="stSidebar"] label {
        color: #e2e8f0;
    }

    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
        background: transparent;
        padding: 8px 12px;
        border-radius: 8px;
        margin-bottom: 4px;
        transition: all 0.2s ease;
    }

    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
        background-color: #1e293b;
    }

    /* Cards and Containers */
    .fin-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -2px rgba(0, 0, 0, 0.04);
        margin-bottom: 16px;
    }

    .fin-card-header {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 6px;
    }

    .fin-card-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 4px;
    }

    .fin-card-sub {
        font-size: 0.82rem;
        color: #64748b;
    }

    /* Risk Badges */
    .badge-flagged {
        display: inline-block;
        background-color: #fef2f2;
        color: #b91c1c;
        border: 1px solid #fecaca;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
    }

    .badge-lower-risk {
        display: inline-block;
        background-color: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
    }

    .badge-pending {
        display: inline-block;
        background-color: #fffbeb;
        color: #b45309;
        border: 1px solid #fde68a;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
    }

    /* Primary buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# API Helper Functions
@st.cache_data(ttl=5)
def fetch_api_health() -> Dict[str, Any]:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {"status": "unreachable", "model_loaded": False, "database_connected": False}


@st.cache_data(ttl=10)
def fetch_model_info() -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(f"{API_URL}/model-info", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


@st.cache_data(ttl=10)
def fetch_metrics() -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(f"{API_URL}/metrics", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def fetch_transactions_page(
    page: int = 1,
    page_size: int = 20,
    status_filter: Optional[str] = None,
    review_filter: Optional[str] = None,
    search: Optional[str] = None,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {"page": page, "page_size": page_size}
    if status_filter and status_filter != "All":
        params["status_filter"] = status_filter
    if review_filter and review_filter != "All":
        params["review_filter"] = review_filter
    if search:
        params["search"] = search

    try:
        r = requests.get(f"{API_URL}/transactions", params=params, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        st.error(f"Failed to connect to FinGuard API: {e}")
    return {"items": [], "total": 0, "page": 1, "page_size": page_size, "total_pages": 1}


# Sidebar Navigation
with st.sidebar:
    st.markdown("### 🛡️ FinGuard AI")
    st.caption("Intelligent Financial Transaction Fraud Detection")
    st.markdown("---")

    nav_selection = st.radio(
        "Navigation",
        [
            "Overview",
            "Transaction Analysis",
            "Batch Detection",
            "Fraud Analytics",
            "Transaction History",
            "Model Performance",
            "Settings",
            "About Project",
        ],
        index=0,
    )

    st.markdown("---")
    # API & Model Status Indicator
    health = fetch_api_health()
    if health.get("status") == "healthy":
        st.success("🟢 API: Online | Model: Ready")
    elif health.get("status") == "degraded":
        st.warning("🟡 API: Online | Model/DB: Degraded")
    else:
        st.error("🔴 API: Offline (Run 'uvicorn src.api')")

    thresh_val = health.get("decision_threshold", DEFAULT_DECISION_THRESHOLD)
    st.caption(f"Active Threshold: **{thresh_val:.2f}** | v{APP_VERSION}")


# ==========================================
# PAGE 1: OVERVIEW DASHBOARD
# ==========================================
if nav_selection == "Overview":
    st.title("Financial Risk Overview")
    st.markdown(
        "Real-time transaction risk screening, triage metrics, and human review distribution."
    )

    # Fetch recent transactions
    tx_data = fetch_transactions_page(page=1, page_size=100)
    total_tx = tx_data.get("total", 0)
    items = tx_data.get("items", [])

    flagged_tx = sum(1 for it in items if it["prediction_status"] == "Flag for Review")
    review_rate = (flagged_tx / len(items) * 100.0) if items else 0.0
    avg_score = (sum(it["risk_score"] for it in items) / len(items)) if items else 0.0

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="fin-card">
                <div class="fin-card-header">Total Transactions</div>
                <div class="fin-card-value">{total_tx:,}</div>
                <div class="fin-card-sub">Recorded across all sessions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="fin-card">
                <div class="fin-card-header">Flagged For Review</div>
                <div class="fin-card-value" style="color: #b91c1c;">{flagged_tx:,}</div>
                <div class="fin-card-sub">Exceeded risk threshold</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="fin-card">
                <div class="fin-card-header">Review Rate</div>
                <div class="fin-card-value" style="color: #0284c7;">{review_rate:.1f}%</div>
                <div class="fin-card-sub">Workload queue proportion</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="fin-card">
                <div class="fin-card-header">Avg Risk Score</div>
                <div class="fin-card-value">{avg_score:.3f}</div>
                <div class="fin-card-sub">Screening score index</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Interactive Charts
    st.markdown("### Risk Dynamics & Distribution")
    chart_col1, chart_col2 = st.columns(2)

    if items:
        df_items = pd.DataFrame(items)

        with chart_col1:
            fig_hist = px.histogram(
                df_items,
                x="risk_score",
                nbins=20,
                color="prediction_status",
                color_discrete_map={
                    "Flag for Review": "#ef4444",
                    "Lower Predicted Risk": "#10b981",
                },
                title="Predicted Risk Score Distribution (Model Prediction)",
                labels={"risk_score": "Risk Score", "count": "Transactions"},
            )
            fig_hist.update_layout(
                plot_bgcolor="white",
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with chart_col2:
            status_counts = df_items["prediction_status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            fig_pie = px.pie(
                status_counts,
                values="Count",
                names="Status",
                hole=0.6,
                color="Status",
                color_discrete_map={
                    "Flag for Review": "#ef4444",
                    "Lower Predicted Risk": "#10b981",
                },
                title="Triage Status Proportion",
            )
            fig_pie.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    else:
        with chart_col1:
            st.info("No transaction history yet. Analyze a single transaction or upload a batch CSV to view analytics.")
        with chart_col2:
            st.info("Demonstration charts will appear as transactions are evaluated.")

    # Recent Transactions Table
    st.markdown("### Recent Transactions")
    if items:
        display_rows = []
        for it in items[:10]:
            rev_status = it["review"]["review_status"] if it.get("review") else "Pending Review"
            display_rows.append(
                {
                    "ID": it["id"],
                    "Reference": it.get("transaction_ref") or f"TX-{it['id']:05d}",
                    "Timestamp": it["timestamp"][:19].replace("T", " "),
                    "Amount ($)": f"${it['amount']:.2f}",
                    "Risk Score": f"{it['risk_score']:.4f}",
                    "Prediction Status": it["prediction_status"],
                    "Review Status": rev_status,
                }
            )
        st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)
    else:
        st.write("No transactions recorded yet.")


# ==========================================
# PAGE 2: SINGLE TRANSACTION ANALYSIS
# ==========================================
elif nav_selection == "Transaction Analysis":
    st.title("Single Transaction Risk Analysis")
    st.markdown(
        "Evaluate individual transaction features against the trained model threshold via the FastAPI backend."
    )

    # Sample Data Loader Buttons
    st.markdown("##### Quick Test Samples")
    sample_col1, sample_col2, sample_col3 = st.columns([1, 1, 2])

    if "single_tx_data" not in st.session_state:
        st.session_state.single_tx_data = get_sample_transaction(is_fraud_pattern=False)

    with sample_col1:
        if st.button("Load Benign Sample", use_container_width=True):
            st.session_state.single_tx_data = get_sample_transaction(is_fraud_pattern=False)
            st.rerun()

    with sample_col2:
        if st.button("Load Suspicious Sample", use_container_width=True):
            st.session_state.single_tx_data = get_sample_transaction(is_fraud_pattern=True)
            st.rerun()

    current_data = st.session_state.single_tx_data

    # Main Transaction Form
    with st.form("single_transaction_form"):
        st.markdown("##### Core Transaction Parameters")
        p_col1, p_col2, p_col3 = st.columns(3)

        with p_col1:
            tx_ref = st.text_input("External Transaction Reference", value="REF-98124")
        with p_col2:
            tx_amount = st.number_input(
                "Amount ($)",
                min_value=0.0,
                max_value=1000000.0,
                value=float(current_data.get(AMOUNT_COLUMN, 45.50)),
                step=1.0,
                format="%.2f",
            )
        with p_col3:
            tx_time = st.number_input(
                "Time (Elapsed Seconds)",
                min_value=0.0,
                max_value=500000.0,
                value=float(current_data.get(TIME_COLUMN, 85200.0)),
                step=10.0,
            )

        # Collapsible V1-V28 inputs
        with st.expander("Anonymized PCA Features (V1 through V28)", expanded=False):
            st.caption("PCA transformed numerical components from credit card activity.")
            v_inputs: Dict[str, float] = {}

            # Render 4 columns of 7 features each
            v_cols = st.columns(4)
            for i in range(1, 29):
                feat_name = f"V{i}"
                col_idx = (i - 1) % 4
                with v_cols[col_idx]:
                    v_inputs[feat_name] = st.number_input(
                        feat_name,
                        value=float(current_data.get(feat_name, 0.0)),
                        format="%.4f",
                        key=f"input_{feat_name}",
                    )

        submit_btn = st.form_submit_button("🛡️ Score Transaction Risk", use_container_width=True)

    if submit_btn:
        payload = {
            "Time": tx_time,
            "Amount": tx_amount,
            "transaction_ref": tx_ref if tx_ref else None,
            **v_inputs,
        }

        try:
            with st.spinner("Scoring via FinGuard API..."):
                resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)

            if resp.status_code == 200:
                result = resp.json()
                st.markdown("---")
                st.markdown("### Risk Assessment Results")

                res_col1, res_col2 = st.columns([1, 2])

                with res_col1:
                    risk_score = result["risk_score"]
                    is_flagged = result["prediction_status"] == "Flag for Review"

                    # Score Gauge
                    fig_gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=risk_score * 100.0,
                            title={"text": "Fraud Risk Index (%)"},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": "#ef4444" if is_flagged else "#10b981"},
                                "steps": [
                                    {"range": [0, 30], "color": "#ecfdf5"},
                                    {"range": [30, 70], "color": "#fffbeb"},
                                    {"range": [70, 100], "color": "#fef2f2"},
                                ],
                                "threshold": {
                                    "line": {"color": "#0f172a", "width": 3},
                                    "thickness": 0.75,
                                    "value": result["threshold"] * 100.0,
                                },
                            },
                        )
                    )
                    fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with res_col2:
                    if is_flagged:
                        st.markdown(
                            f"""
                            <div class="fin-card" style="border-left: 5px solid #ef4444;">
                                <div class="badge-flagged">⚠️ FLAG FOR REVIEW</div>
                                <h3 style="margin-top: 10px; color: #b91c1c;">Elevated Risk Score: {risk_score:.4f}</h3>
                                <p><strong>Active Threshold:</strong> {result['threshold']:.4f}</p>
                                <p><strong>Recommended Action:</strong> {result['recommended_action']}</p>
                                <p style="font-size: 0.85rem; color: #64748b;">Transaction recorded as ID #{result['transaction_id']} in review queue.</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="fin-card" style="border-left: 5px solid #10b981;">
                                <div class="badge-lower-risk">✓ LOWER PREDICTED RISK</div>
                                <h3 style="margin-top: 10px; color: #047857;">Normal Risk Score: {risk_score:.4f}</h3>
                                <p><strong>Active Threshold:</strong> {result['threshold']:.4f}</p>
                                <p><strong>Recommended Action:</strong> {result['recommended_action']}</p>
                                <p style="font-size: 0.85rem; color: #64748b;">Transaction recorded as ID #{result['transaction_id']}.</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                st.caption(f"ℹ️ {result['disclaimer']}")

            else:
                st.error(f"Prediction failed with status code {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"Connection error to FinGuard API: {e}")


# ==========================================
# PAGE 3: BATCH FRAUD DETECTION
# ==========================================
elif nav_selection == "Batch Detection":
    st.title("Batch Transaction Risk Screening")
    st.markdown(
        "Upload a CSV file containing transactions to perform high-throughput risk scoring and automated triage."
    )

    # Provide download for sample CSV
    st.markdown("##### Test with Sample Data")
    sample_csv_col1, sample_csv_col2 = st.columns([1, 3])
    with sample_csv_col1:
        demo_df = generate_synthetic_transactions(n_samples=25, fraud_ratio=0.12, random_state=77)
        demo_csv_buffer = io.StringIO()
        demo_df.drop(columns=["Class"], errors="ignore").to_csv(demo_csv_buffer, index=False)
        st.download_button(
            label="📥 Download Test CSV (25 Rows)",
            data=demo_csv_buffer.getvalue(),
            file_name="finguard_test_transactions.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Select transaction CSV file (Max 25,000 rows)",
        type=["csv"],
        help="CSV must contain Time, V1-V28, and Amount columns.",
    )

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.success(f"Loaded CSV with {len(df_upload):,} rows and {len(df_upload.columns)} columns.")

            # Validate required columns
            missing = [c for c in FEATURE_COLUMNS if c not in df_upload.columns]
            if missing:
                st.error(f"Uploaded CSV is missing required columns: {missing[:5]}")
            else:
                has_ground_truth = "Class" in df_upload.columns
                if has_ground_truth:
                    st.info(
                        "Ground truth 'Class' column detected. It will NOT be passed to the model, "
                        "and will be reserved strictly for optional retrospective comparison."
                    )

                if st.button("🚀 Execute Batch Risk Analysis", use_container_width=True):
                    # Prepare batch payload
                    rows_to_process = df_upload[FEATURE_COLUMNS].copy()
                    if "transaction_ref" in df_upload.columns:
                        rows_to_process["transaction_ref"] = df_upload["transaction_ref"]

                    records = rows_to_process.to_dict(orient="records")

                    with st.spinner("Processing batch via FastAPI backend..."):
                        progress_bar = st.progress(0.2)
                        resp = requests.post(
                            f"{API_URL}/predict/batch",
                            json={"transactions": records},
                            timeout=60,
                        )
                        progress_bar.progress(1.0)

                    if resp.status_code == 200:
                        batch_res = resp.json()
                        st.markdown("### Batch Screening Summary")

                        b1, b2, b3, b4 = st.columns(4)
                        b1.metric("Total Scored", f"{batch_res['total_transactions']:,}")
                        b2.metric("Flagged For Review", f"{batch_res['flagged_for_review_count']:,}")
                        b3.metric("Flag Rate", f"{batch_res['flag_rate']:.1f}%")
                        b4.metric("Average Risk", f"{batch_res['average_risk_score']:.4f}")

                        # Combine predictions with original dataframe
                        preds = batch_res["predictions"]
                        df_res = df_upload.copy()
                        df_res["Risk_Score"] = [p["risk_score"] for p in preds]
                        df_res["Prediction_Status"] = [p["prediction_status"] for p in preds]
                        df_res["Decision_Threshold"] = batch_res["threshold_used"]

                        # Retrospective evaluation if Class exists
                        if has_ground_truth:
                            st.markdown("#### Retrospective Evaluation (Against Uploaded Ground Truth)")
                            y_true = df_res["Class"].astype(int)
                            y_pred = (df_res["Risk_Score"] >= batch_res["threshold_used"]).astype(int)

                            from sklearn.metrics import precision_score, recall_score, f1_score
                            p_ret = precision_score(y_true, y_pred, zero_division=0)
                            r_ret = recall_score(y_true, y_pred, zero_division=0)
                            f1_ret = f1_score(y_true, y_pred, zero_division=0)

                            r_col1, r_col2, r_col3 = st.columns(3)
                            r_col1.metric("Batch Precision", f"{p_ret:.4f}")
                            r_col2.metric("Batch Recall", f"{r_ret:.4f}")
                            r_col3.metric("Batch F1-Score", f"{f1_ret:.4f}")

                        # Filters for viewing batch results
                        st.markdown("#### Filter Batch Results")
                        f_col1, f_col2 = st.columns(2)
                        with f_col1:
                            status_choice = st.selectbox(
                                "Filter by Status",
                                ["All", "Flag for Review", "Lower Predicted Risk"],
                            )
                        with f_col2:
                            min_score = st.slider("Minimum Risk Score", 0.0, 1.0, 0.0, 0.05)

                        filtered_df = df_res.copy()
                        if status_choice != "All":
                            filtered_df = filtered_df[filtered_df["Prediction_Status"] == status_choice]
                        filtered_df = filtered_df[filtered_df["Risk_Score"] >= min_score]

                        st.dataframe(
                            filtered_df[[c for c in ["Time", "Amount", "Risk_Score", "Prediction_Status"] if c in filtered_df.columns]],
                            use_container_width=True,
                        )

                        # CSV Download of Results
                        csv_download_buf = io.StringIO()
                        df_res.to_csv(csv_download_buf, index=False)
                        st.download_button(
                            label="📥 Download Full Scored Batch CSV",
                            data=csv_download_buf.getvalue(),
                            file_name=f"finguard_scored_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

                    else:
                        st.error(f"Batch prediction failed: {resp.status_code} - {resp.text}")

        except Exception as e:
            st.error(f"Error processing CSV: {e}")


# ==========================================
# PAGE 4: FRAUD ANALYTICS
# ==========================================
elif nav_selection == "Fraud Analytics":
    st.title("Transaction Fraud Analytics")
    st.markdown(
        "Interactive analytics distinguishing model risk predictions from ground-truth patterns."
    )

    tx_data = fetch_transactions_page(page=1, page_size=1000)
    items = tx_data.get("items", [])

    if not items:
        st.info("No transaction records available. Generate predictions via Single or Batch Analysis first.")
    else:
        df = pd.DataFrame(items)

        # Filters
        st.markdown("##### Filter Analytics")
        flt1, flt2, flt3 = st.columns(3)
        with flt1:
            st_flt = st.selectbox("Prediction Status", ["All", "Flag for Review", "Lower Predicted Risk"])
        with flt2:
            min_amt, max_amt = float(df["amount"].min()), float(df["amount"].max())
            amt_range = st.slider("Amount Range ($)", min_amt, max(max_amt, 10.0), (min_amt, max_amt))
        with flt3:
            score_range = st.slider("Risk Score Range", 0.0, 1.0, (0.0, 1.0))

        # Apply filters
        dff = df.copy()
        if st_flt != "All":
            dff = dff[dff["prediction_status"] == st_flt]
        dff = dff[(dff["amount"] >= amt_range[0]) & (dff["amount"] <= amt_range[1])]
        dff = dff[(dff["risk_score"] >= score_range[0]) & (dff["risk_score"] <= score_range[1])]

        st.caption(f"Showing **{len(dff)}** of **{len(df)}** transactions.")

        # Analytics Charts
        c1, c2 = st.columns(2)
        with c1:
            fig_amt = px.box(
                dff,
                x="prediction_status",
                y="amount",
                color="prediction_status",
                color_discrete_map={"Flag for Review": "#ef4444", "Lower Predicted Risk": "#10b981"},
                title="Transaction Amount Distribution by Prediction Status",
                labels={"amount": "Amount ($)", "prediction_status": "Status"},
            )
            fig_amt.update_layout(plot_bgcolor="white")
            st.plotly_chart(fig_amt, use_container_width=True)

        with c2:
            # Risk bands
            bins = [0.0, 0.25, 0.50, 0.75, 1.0]
            labels = ["Low (0-25%)", "Moderate (25-50%)", "High (50-75%)", "Critical (75-100%)"]
            dff["Risk Band"] = pd.cut(dff["risk_score"], bins=bins, labels=labels, include_lowest=True)
            band_counts = dff["Risk Band"].value_counts().reset_index()
            band_counts.columns = ["Risk Band", "Transactions"]

            fig_bar = px.bar(
                band_counts,
                x="Risk Band",
                y="Transactions",
                color="Risk Band",
                title="Risk Severity Bands (Model Predictions)",
            )
            fig_bar.update_layout(plot_bgcolor="white")
            st.plotly_chart(fig_bar, use_container_width=True)


# ==========================================
# PAGE 5: TRANSACTION HISTORY & REVIEW
# ==========================================
elif nav_selection == "Transaction History":
    st.title("Transaction History & Review Workflow")
    st.markdown(
        "Audit trail of transactions, model scores, and human fraud review workflow."
    )

    # Search and Filter Bar
    h_col1, h_col2, h_col3 = st.columns([2, 1, 1])
    with h_col1:
        search_query = st.text_input("Search by Transaction ID or Reference", placeholder="e.g. 15 or REF-98124")
    with h_col2:
        pred_filter = st.selectbox("Prediction Status", ["All", "Flag for Review", "Lower Predicted Risk"])
    with h_col3:
        rev_filter = st.selectbox("Human Review Status", ["All", "Pending Review", "Reviewed", "Confirmed Fraud", "Legitimate"])

    # Pagination controls
    page_col1, page_col2 = st.columns([1, 4])
    with page_col1:
        curr_page = st.number_input("Page", min_value=1, value=1, step=1)

    tx_res = fetch_transactions_page(
        page=curr_page,
        page_size=15,
        status_filter=pred_filter,
        review_filter=rev_filter,
        search=search_query if search_query else None,
    )

    items = tx_res.get("items", [])
    total = tx_res.get("total", 0)
    total_pages = tx_res.get("total_pages", 1)

    st.caption(f"Showing page **{curr_page}** of **{total_pages}** (Total: **{total}** transactions)")

    if items:
        # Render Table
        table_rows = []
        for it in items:
            rev_status = it["review"]["review_status"] if it.get("review") else "Pending Review"
            table_rows.append(
                {
                    "ID": it["id"],
                    "Ref": it.get("transaction_ref") or "-",
                    "Timestamp": it["timestamp"][:19].replace("T", " "),
                    "Amount": f"${it['amount']:.2f}",
                    "Risk Score": f"{it['risk_score']:.4f}",
                    "Status": it["prediction_status"],
                    "Review Outcome": rev_status,
                }
            )
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        # Human Review Action Section
        st.markdown("---")
        st.markdown("#### Update Fraud Analyst Decision")
        st.caption(
            "Note: 'Confirmed Fraud' and 'Legitimate' are human review outcomes, "
            "never automatically assigned by the model."
        )

        with st.form("update_review_form"):
            selected_tx_id = st.selectbox(
                "Select Transaction ID to Review",
                [it["id"] for it in items],
            )
            r_action_col1, r_action_col2 = st.columns(2)
            with r_action_col1:
                new_rev_status = st.selectbox(
                    "New Review Status",
                    ["Pending Review", "Reviewed", "Confirmed Fraud", "Legitimate"],
                )
            with r_action_col2:
                analyst_notes = st.text_input("Reviewer Notes", placeholder="e.g. Cardholder confirmed authorized purchase.")

            save_rev_btn = st.form_submit_button("💾 Save Review Decision")

        if save_rev_btn:
            try:
                patch_resp = requests.patch(
                    f"{API_URL}/transactions/{selected_tx_id}/review",
                    json={"review_status": new_rev_status, "reviewer_notes": analyst_notes},
                    timeout=5,
                )
                if patch_resp.status_code == 200:
                    st.success(f"Transaction #{selected_tx_id} updated to '{new_rev_status}'.")
                    st.rerun()
                else:
                    st.error(f"Failed to update review: {patch_resp.text}")
            except Exception as e:
                st.error(f"Error connecting to API: {e}")
    else:
        st.info("No matching transactions found in history.")


# ==========================================
# PAGE 6: MODEL PERFORMANCE
# ==========================================
elif nav_selection == "Model Performance":
    st.title("Model Performance & Governance")
    st.markdown(
        "Evaluation metrics from the held-out untouched test set, threshold trade-offs, and feature importance."
    )

    metrics_resp = fetch_metrics()
    model_info = fetch_model_info()

    if not metrics_resp or not model_info:
        st.warning("Model metrics are not available. Train the model via 'python -m src.train'.")
    else:
        m = metrics_resp.get("metrics", {})
        is_synth = metrics_resp.get("is_synthetic", False)
        ds_source = metrics_resp.get("dataset_source", "unknown")

        if is_synth:
            st.warning(
                "⚠️ Notice: Metrics below are from the DEMO SYNTHETIC DATASET because "
                "the real Kaggle creditcard.csv was not detected in data/raw/.\n"
                "To evaluate on real-world data, download the Kaggle dataset to data/raw/creditcard.csv "
                "and re-run 'python -m src.train'."
            )

        # Primary KPI Cards
        pk1, pk2, pk3, pk4, pk5 = st.columns(5)
        pk1.metric("Precision", f"{m.get('precision', 0):.4f}")
        pk2.metric("Recall", f"{m.get('recall', 0):.4f}")
        pk3.metric("F1-Score", f"{m.get('f1_score', 0):.4f}")
        pk4.metric("PR-AUC (Avg Precision)", f"{m.get('average_precision', 0):.4f}")
        pk5.metric("ROC-AUC", f"{m.get('roc_auc', 0):.4f}")

        # Accuracy Misleading Explanation Alert
        st.info(
            "💡 **Why Precision, Recall, and PR-AUC Matter Over Accuracy in Fraud Detection:**\n\n"
            "In credit card fraud datasets, typically over 99.8% of transactions are genuine. "
            "A naive model predicting *every* transaction as genuine would achieve **99.8% accuracy**, "
            "yet detect **zero** fraud cases (0% Recall, 0% F1). Therefore, we prioritize **Average Precision (PR-AUC)** "
            "and tune the decision threshold specifically to maximize minority fraud detection."
        )

        st.markdown("### Test Evaluation Visualizations")
        p_col1, p_col2 = st.columns(2)

        # Confusion Matrix Heatmap
        cm_data = m.get("confusion_matrix", {})
        matrix = cm_data.get("matrix", [[0, 0], [0, 0]])
        with p_col1:
            cm_z = [[matrix[0][0], matrix[0][1]], [matrix[1][0], matrix[1][1]]]
            cm_text = [
                [f"TN: {matrix[0][0]}<br>(Genuine Correct)", f"FP: {matrix[0][1]}<br>(False Alarms)"],
                [f"FN: {matrix[1][0]}<br>(Missed Fraud)", f"TP: {matrix[1][1]}<br>(Fraud Caught)"],
            ]
            fig_cm = ff_cm = go.Figure(
                data=go.Heatmap(
                    z=cm_z,
                    x=["Predicted Genuine", "Predicted Fraud (Flagged)"],
                    y=["Actual Genuine", "Actual Fraud"],
                    text=cm_text,
                    texttemplate="%{text}",
                    colorscale="Blues",
                    showscale=False,
                )
            )
            fig_cm.update_layout(
                title="Confusion Matrix (Untouched Hold-Out Test Set)",
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        # Precision-Recall Curve
        pr_curve = m.get("pr_curve", {})
        with p_col2:
            if pr_curve and "recall" in pr_curve:
                fig_pr = px.line(
                    x=pr_curve["recall"],
                    y=pr_curve["precision"],
                    title=f"Precision-Recall Curve (AP = {m.get('average_precision', 0):.4f})",
                    labels={"x": "Recall", "y": "Precision"},
                )
                fig_pr.update_layout(height=350, plot_bgcolor="white", margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_pr, use_container_width=True)
            else:
                st.write("Curve points not recorded.")

        # Candidate Models Comparison Table
        st.markdown("### Candidate Models Validation Comparison")
        comp = model_info.get("candidate_models_comparison", {})
        if comp:
            comp_rows = []
            for name, stats in comp.items():
                comp_rows.append(
                    {
                        "Model": name,
                        "Val Average Precision": stats.get("validation_average_precision"),
                        "Val ROC-AUC": stats.get("validation_roc_auc"),
                        "Tuned Threshold": stats.get("optimal_threshold"),
                    }
                )
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

        # Feature Importances
        st.markdown("### Top Predictive Features")
        feat_imp = model_info.get("feature_importances", [])
        if feat_imp:
            df_imp = pd.DataFrame(feat_imp[:12])
            fig_imp = px.bar(
                df_imp,
                x="importance",
                y="feature",
                orientation="h",
                title="Top 12 Features by Relative Model Importance",
                labels={"importance": "Relative Importance", "feature": "Feature"},
            )
            fig_imp.update_layout(yaxis=dict(autorange="reversed"), plot_bgcolor="white", height=380)
            st.plotly_chart(fig_imp, use_container_width=True)


# ==========================================
# PAGE 7: SETTINGS
# ==========================================
elif nav_selection == "Settings":
    st.title("System Settings & Threshold Tuning")
    st.markdown(
        "Configure inference thresholds, monitor dataset readiness, and manage demonstration state."
    )

    health = fetch_api_health()
    ds_status = check_dataset_status()

    # Model & Dataset Status Cards
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        st.markdown("##### Machine Learning Model State")
        st.write(f"- **Active Model:** {health.get('model_name', 'Not Loaded')}")
        st.write(f"- **Model Version:** {health.get('model_version', 'N/A')}")
        st.write(f"- **Application Version:** {APP_VERSION}")
        st.write(f"- **Backend API URL:** {API_URL}")

    with s_col2:
        st.markdown("##### Dataset Repository Status")
        if ds_status["exists"]:
            st.success(f"✓ Real Kaggle dataset detected ({ds_status['file_size_mb']} MB).")
        else:
            st.warning("⚠️ Kaggle creditcard.csv missing. System running in labeled demo mode.")
            st.caption(
                "To use real data: Download creditcard.csv from Kaggle and place it in `data/raw/creditcard.csv`."
            )

    st.markdown("---")
    st.markdown("### Configurable Decision Threshold")
    st.markdown(
        "Adjusting the threshold directly alters the sensitivity of the fraud screening system.\n\n"
        "- **Lower threshold (e.g. 0.15):** Flags more suspicious cases, catching subtle fraud at the expense of more manual review false alarms.\n"
        "- **Higher threshold (e.g. 0.60):** Flags fewer cases, reducing analyst workload but potentially missing edge-case fraud."
    )

    curr_thresh = float(health.get("decision_threshold", DEFAULT_DECISION_THRESHOLD))
    new_thresh = st.slider("Active Decision Threshold", min_value=0.05, max_value=0.95, value=curr_thresh, step=0.01)

    if st.button("Apply New Decision Threshold", use_container_width=True):
        try:
            r = requests.patch(f"{API_URL}/settings/threshold", json={"threshold": new_thresh}, timeout=5)
            if r.status_code == 200:
                st.success(f"Successfully updated active decision threshold to {new_thresh:.2f}.")
                st.rerun()
            else:
                st.error(f"Failed to update threshold: {r.text}")
        except Exception as e:
            st.error(f"Error connecting to API: {e}")

    st.markdown("---")
    st.markdown("### Demonstration History Reset")
    st.caption("Clear recorded predictions and review history for a fresh demonstration session.")

    confirm_reset = st.checkbox("I confirm that I want to clear all stored demo transactions.")
    if st.button("🗑️ Clear Demo Transaction History", type="secondary"):
        if not confirm_reset:
            st.error("Please check the confirmation box above before clearing history.")
        else:
            try:
                r = requests.delete(f"{API_URL}/history?confirm=true", timeout=5)
                if r.status_code == 200:
                    st.success("Successfully reset transaction history.")
                    st.rerun()
                else:
                    st.error(f"Failed to reset: {r.text}")
            except Exception as e:
                st.error(f"Error connecting to API: {e}")


# ==========================================
# PAGE 8: ABOUT PROJECT
# ==========================================
elif nav_selection == "About Project":
    st.title("About FinGuard AI")
    st.markdown("#### Intelligent Financial Transaction Fraud Detection System")

    st.markdown(
        """
        ### Project Mission & Problem Statement
        Financial fraud causes tens of billions of dollars in annual losses worldwide. In digital payment
        networks, billions of transactions occur daily, of which only a tiny fraction (typically 0.1% to 0.2%)
        are fraudulent.
        
        FinGuard AI demonstrates an end-to-end, reproducible, leakage-free machine learning system designed
        to score transaction risk, prioritize suspect payments for human fraud analysts, and record triage decisions.

        ### Educational Screening Disclaimer
        > **Important Disclaimer:** FinGuard AI is an educational screening prototype. It does **not**
        > automatically freeze bank accounts or prove criminal misconduct. Predictions represent a statistical risk
        > score to assist qualified fraud analysts in manual verification.

        ### Technical Architecture
        1. **Data Pipeline:** Rigorous stratified splitting (60% Train, 20% Validation, 20% Test) ensuring no data leakage.
        2. **Preprocessing:** `RobustScaler` handling extreme outliers in transaction amounts and PCA features.
        3. **Imbalance Handling:** Balanced class weighting in Logistic Regression and Random Forest classifiers.
        4. **Model Selection:** Validation Average Precision (PR-AUC) optimization and systematic F1-score threshold tuning.
        5. **Backend API:** FastAPI REST service with Pydantic validation, structured logging, and SQLite persistence.
        6. **Frontend Dashboard:** Modern Streamlit interface with Plotly charts and custom fintech styling.

        ### Dataset Information
        - **Source:** Kaggle Credit Card Fraud Detection Dataset (ULB Machine Learning Group).
        - **Features:** 28 PCA-transformed numerical features (`V1` to `V28`) to preserve customer confidentiality, plus `Time` and `Amount`.
        - **Target:** `Class` (0 = Genuine, 1 = Fraudulent).
        - **Dimensions:** 284,807 transactions with 492 frauds (0.172% fraud incidence).

        ### Ethical & Governance Considerations
        - **Human in the Loop:** High-stakes fraud decisions require analyst oversight rather than autonomous automated denials.
        - **Transparency:** The system provides transparent decision thresholds and feature importance breakdowns.
        - **Data Privacy:** Raw cardholder credentials (PAN, CVV, passwords) are never requested or stored.
        """
    )
