"""FinGuard AI Exploratory Data Analysis (EDA) Module.

Computes descriptive statistics, class imbalance metrics, amount distributions,
and correlation summaries for transaction data.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd

from src.config import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    AMOUNT_COLUMN,
    TIME_COLUMN,
    V_COLUMNS,
)


def run_exploratory_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform exploratory data analysis on a transaction dataset.

    Args:
        df: Pandas DataFrame containing transactions.

    Returns:
        dict: Comprehensive summary of dataset properties, class distribution,
              amount statistics, time range, and top correlations with fraud.
    """
    total_records = len(df)
    has_target = TARGET_COLUMN in df.columns

    eda_report: Dict[str, Any] = {
        "dimensions": {
            "rows": total_records,
            "columns": len(df.columns),
            "features_count": len(FEATURE_COLUMNS),
        },
        "time_stats": {},
        "amount_stats_overall": {},
        "amount_stats_by_class": {},
        "class_distribution": {},
        "top_feature_correlations": [],
    }

    # Time statistics (Time is elapsed seconds from first transaction in dataset)
    if TIME_COLUMN in df.columns:
        t = df[TIME_COLUMN]
        duration_hours = (t.max() - t.min()) / 3600.0 if total_records > 0 else 0.0
        eda_report["time_stats"] = {
            "min_seconds": float(t.min()),
            "max_seconds": float(t.max()),
            "duration_hours": round(float(duration_hours), 2),
        }

    # Overall amount statistics
    if AMOUNT_COLUMN in df.columns:
        amt = df[AMOUNT_COLUMN]
        eda_report["amount_stats_overall"] = {
            "min": float(amt.min()),
            "max": float(amt.max()),
            "mean": round(float(amt.mean()), 2),
            "median": round(float(amt.median()), 2),
            "std": round(float(amt.std()), 2),
            "p25": round(float(amt.quantile(0.25)), 2),
            "p75": round(float(amt.quantile(0.75)), 2),
            "p95": round(float(amt.quantile(0.95)), 2),
            "p99": round(float(amt.quantile(0.99)), 2),
        }

    # Breakdown by Class if target is present
    if has_target:
        counts = df[TARGET_COLUMN].value_counts()
        genuine_n = int(counts.get(0, 0))
        fraud_n = int(counts.get(1, 0))
        fraud_pct = (fraud_n / total_records * 100.0) if total_records > 0 else 0.0

        eda_report["class_distribution"] = {
            "genuine_count": genuine_n,
            "fraud_count": fraud_n,
            "fraud_percentage": round(fraud_pct, 4),
            "genuine_percentage": round(100.0 - fraud_pct, 4),
            "imbalance_ratio": f"1 : {round(genuine_n / fraud_n, 1)}" if fraud_n > 0 else "N/A",
        }

        if AMOUNT_COLUMN in df.columns:
            gen_amt = df[df[TARGET_COLUMN] == 0][AMOUNT_COLUMN]
            frd_amt = df[df[TARGET_COLUMN] == 1][AMOUNT_COLUMN]

            eda_report["amount_stats_by_class"] = {
                "genuine": {
                    "count": len(gen_amt),
                    "mean": round(float(gen_amt.mean()), 2) if len(gen_amt) > 0 else 0.0,
                    "median": round(float(gen_amt.median()), 2) if len(gen_amt) > 0 else 0.0,
                    "max": round(float(gen_amt.max()), 2) if len(gen_amt) > 0 else 0.0,
                    "min": round(float(gen_amt.min()), 2) if len(gen_amt) > 0 else 0.0,
                },
                "fraud": {
                    "count": len(frd_amt),
                    "mean": round(float(frd_amt.mean()), 2) if len(frd_amt) > 0 else 0.0,
                    "median": round(float(frd_amt.median()), 2) if len(frd_amt) > 0 else 0.0,
                    "max": round(float(frd_amt.max()), 2) if len(frd_amt) > 0 else 0.0,
                    "min": round(float(frd_amt.min()), 2) if len(frd_amt) > 0 else 0.0,
                },
            }

        # Feature correlation with Class
        corrs = []
        for col in FEATURE_COLUMNS:
            if col in df.columns:
                corr_val = df[col].corr(df[TARGET_COLUMN])
                if not np.isnan(corr_val):
                    corrs.append({"feature": col, "correlation": round(float(corr_val), 4)})
        corrs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        eda_report["top_feature_correlations"] = corrs[:10]

    return eda_report
