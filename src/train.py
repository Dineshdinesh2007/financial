"""FinGuard AI Machine Learning Training and Evaluation Pipeline.

Implements data loading, exploratory checks, leakage-free 60/20/20 stratified
splitting, model training with class weighting (Logistic Regression vs Random Forest),
validation average-precision comparison, threshold optimization on validation data,
and comprehensive final evaluation on untouched test data.
"""

from typing import Dict, Any, Tuple
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.config import (
    EVALUATION_FILE,
    FEATURE_COLUMNS,
    METADATA_FILE,
    MODEL_FILE,
    MODEL_VERSION,
    MODELS_DIR,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)
from src.data_loader import (
    compute_data_quality_summary,
    load_dataset,
    validate_dataset_schema,
)


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    val_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split dataset into stratified 60% Train, 20% Validation, and 20% untouched Test sets.

    Note on Validation Strategy:
    Stratified random splitting preserves class proportions across partitions.
    For production transaction monitoring, chronological/temporal splits (e.g. earlier days
    for train, later for test) should also be considered to test against concept drift.

    Args:
        df: Input DataFrame containing all features and target.
        test_size: Ratio for final hold-out test set (default 0.20).
        val_size: Ratio for validation set relative to full dataset (default 0.20).
        random_state: Seed for reproducibility.

    Returns:
        Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    # Step 1: Split off untouched test set (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Step 2: From remaining 80%, split 25% for validation (0.25 * 0.80 = 0.20 of total)
    val_ratio_of_temp = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_of_temp, stratify=y_temp, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def find_optimal_threshold(
    y_true: pd.Series,
    y_scores: np.ndarray,
    min_precision: float = 0.60
) -> Tuple[float, Dict[str, float]]:
    """Tune classification threshold on validation data to maximize F1-score.

    Evaluates thresholds across 0.05 to 0.95 with 0.01 step. Selects the threshold
    optimizing the F1-score on the minority fraud class.

    Args:
        y_true: Ground truth binary labels for validation set.
        y_scores: Predicted probabilities for the positive fraud class.
        min_precision: Target precision constraint if available.

    Returns:
        Tuple[float, dict]: Best threshold and associated validation metrics.
    """
    best_thresh = 0.50
    best_f1 = -1.0
    best_metrics = {}

    thresholds = np.linspace(0.05, 0.95, 91)
    for thresh in thresholds:
        preds = (y_scores >= thresh).astype(int)
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)
            best_metrics = {
                "threshold": round(float(thresh), 3),
                "val_precision": round(float(p), 4),
                "val_recall": round(float(r), 4),
                "val_f1": round(float(f1), 4),
            }

    return round(best_thresh, 3), best_metrics


def train_and_evaluate_pipeline(
    data_path: Path = RAW_DATA_PATH,
    allow_synthetic: bool = True
) -> Dict[str, Any]:
    """Execute complete end-to-end training, threshold tuning, and test evaluation.

    Args:
        data_path: CSV path to load.
        allow_synthetic: If True, generate synthetic dataset when real is absent.

    Returns:
        dict: Complete training artifacts, metrics, and metadata.
    """
    print("=" * 60)
    print("FinGuard AI — Machine Learning Pipeline Initialization")
    print("=" * 60)

    # 1. Load Data
    df, is_synthetic = load_dataset(data_path, allow_synthetic_fallback=allow_synthetic)
    data_source_label = "synthetic_demo" if is_synthetic else "kaggle_real"

    if is_synthetic:
        print("[FinGuard AI Notice] Real Kaggle dataset was not found.")
        print("[FinGuard AI Notice] Training demonstration model using synthetic data.")
        print("[FinGuard AI Notice] DO NOT cite these synthetic metrics as real-world results.")
    else:
        print(f"[FinGuard AI] Training on verified Kaggle dataset ({len(df)} transactions).")

    # 2. Schema and Data Quality Summary
    val_report = validate_dataset_schema(df, require_target=True)
    quality_summary = compute_data_quality_summary(df)
    print(f"Dataset Size: {quality_summary['total_records']} rows, {quality_summary['total_features']} features")
    if "class_distribution" in quality_summary:
        print(f"Class Distribution: {quality_summary['class_distribution']}")

    # 3. Stratified Split (60% Train, 20% Val, 20% Test)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    print(f"Split sizes -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 4. Define candidate model pipelines
    # Both use RobustScaler (less sensitive to outliers in Amount and PCA features)
    # Class weighting addresses extreme imbalance without synthetic oversampling of test/val
    candidates = {
        "LogisticRegression": Pipeline([
            ("scaler", RobustScaler()),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
                solver="lbfgs"
            )),
        ]),
        "RandomForest": Pipeline([
            ("scaler", RobustScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )),
        ]),
    }

    comparison_results = {}
    fitted_models = {}

    print("\n--- Training & Validating Candidate Models ---")
    for name, pipeline in candidates.items():
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline

        # Predict validation probabilities
        val_proba = pipeline.predict_proba(X_val)[:, 1]
        val_ap = float(average_precision_score(y_val, val_proba))
        val_roc = float(roc_auc_score(y_val, val_proba))

        opt_thresh, opt_metrics = find_optimal_threshold(y_val, val_proba)

        comparison_results[name] = {
            "validation_average_precision": round(val_ap, 4),
            "validation_roc_auc": round(val_roc, 4),
            "optimal_threshold": opt_thresh,
            "validation_metrics_at_threshold": opt_metrics,
        }
        print(f"  {name} -> Val Average Precision: {val_ap:.4f}, Val ROC-AUC: {val_roc:.4f}, Tuned Thresh: {opt_thresh}")

    # 5. Model Selection based on Validation Average Precision (PR-AUC)
    # PR-AUC is preferred over ROC-AUC for severe class imbalance
    best_model_name = max(
        comparison_results,
        key=lambda k: comparison_results[k]["validation_average_precision"]
    )
    best_pipeline = fitted_models[best_model_name]
    best_threshold = comparison_results[best_model_name]["optimal_threshold"]
    print(f"\n[FinGuard AI Selection] Selected Model: {best_model_name}")
    print(f"[FinGuard AI Selection] Tuned Threshold: {best_threshold}")

    # 6. Final Evaluation on Untouched Test Set
    print("\n--- Final Evaluation on Untouched Test Set ---")
    test_proba = best_pipeline.predict_proba(X_test)[:, 1]
    test_preds = (test_proba >= best_threshold).astype(int)

    test_precision = float(precision_score(y_test, test_preds, zero_division=0))
    test_recall = float(recall_score(y_test, test_preds, zero_division=0))
    test_f1 = float(f1_score(y_test, test_preds, zero_division=0))
    test_ap = float(average_precision_score(y_test, test_proba))
    test_roc_auc = float(roc_auc_score(y_test, test_proba))

    cm = confusion_matrix(y_test, test_preds).tolist()
    # cm is [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    clf_report = classification_report(
        y_test, test_preds,
        target_names=["Genuine", "Fraud"],
        output_dict=True,
        zero_division=0
    )

    # Compute PR curve and ROC curve sampled points for plotting
    p_curve, r_curve, _ = precision_recall_curve(y_test, test_proba)
    fpr_curve, tpr_curve, _ = roc_curve(y_test, test_proba)

    # Subsample curve points for compact JSON storage (at most 100 points)
    step_pr = max(1, len(p_curve) // 100)
    step_roc = max(1, len(fpr_curve) // 100)

    pr_curve_data = {
        "precision": [round(float(v), 4) for v in p_curve[::step_pr]],
        "recall": [round(float(v), 4) for v in r_curve[::step_pr]],
    }
    roc_curve_data = {
        "fpr": [round(float(v), 4) for v in fpr_curve[::step_roc]],
        "tpr": [round(float(v), 4) for v in tpr_curve[::step_roc]],
    }

    # Extract Feature Importances
    feature_importances = []
    classifier_step = best_pipeline.named_steps["classifier"]
    if hasattr(classifier_step, "feature_importances_"):
        raw_imp = classifier_step.feature_importances_
        for feat, imp in zip(FEATURE_COLUMNS, raw_imp):
            feature_importances.append({"feature": feat, "importance": round(float(imp), 4)})
    elif hasattr(classifier_step, "coef_"):
        raw_imp = np.abs(classifier_step.coef_[0])
        # Normalize
        total_imp = np.sum(raw_imp) if np.sum(raw_imp) > 0 else 1.0
        for feat, imp in zip(FEATURE_COLUMNS, raw_imp / total_imp):
            feature_importances.append({"feature": feat, "importance": round(float(imp), 4)})

    feature_importances.sort(key=lambda x: x["importance"], reverse=True)

    test_metrics = {
        "precision": round(test_precision, 4),
        "recall": round(test_recall, 4),
        "f1_score": round(test_f1, 4),
        "average_precision": round(test_ap, 4),
        "roc_auc": round(test_roc_auc, 4),
        "confusion_matrix": {
            "matrix": cm,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
        },
        "classification_report": clf_report,
        "pr_curve": pr_curve_data,
        "roc_curve": roc_curve_data,
    }

    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall:    {test_recall:.4f}")
    print(f"Test F1-Score:  {test_f1:.4f}")
    print(f"Test PR-AUC:    {test_ap:.4f}")
    print(f"Test ROC-AUC:   {test_roc_auc:.4f}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # 7. Save Model and Metadata Artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save fitted model
    joblib.dump(best_pipeline, MODEL_FILE)
    print(f"Saved trained model to {MODEL_FILE}")

    # Create metadata
    metadata = {
        "model_name": best_model_name,
        "model_version": MODEL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_source": data_source_label,
        "dataset_records": len(df),
        "is_synthetic": is_synthetic,
        "feature_columns": FEATURE_COLUMNS,
        "features_count": len(FEATURE_COLUMNS),
        "selected_threshold": best_threshold,
        "candidate_models_comparison": comparison_results,
        "feature_importances": feature_importances,
        "training_config": {
            "test_split": 0.20,
            "val_split": 0.20,
            "train_split": 0.60,
            "scaler": "RobustScaler",
            "selection_metric": "validation_average_precision",
            "threshold_objective": "maximize_f1_score",
        },
        "evaluation_metrics": test_metrics,
        "quality_summary": quality_summary,
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {METADATA_FILE}")

    with open(EVALUATION_FILE, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"Saved evaluation metrics to {EVALUATION_FILE}")

    print("=" * 60)
    print("Pipeline Execution Completed Successfully.")
    print("=" * 60)

    return metadata


if __name__ == "__main__":
    train_and_evaluate_pipeline()
