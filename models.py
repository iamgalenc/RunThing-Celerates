"""
models.py
=========
Machine-learning pipelines for:
  A. Pace Prediction   – regression (Linear Regression, Random Forest)
  B. Run Classification – classification (Logistic Regression, Random Forest)

Each function returns a structured result dict so the Streamlit app can
display metrics and plots without knowing sklearn internals.
"""

import numpy as np
import pandas as pd
import joblib
import os

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline

from feature_engineering import RUN_TYPE_MAP, RUN_TYPE_INV

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGRESSION_FEATURES = [
    "total_dist_km",
    "total_elevation_gain_m",
    "elev_gain_per_km",
    "run_type_code",
    "avg_pace_variability",
    "fatigue_index",
]

CLASSIFICATION_FEATURES = [
    "total_dist_km",
    "total_elevation_gain_m",
    "avg_pace_min_km",
    "std_pace_min_km",
    "avg_pace_variability",
    "fatigue_index",
    "total_duration_min",
    "elev_gain_per_km",
]

REGRESSION_TARGET = "avg_pace_min_km"
CLASSIFICATION_TARGET = "run_type_code"

MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split(X, y, test_size=0.25, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# ---------------------------------------------------------------------------
# A. Pace Prediction (Regression)
# ---------------------------------------------------------------------------

def train_pace_regression(
    df: pd.DataFrame,
    model_name: str = "random_forest",
    rf_n_estimators: int = 100,
    rf_max_depth: int = None,
    gb_n_estimators: int = 100,
    gb_max_depth: int = 3,
    gb_learning_rate: float = 0.1,
    ridge_alpha: float = 1.0,
    test_size: float = 0.25,
) -> dict:
    """
    Train a regression model to predict avg pace from run features.

    Parameters
    ----------
    df            : feature matrix (one row per run)
    model_name    : "linear_regression" | "random_forest" | "ridge" | "gradient_boosting"
    rf_n_estimators, rf_max_depth : RF hyperparams
    gb_n_estimators, gb_max_depth, gb_learning_rate : GB hyperparams
    ridge_alpha   : Ridge regularisation alpha
    test_size     : fraction of data held out for evaluation

    Returns
    -------
    dict with keys:
        model_name, pipeline, features, X_test, y_test, y_pred,
        mae, rmse, r2, mape, feature_importances,
        train_size, test_size_n, model_path
    """
    available = [f for f in REGRESSION_FEATURES if f in df.columns]
    df_clean = df[available + [REGRESSION_TARGET]].dropna()

    if len(df_clean) < 4:
        raise ValueError(f"Need at least 4 runs for regression, got {len(df_clean)}.")

    X = df_clean[available].values
    y = df_clean[REGRESSION_TARGET].values

    X_train, X_test, y_train, y_test = _split(X, y, test_size=test_size)

    # Build sklearn Pipeline with scaler + estimator
    if model_name == "linear_regression":
        estimator = LinearRegression()
    elif model_name == "ridge":
        estimator = Ridge(alpha=ridge_alpha)
    elif model_name == "gradient_boosting":
        estimator = GradientBoostingRegressor(
            n_estimators=gb_n_estimators,
            max_depth=gb_max_depth,
            learning_rate=gb_learning_rate,
            random_state=42,
        )
    else:
        estimator = RandomForestRegressor(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            random_state=42,
        )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    # Feature importances (RF and GB only)
    feat_imp = None
    if model_name in ("random_forest", "gradient_boosting"):
        feat_imp = pd.Series(
            pipeline.named_steps["model"].feature_importances_,
            index=available,
        ).sort_values(ascending=False)

    # Persist model
    model_path = os.path.join(MODEL_DIR, f"pace_regression_{model_name}.joblib")
    joblib.dump(pipeline, model_path)

    return {
        "model_name": model_name,
        "pipeline": pipeline,
        "features": available,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "feature_importances": feat_imp,
        "train_size": len(X_train),
        "test_size_n": len(X_test),
        "model_path": model_path,
    }


def predict_pace(pipeline, feature_values: dict, features: list) -> float:
    """
    Predict avg pace for a hypothetical run.

    Parameters
    ----------
    pipeline      : fitted sklearn Pipeline
    feature_values: dict mapping feature name → value
    features      : ordered list of feature names used during training

    Returns
    -------
    Predicted pace in min/km
    """
    X = np.array([[feature_values.get(f, 0) for f in features]])
    return float(pipeline.predict(X)[0])


# ---------------------------------------------------------------------------
# B. Run Classification
# ---------------------------------------------------------------------------

def train_run_classifier(
    df: pd.DataFrame,
    model_name: str = "random_forest",
    rf_n_estimators: int = 100,
    rf_max_depth: int = None,
    lr_C: float = 1.0,
    gb_n_estimators: int = 100,
    gb_max_depth: int = 3,
    gb_learning_rate: float = 0.1,
    svm_C: float = 1.0,
    svm_kernel: str = "rbf",
    dt_max_depth: int = None,
    test_size: float = 0.25,
) -> dict:
    """
    Train a classifier to predict run type (easy/tempo/interval/long).

    Returns
    -------
    dict with keys:
        model_name, pipeline, features, X_test, y_test, y_pred,
        accuracy, macro_precision, macro_recall, macro_f1,
        conf_matrix, class_report, feature_importances,
        train_size, test_size_n, class_names, unique_classes, model_path
    """
    available = [f for f in CLASSIFICATION_FEATURES if f in df.columns]
    df_clean = df[available + [CLASSIFICATION_TARGET]].dropna()

    if len(df_clean) < 4:
        raise ValueError(f"Need at least 4 runs for classification, got {len(df_clean)}.")

    X = df_clean[available].values
    y = df_clean[CLASSIFICATION_TARGET].values.astype(int)

    X_train, X_test, y_train, y_test = _split(X, y, test_size=test_size)

    if model_name == "logistic_regression":
        estimator = LogisticRegression(C=lr_C, max_iter=1000, random_state=42)
    elif model_name == "gradient_boosting":
        estimator = GradientBoostingClassifier(
            n_estimators=gb_n_estimators,
            max_depth=gb_max_depth,
            learning_rate=gb_learning_rate,
            random_state=42,
        )
    elif model_name == "svm":
        estimator = SVC(C=svm_C, kernel=svm_kernel, probability=True, random_state=42)
    elif model_name == "decision_tree":
        estimator = DecisionTreeClassifier(max_depth=dt_max_depth, random_state=42)
    else:
        estimator = RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            random_state=42,
        )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Present classes as strings for readability
    unique_classes = sorted(set(y_train.tolist() + y_test.tolist()))
    class_names = [RUN_TYPE_INV.get(c, str(c)) for c in unique_classes]

    accuracy = accuracy_score(y_test, y_pred)
    macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    conf = confusion_matrix(y_test, y_pred, labels=unique_classes)
    report = classification_report(
        y_test, y_pred,
        labels=unique_classes,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    # Feature importances (RF, GB, and DT only)
    feat_imp = None
    if model_name in ("random_forest", "gradient_boosting", "decision_tree"):
        feat_imp = pd.Series(
            pipeline.named_steps["model"].feature_importances_,
            index=available,
        ).sort_values(ascending=False)

    # Persist
    model_path = os.path.join(MODEL_DIR, f"run_classifier_{model_name}.joblib")
    joblib.dump(pipeline, model_path)

    return {
        "model_name": model_name,
        "pipeline": pipeline,
        "features": available,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "conf_matrix": conf,
        "class_report": report,
        "feature_importances": feat_imp,
        "train_size": len(X_train),
        "test_size_n": len(X_test),
        "class_names": class_names,
        "unique_classes": unique_classes,
        "model_path": model_path,
    }


def classify_run(pipeline, feature_values: dict, features: list) -> tuple[str, dict]:
    """
    Classify a hypothetical run.

    Returns (predicted_label_int, probability_dict)
    """
    X = np.array([[feature_values.get(f, 0) for f in features]])
    pred_code = int(pipeline.predict(X)[0])
    pred_label = RUN_TYPE_INV.get(pred_code, str(pred_code))

    proba = {}
    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        probs = pipeline.predict_proba(X)[0]
        classes = pipeline.classes_
        proba = {RUN_TYPE_INV.get(int(c), str(c)): float(p) for c, p in zip(classes, probs)}

    return pred_label, proba


# ---------------------------------------------------------------------------
# Load persisted model
# ---------------------------------------------------------------------------

def load_model(model_path: str):
    """Load a previously saved joblib pipeline."""
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None
