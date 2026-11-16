
"""
ml_trainer.py – Train and compare multiple ML models for used car price prediction.
Saves the best model as models/best_model.joblib.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


FEATURE_COLS = ["Brand", "Year", "Kilometers_Driven", "Fuel_Type", "Transmission"]
TARGET_COL   = "Selling_Price"

CAT_COLS = ["Brand", "Fuel_Type", "Transmission"]
NUM_COLS = ["Year", "Kilometers_Driven"]


def _build_preprocessor():
    cat_pipe = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    num_pipe = Pipeline([("scaler", StandardScaler())])
    return ColumnTransformer([
        ("cat", cat_pipe, CAT_COLS),
        ("num", num_pipe, NUM_COLS),
    ])


def _get_models():
    return {
        "Linear Regression":        LinearRegression(),
        "Ridge Regression":         Ridge(alpha=1.0),
        "Lasso Regression":         Lasso(alpha=100),
        "Decision Tree":            DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest":            RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting":        GradientBoostingRegressor(n_estimators=200, random_state=42),
        "XGBoost":                  xgb.XGBRegressor(n_estimators=200, random_state=42, verbosity=0),
        "Extra Trees":              ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "K-Nearest Neighbors":      KNeighborsRegressor(n_neighbors=7),
        "AdaBoost":                 AdaBoostRegressor(n_estimators=100, random_state=42),
    }


def prepare_data(df):
    """Filter and return X, y ready for modelling."""
    df2 = df.copy()
    df2 = df2.dropna(subset=FEATURE_COLS + [TARGET_COL])
    # Remove rows where Transmission is 'Not available'
    df2 = df2[df2["Transmission"].str.lower() != "not available"]
    df2 = df2[df2[TARGET_COL] > 0]
    X = df2[FEATURE_COLS]
    y = df2[TARGET_COL]
    return X, y


def train_and_evaluate(df, test_size=0.2, cv_folds=5, progress_cb=None):
    """
    Train all models, return a results dict and save best model.

    Returns
    -------
    dict with keys:
        results       : list of dicts (one per model)
        best_name     : str
        best_pipeline : fitted sklearn Pipeline
        X_test, y_test, y_pred_best : arrays for residual plots
    """
    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    preprocessor = _build_preprocessor()
    models       = _get_models()
    results      = []

    best_r2       = -np.inf
    best_name     = None
    best_pipeline = None
    best_y_pred   = None

    for i, (name, estimator) in enumerate(models.items()):
        if progress_cb:
            progress_cb(i / len(models), f"Training {name}…")

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])

        # Cross-validation on training set
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv_folds, scoring="r2", n_jobs=-1)

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)
        mse  = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_test, y_pred)

        results.append({
            "Model":      name,
            "MAE":        mae,
            "RMSE":       rmse,
            "R²":         r2,
            "CV R² Mean": cv_scores.mean(),
            "CV R² Std":  cv_scores.std(),
        })

        if r2 > best_r2:
            best_r2       = r2
            best_name     = name
            best_pipeline = pipe
            best_y_pred   = y_pred

    if progress_cb:
        progress_cb(1.0, "Saving best model…")

    # Save best model
    best_path = MODEL_DIR / "best_model.joblib"
    joblib.dump(best_pipeline, best_path)

    # Also save all pipelines for per-model prediction
    all_path = MODEL_DIR / "all_models.joblib"
    # Re-train all pipelines (they're already fitted above but re-fit needed
    # since we only kept best_pipeline fitted. Workaround: store results only)
    # We'll store in a dict
    all_fitted = {}
    for name, estimator in models.items():
        pipe = Pipeline([("preprocessor", _build_preprocessor()), ("model", estimator)])
        pipe.fit(X_train, y_train)
        all_fitted[name] = pipe
    joblib.dump(all_fitted, all_path)

    return {
        "results":       results,
        "best_name":     best_name,
        "best_pipeline": best_pipeline,
        "X_test":        X_test,
        "y_test":        y_test.values,
        "y_pred_best":   best_y_pred,
        "train_size":    len(X_train),
        "test_size":     len(X_test),
        "feature_cols":  FEATURE_COLS,
    }
