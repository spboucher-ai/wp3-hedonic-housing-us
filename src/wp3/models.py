# Author: Simon-Pierre Boucher — contact@spboucher.ai
#
"""Shared model-training helpers."""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from .config import RANDOM_STATE

XGB_PARAMS = dict(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    early_stopping_rounds=50,
)


def _fit_and_score(X, y, idx_train, idx_test):
    """Fit XGBoost on one split (10% of train reserved for early stopping)."""
    X_tr, X_te = X[idx_train], X[idx_test]
    y_tr, y_te = y[idx_train], y[idx_test]

    X_fit, X_eval, y_fit, y_eval = train_test_split(
        X_tr, y_tr, test_size=0.1, random_state=RANDOM_STATE)

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_fit, y_fit, eval_set=[(X_eval, y_eval)], verbose=0)

    y_pred = model.predict(X_te)
    return {
        "r2": r2_score(y_te, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_te, y_pred))),
        "best_iter": getattr(model, "best_iteration", XGB_PARAMS["n_estimators"]),
    }


def train_xgb(X, y, idx_train, idx_test, idx_geo_train, idx_geo_test):
    """Train XGBoost under both validation schemes and return the metrics.

    Two independent models are fit: one on the random 80/20 split and one on
    the geographic (state-holdout) split.
    """
    random_split = _fit_and_score(X, y, idx_train, idx_test)
    geo_split = _fit_and_score(X, y, idx_geo_train, idx_geo_test)
    return {
        "r2_random": random_split["r2"],
        "rmse_random": random_split["rmse"],
        "r2_geo": geo_split["r2"],
        "rmse_geo": geo_split["rmse"],
        "best_iter_random": random_split["best_iter"],
        "best_iter_geo": geo_split["best_iter"],
        "n_features": X.shape[1],
    }
