# Author: Simon-Pierre Boucher — contact@spboucher.ai
#
"""Loaders for the processed datasets and stored result pickles."""

import pickle

import pandas as pd

from . import config


def load_analytical_sample() -> pd.DataFrame:
    """Return the analytical sample (788,842 listings x 68 engineered columns)."""
    return pd.read_pickle(config.ANALYTICAL_SAMPLE)


def load_model_data() -> dict:
    """Return model-ready data: standardized X_const, y_clean, df_clean, OLS results."""
    with open(config.MODEL_DATA, "rb") as f:
        return pickle.load(f)


def load_shap_data() -> dict:
    """Return SHAP values for the 10,000-observation XGBoost explanation sample."""
    with open(config.SHAP_DATA, "rb") as f:
        return pickle.load(f)


def load_result(name: str) -> dict:
    """Load a stored result pickle from results/ by file name (with or without .pkl)."""
    if not name.endswith(".pkl"):
        name += ".pkl"
    with open(config.RESULTS_DIR / name, "rb") as f:
        return pickle.load(f)


def build_feature_matrix(df: pd.DataFrame):
    """Build the 62-regressor design matrix used by the tree models and FE OLS.

    Returns
    -------
    X_full : pd.DataFrame
        Base features plus one-hot categorical dummies (drop-first).
    parts : dict
        Column groups: 'region_dummies', 'pure_categorical_dummies',
        and the raw dummy frame under 'cat_dummies'.
    """
    cat_dummies = pd.get_dummies(df[config.CATEGORICAL_COLS], drop_first=True).astype(int)
    region_cols = [c for c in cat_dummies.columns if c.startswith("region_")]
    pure_cat_cols = [c for c in cat_dummies.columns if not c.startswith("region_")]

    X_full = pd.concat([df[config.ALL_BASE_FEATS], cat_dummies], axis=1)
    parts = {
        "cat_dummies": cat_dummies,
        "region_dummies": region_cols,
        "pure_categorical_dummies": pure_cat_cols,
    }
    return X_full, parts
