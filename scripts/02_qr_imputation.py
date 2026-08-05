#!/usr/bin/env python3
# Author: Simon-Pierre Boucher — contact@spboucher.ai
#
"""Quantile-regression stability, inter-quantile tests, imputation and
winsorization sensitivity (refactor of the original v3_qr_imputation_analysis.py).

Produces results/v3_qr_imputation_results.pkl with:
  1. QR coefficient stability at tau=0.50 over 10 subsamples of 150,000
  2. Inter-quantile Wald tests (tau=0.10 vs tau=0.90)
  3. Imputation sensitivity: OLS re-estimated after dropping observations with
     originally missing year_built / lot_size_sqft (missingness read from DuckDB)
  4. Winsorization sensitivity: lot size capped at the 99.5th percentile,
     bike score capped at 100

Usage:  python scripts/02_qr_imputation.py [--output results/v3_qr_imputation_results.pkl]
"""

import argparse
import pickle
import sys
import time
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wp3 import config, data

warnings.filterwarnings("ignore")

OLS_FEATURES = ["ln_sqft", "bedrooms", "bathrooms", "age", "age_sq",
                "ln_lot", "has_pool", "has_garage", "luxury_score",
                "tag_foreclosure", "on_waterfront"]


def qr_stability(X_const, y_clean):
    """Median-regression coefficient stability across 10 random subsamples."""
    coef_matrix = {v: [] for v in config.KEY_VARS}
    for i, seed in enumerate(config.QR_STABILITY_SEEDS):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X_const), size=config.QR_SUBSAMPLE_SIZE, replace=False)
        print(f"  QR subsample {i + 1}/{len(config.QR_STABILITY_SEEDS)} (seed={seed})...")
        res = sm.QuantReg(y_clean.iloc[idx], X_const.iloc[idx]).fit(q=0.50, max_iter=1000)
        for v in config.KEY_VARS:
            coef_matrix[v].append(res.params[v])

    stability = {}
    for v, vals_list in coef_matrix.items():
        vals = np.array(vals_list)
        m, s = vals.mean(), vals.std()
        stability[v] = {
            "mean": m, "std": s,
            "cv": abs(s / m) if abs(m) > 1e-10 else np.inf,
            "min": vals.min(), "max": vals.max(),
            "sign_stability_pct": float(np.mean(np.sign(vals) == np.sign(m)) * 100),
        }
    return stability


def inter_quantile_test(X_const, y_clean):
    """Wald z-tests for coefficient equality between tau=0.10 and tau=0.90."""
    rng = np.random.RandomState(config.RANDOM_STATE)
    idx = rng.choice(len(X_const), size=config.QR_SUBSAMPLE_SIZE, replace=False)
    X_sub, y_sub = X_const.iloc[idx], y_clean.iloc[idx]

    print("  Fitting QR at tau=0.10 and tau=0.90...")
    res10 = sm.QuantReg(y_sub, X_sub).fit(q=0.10, max_iter=1000)
    res90 = sm.QuantReg(y_sub, X_sub).fit(q=0.90, max_iter=1000)

    out = {}
    for v in config.KEY_VARS:
        diff = res10.params[v] - res90.params[v]
        z = diff / np.sqrt(res10.bse[v] ** 2 + res90.bse[v] ** 2)
        out[v] = {
            "beta_010": res10.params[v], "beta_090": res90.params[v],
            "difference": diff, "z_stat": z,
            "significant_5pct": bool(abs(z) > 1.96),
        }
    return out


def load_missingness_flags():
    """Read original missingness of year_built / lot_size_sqft from the raw DB."""
    con = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
    flags = con.execute("""
        SELECT zpid,
               CASE WHEN year_built IS NULL THEN 1 ELSE 0 END AS yb_missing,
               CASE WHEN lot_size_sqft IS NULL THEN 1 ELSE 0 END AS lot_missing
        FROM properties
    """).fetchdf()
    con.close()
    flags["zpid"] = flags["zpid"].astype(np.int64)
    return flags


def build_unstandardized_matrix(df_merged, X_const):
    """Rebuild the OLS design matrix on the raw (unstandardized) scale."""
    xcols = [c for c in X_const.columns if c != "const"]
    missing = [c for c in xcols if c not in df_merged.columns]
    if missing:
        for cat_col in config.CATEGORICAL_COLS:
            if cat_col not in df_merged.columns:
                continue
            dummies = pd.get_dummies(df_merged[cat_col], prefix=cat_col)
            for dc in dummies.columns:
                if dc in missing:
                    df_merged[dc] = dummies[dc].astype(float)
    return df_merged[xcols].values.astype(np.float64), xcols


def imputation_sensitivity(df_merged, X_unstd, xcols):
    """OLS on subsets that drop originally-missing year_built / lot_size rows."""
    y = df_merged["ln_price"].values

    def run(mask, label):
        lr = LinearRegression()
        lr.fit(X_unstd[mask], y[mask])
        coefs = dict(zip(xcols, lr.coef_))
        return {"label": label, "N": int(mask.sum()),
                "R2": lr.score(X_unstd[mask], y[mask]),
                "key_coefficients": {v: coefs[v] for v in OLS_FEATURES}}

    no_yb = df_merged["yb_missing"].values == 0
    no_lot = df_merged["lot_missing"].values == 0
    scenarios = [
        (np.ones(len(df_merged), dtype=bool), "Full sample (baseline)"),
        (no_yb, "Drop missing year_built"),
        (no_lot, "Drop missing lot_size_sqft"),
        (no_yb & no_lot, "Drop both missing"),
    ]
    out = []
    for mask, label in scenarios:
        r = run(mask, label)
        out.append(r)
        print(f"  {label:<30} N={r['N']:>9,}  R2={r['R2']:.4f}")
    return out


def winsorization_sensitivity(df_merged, X_unstd, xcols):
    """Re-estimate OLS with lot size winsorized at p99.5 and bike score capped at 100."""
    df_w = df_merged.copy()
    y = df_merged["ln_price"].values

    lot_p995 = df_w["lot_size_sqft"].quantile(0.995)
    n_lot = int((df_w["lot_size_sqft"] > lot_p995).sum())
    df_w["lot_size_sqft"] = df_w["lot_size_sqft"].clip(upper=lot_p995)
    df_w["ln_lot"] = np.where(df_w["lot_size_sqft"] > 0, np.log(df_w["lot_size_sqft"]), 0.0)

    n_bike = int((df_w["bike_score"] > 100).sum())
    df_w["bike_score"] = df_w["bike_score"].clip(upper=100)

    def fit(X):
        lr = LinearRegression()
        lr.fit(X, y)
        coefs = dict(zip(xcols, lr.coef_))
        return {"R2": lr.score(X, y),
                "key_coefficients": {v: coefs[v] for v in OLS_FEATURES}}

    X_w = df_w[xcols].values.astype(np.float64)
    return {
        "baseline": fit(X_unstd),
        "winsorized": fit(X_w),
        "lot_p995": lot_p995,
        "n_lot_winsorized": n_lot,
        "n_bike_capped": n_bike,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=config.RESULTS_DIR / "v3_qr_imputation_results.pkl")
    args = parser.parse_args()

    t0 = time.time()
    print("Loading model data...")
    md = data.load_model_data()
    X_const, y_clean, df_clean = md["X_const"], md["y_clean"], md["df_clean"]

    results = {}

    print("\n1. QR stability (tau=0.50, 10 subsamples)...")
    results["qr_stability"] = qr_stability(X_const, y_clean)

    print("\n2. Inter-quantile difference test (tau=0.10 vs 0.90)...")
    results["iqr_test"] = inter_quantile_test(X_const, y_clean)

    print("\n3. Imputation sensitivity...")
    flags = load_missingness_flags()
    df_merged = df_clean.merge(flags, on="zpid", how="left")
    X_unstd, xcols = build_unstandardized_matrix(df_merged, X_const)
    results["imputation_sensitivity"] = imputation_sensitivity(df_merged, X_unstd, xcols)

    print("\n4. Winsorization sensitivity...")
    results["winsorization_sensitivity"] = winsorization_sensitivity(
        df_merged, X_unstd, xcols)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved to {args.output}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
