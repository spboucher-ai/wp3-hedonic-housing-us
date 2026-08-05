#!/usr/bin/env python3
# Author: Simon-Pierre Boucher — contact@spboucher.ai
#
"""Spatial and robustness analysis (refactor of the original run_v3_spatial.py).

Produces results/v3_spatial_results.pkl with:
  1. ZIP3 fixed-effects OLS (in/out-of-sample R2)
  2. Moran's I on OLS residuals (3 random subsamples of 5,000, KNN k=8)
  3. XGBoost with latitude/longitude
  4. XGBoost without any geographic features
  5. Ablation analysis over 6 progressive feature sets

Usage:  python scripts/01_spatial_robustness.py [--output results/v3_spatial_results.pkl]
"""

import argparse
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wp3 import config, data
from wp3.models import train_xgb

warnings.filterwarnings("ignore")


def zip3_fixed_effects_ols(df, X_full, y, idx_train, idx_test):
    """OLS with 3-digit-ZIP fixed effects, evaluated on the random holdout."""
    zip3 = df["address_zipcode"].fillna("000").str[:3]
    zip3_dummies = pd.get_dummies(zip3, prefix="zip3", drop_first=True).astype(np.int8)
    X_zip3 = pd.concat([X_full, zip3_dummies], axis=1)

    ols = LinearRegression(n_jobs=-1)
    ols.fit(X_zip3.iloc[idx_train].values, y[idx_train])
    y_pred_train = ols.predict(X_zip3.iloc[idx_train].values)
    y_pred_test = ols.predict(X_zip3.iloc[idx_test].values)

    return {
        "r2_insample": r2_score(y[idx_train], y_pred_train),
        "r2_outsample": r2_score(y[idx_test], y_pred_test),
        "rmse_outsample": float(np.sqrt(mean_squared_error(y[idx_test], y_pred_test))),
        "n_zip3": int(zip3.nunique()),
        "n_features": X_zip3.shape[1],
    }


def morans_i_robustness(df, X_full, y, idx_train, n_subsamples=3, size=5000, k=8):
    """Moran's I on baseline-OLS residuals over independent random subsamples."""
    from esda.moran import Moran
    from libpysal.weights import KNN

    ols = LinearRegression(n_jobs=-1)
    ols.fit(X_full.iloc[idx_train].values, y[idx_train])
    residuals = y - ols.predict(X_full.values)

    coords = df[["latitude", "longitude"]].values
    values, pvalues = [], []
    for i in range(n_subsamples):
        rng = np.random.RandomState(config.RANDOM_STATE + i)
        idx_sub = rng.choice(len(df), size=size, replace=False)
        w = KNN.from_array(coords[idx_sub], k=k)
        w.transform = "r"
        mi = Moran(residuals[idx_sub], w)
        values.append(mi.I)
        pvalues.append(mi.p_sim)
        print(f"  Subsample {i + 1}: Moran's I = {mi.I:.4f}, p = {mi.p_sim:.4f}")

    return {
        "values": values,
        "pvalues": pvalues,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "pval_mean": float(np.mean(pvalues)),
        "n_subsamples": n_subsamples,
        "subsample_size": size,
        "k_neighbors": k,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=config.RESULTS_DIR / "v3_spatial_results.pkl")
    args = parser.parse_args()

    t0 = time.time()
    print("Loading analytical sample...")
    df = data.load_analytical_sample()
    X_full, parts = data.build_feature_matrix(df)
    y = df["ln_price"].values
    print(f"  {df.shape[0]:,} rows, {X_full.shape[1]} regressors")

    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=config.RANDOM_STATE)
    geo_mask = df["address_state"].isin(config.HOLDOUT_STATES).values
    idx_geo_train, idx_geo_test = np.where(~geo_mask)[0], np.where(geo_mask)[0]

    results = {}

    print("\n1. ZIP3 fixed-effects OLS...")
    results["zip3_fe_ols"] = zip3_fixed_effects_ols(df, X_full, y, idx_train, idx_test)
    print(f"   R2 in={results['zip3_fe_ols']['r2_insample']:.4f} "
          f"out={results['zip3_fe_ols']['r2_outsample']:.4f}")

    print("\n2. Moran's I robustness...")
    results["morans_i_robustness"] = morans_i_robustness(df, X_full, y, idx_train)

    print("\n3. XGBoost with latitude/longitude...")
    X_latlon = pd.concat([X_full, df[["latitude", "longitude"]]], axis=1)
    results["xgb_with_latlon"] = train_xgb(
        X_latlon.values.astype(np.float32), y,
        idx_train, idx_test, idx_geo_train, idx_geo_test)
    print(f"   R2 random={results['xgb_with_latlon']['r2_random']:.4f} "
          f"geo={results['xgb_with_latlon']['r2_geo']:.4f}")

    print("\n4. XGBoost without geographic features...")
    X_nogeo = pd.concat(
        [df[config.ALL_BASE_FEATS],
         parts["cat_dummies"][parts["pure_categorical_dummies"]]], axis=1)
    results["xgb_no_geography"] = train_xgb(
        X_nogeo.values.astype(np.float32), y,
        idx_train, idx_test, idx_geo_train, idx_geo_test)
    print(f"   R2 random={results['xgb_no_geography']['r2_random']:.4f} "
          f"geo={results['xgb_no_geography']['r2_geo']:.4f}")

    print("\n5. Ablation analysis...")
    ablation_specs = [
        ("1_structural", config.STRUCTURAL_FEATS, []),
        ("2_plus_lot", config.STRUCTURAL_FEATS + config.LOT_FEATS, []),
        ("3_plus_amenities",
         config.STRUCTURAL_FEATS + config.LOT_FEATS + config.AMENITY_FEATS, []),
        ("4_plus_neighborhood",
         config.STRUCTURAL_FEATS + config.LOT_FEATS + config.AMENITY_FEATS
         + config.NEIGHBORHOOD_FEATS, []),
        ("5_plus_market",
         config.STRUCTURAL_FEATS + config.LOT_FEATS + config.AMENITY_FEATS
         + config.NEIGHBORHOOD_FEATS + config.MARKET_FEATS, []),
        ("6_full_model", config.ALL_BASE_FEATS, list(parts["cat_dummies"].columns)),
    ]
    ablation = {}
    for name, feats, dummy_cols in ablation_specs:
        X_abl = (pd.concat([df[feats], parts["cat_dummies"][dummy_cols]], axis=1)
                 if dummy_cols else df[feats])
        res = train_xgb(X_abl.values.astype(np.float32), y,
                        idx_train, idx_test, idx_geo_train, idx_geo_test)
        res["features"] = list(X_abl.columns)
        ablation[name] = res
        print(f"   {name}: R2 random={res['r2_random']:.4f} geo={res['r2_geo']:.4f}")
    results["ablation"] = ablation

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved to {args.output}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
