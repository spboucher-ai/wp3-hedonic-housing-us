#!/usr/bin/env python3
# Author: Simon-Pierre Boucher — contact@spboucher.ai
#
"""Export the key result tables of the paper as CSV files into results/tables/.

Each CSV mirrors a table in the paper and is generated directly from the
stored result pickles, providing a machine-readable audit trail between the
pickles and the numbers reported in the LaTeX source.

Usage:  python scripts/04_export_tables.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wp3 import config, data

OUT = config.RESULTS_DIR / "tables"


def export_qr_coefficients():
    """Quantile regression coefficients across taus (paper Table: QR results)."""
    qr = data.load_result("qr_results.pkl")
    taus = sorted(qr.keys())
    rows = {}
    for t in taus:
        rows[f"tau_{t}"] = pd.Series(qr[t]["params"])
    df = pd.DataFrame(rows)
    df.index.name = "variable"
    df.to_csv(OUT / "qr_coefficients.csv")
    pd.DataFrame({f"tau_{t}": {"pseudo_r2": qr[t]["pseudo_r2"]} for t in taus}) \
        .to_csv(OUT / "qr_pseudo_r2.csv")


def export_iqr_tests():
    """Inter-quantile Wald tests, tau=0.10 vs tau=0.90."""
    r = data.load_result("v3_qr_imputation_results.pkl")
    df = pd.DataFrame(r["iqr_test"]).T
    df.index.name = "variable"
    df.to_csv(OUT / "iqr_wald_tests.csv")


def export_qr_stability():
    """QR subsample stability (CV, sign stability)."""
    r = data.load_result("v3_qr_imputation_results.pkl")
    df = pd.DataFrame(r["qr_stability"]).T
    df.index.name = "variable"
    df.to_csv(OUT / "qr_stability.csv")


def export_imputation_winsorization():
    """Imputation and winsorization sensitivity."""
    r = data.load_result("v3_qr_imputation_results.pkl")
    imp = pd.DataFrame([
        {"scenario": s["label"], "N": s["N"], "R2": s["R2"],
         **{f"beta_{k}": v for k, v in s["key_coefficients"].items()}}
        for s in r["imputation_sensitivity"]
    ])
    imp.to_csv(OUT / "imputation_sensitivity.csv", index=False)

    w = r["winsorization_sensitivity"]
    win = pd.DataFrame({
        "baseline": {"R2": w["baseline"]["R2"], **w["baseline"]["key_coefficients"]},
        "winsorized": {"R2": w["winsorized"]["R2"], **w["winsorized"]["key_coefficients"]},
    })
    win.index.name = "metric"
    win.to_csv(OUT / "winsorization_sensitivity.csv")


def export_spatial_and_ablation():
    """ZIP3 FE, Moran robustness, XGBoost geography variants, ablation."""
    v3 = data.load_result("v3_spatial_results.pkl")

    pd.Series(v3["zip3_fe_ols"]).to_csv(OUT / "zip3_fixed_effects.csv")

    m = v3["morans_i_robustness"]
    pd.DataFrame({"morans_i": m["values"], "p_value": m["pvalues"]}) \
        .to_csv(OUT / "morans_i_robustness.csv", index_label="subsample")

    rows = []
    for name in ["xgb_with_latlon", "xgb_no_geography"]:
        rows.append({"model": name, **{k: v for k, v in v3[name].items()}})
    for stage, res in v3["ablation"].items():
        rows.append({"model": f"ablation_{stage}",
                     **{k: v for k, v in res.items() if k != "features"}})
    pd.DataFrame(rows).to_csv(OUT / "xgb_geography_and_ablation.csv", index=False)


def export_shap():
    """Mean |SHAP| importance and cross-model stability."""
    sd = data.load_shap_data()
    pd.Series(sd["mean_shap"], name="mean_abs_shap") \
        .sort_values(ascending=False) \
        .to_csv(OUT / "shap_importance_xgb.csv", index_label="variable")

    st = data.load_result("v3_shap_stability.pkl")
    pd.DataFrame({
        "pair": ["xgb_lgb", "xgb_rf", "lgb_rf"],
        "spearman_rho": [float(st["rho_xgb_lgb"]), float(st["rho_xgb_rf"]),
                         float(st["rho_lgb_rf"])],
    }).to_csv(OUT / "shap_cross_model_stability.csv", index=False)


def export_ols():
    """Standardized and unstandardized OLS coefficients."""
    md = data.load_model_data()
    ols = md["ols_results"]
    df = pd.DataFrame({"coef": ols["params"], "se": ols["bse"],
                       "t": ols["tvalues"], "p": ols["pvalues"]})
    df.index.name = "variable"
    df.to_csv(OUT / "ols_standardized.csv")

    ext = data.load_result("extended_results.pkl")
    df_u = pd.DataFrame({"coef": ext["unstd_params"], "se": ext["unstd_bse"],
                         "t": ext["unstd_tvalues"], "p": ext["unstd_pvalues"]})
    df_u.index.name = "variable"
    df_u.to_csv(OUT / "ols_unstandardized.csv")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in [export_ols, export_qr_coefficients, export_iqr_tests,
               export_qr_stability, export_imputation_winsorization,
               export_spatial_and_ablation, export_shap]:
        print(f"Exporting {fn.__name__.replace('export_', '')}...")
        fn()
    print(f"Done. CSVs written to {OUT}")


if __name__ == "__main__":
    main()
