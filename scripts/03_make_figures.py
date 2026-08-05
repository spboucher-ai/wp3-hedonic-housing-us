#!/usr/bin/env python3
# Author: Simon-Pierre Boucher — contact@spboucher.ai
#
"""Regenerate all 11 paper figures from the stored pickles.

Every figure is rebuilt from the data actually used in the paper
(analytical_sample.pkl, model_data.pkl, ml_results.pkl, qr_results.pkl,
shap_data.pkl) in a unified publication style. The script prints
verification statistics (Jarque-Bera, skewness, kurtosis, model R2, OLS
reference coefficients) so the regenerated content can be checked against
the numbers reported in the paper. The six original PNGs remain untouched
in the source archive (immo-wp3-spb-20260519/figures/).

Usage:
    python scripts/03_make_figures.py               # all 11 figures
    python scripts/03_make_figures.py --figs fig2 fig4
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wp3 import config, data
from wp3.plotting import (ACCENT, CMAP_DIV, CMAP_SEQ, NEUTRAL, PRIMARY,
                          PRIMARY_LIGHT, apply_paper_style, label, panel_title)

RNG = np.random.RandomState(config.RANDOM_STATE)

FILE_NAMES = {
    "fig1": "fig1_price_distribution.png",
    "fig2": "fig2_ols_diagnostics.png",
    "fig3": "fig3_quantile_coefficients.png",
    "fig4": "fig4_model_comparison.png",
    "fig5": "fig5_shap_summary.png",
    "fig6": "fig6_shap_importance.png",
    "fig7": "fig7_geographic_prices.png",
    "fig8": "fig8_regional_prices.png",
    "fig9": "fig9_marginal_effects.png",
    "fig10": "fig10_shap_dependence.png",
    "fig11": "fig11_shap_interactions.png",
}


def fig1_price_distribution(path):
    """Histograms of price (thousands) and log-price with median markers."""
    df = data.load_analytical_sample()
    price_k = df["price"].values / 1000.0

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    ax.hist(price_k[price_k <= 3000], bins=100, color=PRIMARY, alpha=0.85,
            edgecolor="white", linewidth=0.2)
    med = np.median(price_k)
    ax.axvline(med, color=ACCENT, linestyle="--", linewidth=1.8,
               label=f"Median: \\${med:,.0f}K")
    ax.set_xlabel("Listing Price (\\$ thousands)")
    ax.set_ylabel("Frequency")
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    panel_title(ax, "(a) Price Distribution")
    ax.legend()

    ax = axes[1]
    ax.hist(df["ln_price"], bins=100, color=PRIMARY, alpha=0.85,
            edgecolor="white", linewidth=0.2)
    med_ln = df["ln_price"].median()
    ax.axvline(med_ln, color=ACCENT, linestyle="--", linewidth=1.8,
               label=f"Median: {med_ln:.2f}")
    ax.set_xlabel("Log(Price)")
    ax.set_ylabel("Frequency")
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    panel_title(ax, "(b) Log-Price Distribution")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig2_ols_diagnostics(path):
    """OLS diagnostics: residuals vs fitted, Q-Q, distribution, scale-location."""
    md = data.load_model_data()
    resid = np.asarray(md["residuals"], dtype=float)
    fitted = np.asarray(md["y_clean"], dtype=float) - resid

    jb = st.jarque_bera(resid)[0]
    print(f"  [verify] skew={st.skew(resid):.2f} (paper: 0.18)  "
          f"kurtosis={st.kurtosis(resid, fisher=False):.2f} (paper: 5.55)  "
          f"JB={jb:,.0f} (paper: 217,078)")

    idx = RNG.choice(len(resid), size=100_000, replace=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.hexbin(fitted[idx], resid[idx], gridsize=60, cmap=CMAP_SEQ, mincnt=1)
    ax.axhline(0, color=ACCENT, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    panel_title(ax, "(a) Residuals vs. Fitted")

    ax = axes[0, 1]
    (osm, osr), (slope, intercept, _) = st.probplot(resid[idx], dist="norm")
    ax.plot(osm, osr, ".", color=PRIMARY, markersize=2.5, rasterized=True)
    ax.plot(osm, slope * osm + intercept, color=ACCENT, linestyle="--", linewidth=1.5)
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    panel_title(ax, "(b) Normal Q-Q Plot")

    ax = axes[1, 0]
    ax.hist(resid, bins=100, color=PRIMARY, alpha=0.85, density=True,
            edgecolor="white", linewidth=0.2)
    x = np.linspace(resid.min(), resid.max(), 400)
    ax.plot(x, st.norm.pdf(x, resid.mean(), resid.std()), color=ACCENT,
            linestyle="--", linewidth=1.6, label="Normal density")
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Density")
    panel_title(ax, "(c) Residual Distribution")
    ax.legend()

    ax = axes[1, 1]
    ax.hexbin(fitted[idx], np.sqrt(np.abs(resid[idx] / resid.std())),
              gridsize=60, cmap=CMAP_SEQ, mincnt=1)
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel(r"$\sqrt{|\mathrm{Standardized\ Residuals}|}$")
    panel_title(ax, "(d) Scale-Location")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig3_quantile_coefficients(path):
    """Quantile-regression coefficient paths with the OLS benchmark."""
    qr = data.load_result("qr_results.pkl")
    md = data.load_model_data()
    ols_params = md["ols_results"]["params"]
    taus = sorted(qr.keys())

    plot_vars = ["ln_sqft", "bathrooms", "age", "ln_lot",
                 "has_pool", "has_garage", "luxury_score", "tag_foreclosure"]
    print(f"  [verify] OLS ref ln_sqft={ols_params['ln_sqft']:.3f} "
          f"(original figure: 0.312)")

    fig, axes = plt.subplots(2, 4, figsize=(19, 9))
    for k, (ax, v) in enumerate(zip(axes.ravel(), plot_vars)):
        coefs = [qr[t]["params"][v] for t in taus]
        ax.plot(taus, coefs, color=PRIMARY, marker="o", markersize=6,
                linewidth=2, zorder=3)
        ax.axhline(ols_params[v], color=ACCENT, linestyle="--", linewidth=1.5,
                   label="OLS")
        ax.set_xticks(taus)
        ax.set_xlabel(r"Quantile $\tau$")
        if k % 4 == 0:
            ax.set_ylabel("Coefficient")
        panel_title(ax, f"({chr(97 + k)}) {label(v)}")
        if k == 0:
            ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig4_model_comparison(path):
    """Predicted vs actual log-prices on the random test set: OLS, XGB, LGBM."""
    from sklearn.metrics import r2_score

    ml = data.load_result("ml_results.pkl")
    y_test = np.asarray(ml["y_test"], dtype=float)
    panels = [
        ("(a) OLS", ml["y_pred_ols"]),
        ("(b) XGBoost", ml["y_pred_xgb"]),
        ("(c) LightGBM", ml["y_pred_lgb"]),
    ]

    idx = RNG.choice(len(y_test), size=50_000, replace=False)
    lims = (y_test.min() - 0.1, y_test.max() + 0.1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    for ax, (title, y_pred) in zip(axes, panels):
        y_pred = np.asarray(y_pred, dtype=float)
        r2 = r2_score(y_test, y_pred)
        print(f"  [verify] {title.split(') ')[1]}: R2={r2:.4f} "
              f"(paper: OLS 0.630 / XGBoost 0.833 / LightGBM 0.809)")
        ax.hexbin(y_test[idx], y_pred[idx], gridsize=70, cmap=CMAP_SEQ, mincnt=1)
        ax.plot(lims, lims, color=ACCENT, linestyle="--", linewidth=1.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Actual Log(Price)")
        panel_title(ax, title)
        ax.text(0.05, 0.92, f"$R^2 = {r2:.3f}$", transform=ax.transAxes,
                fontsize=13)
    axes[0].set_ylabel("Predicted Log(Price)")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _shap_frame():
    """SHAP values and features with human-readable column labels."""
    sd = data.load_shap_data()
    X = sd["X_shap"].copy()
    return sd, X


def fig5_shap_summary(path):
    """SHAP beeswarm summary plot for the XGBoost model (top 20 features)."""
    import shap

    sd, X = _shap_frame()
    X_lab = X.rename(columns={c: label(c) for c in X.columns})
    plt.figure(figsize=(12, 10))
    shap.summary_plot(sd["shap_values_xgb"], X_lab, max_display=20, show=False,
                      plot_size=None)
    ax = plt.gca()
    ax.set_xlabel("SHAP Value (impact on predicted log-price)")
    ax.grid(axis="y", visible=False)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def fig6_shap_importance(path):
    """Mean |SHAP| feature-importance bar chart (top 20)."""
    sd, _ = _shap_frame()
    mean_shap = pd.Series(sd["mean_shap"]).sort_values(ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.barh([label(v) for v in mean_shap.index], mean_shap.values,
            color=PRIMARY, alpha=0.9, height=0.72)
    ax.set_xlabel(r"Mean $|$SHAP Value$|$")
    ax.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.45, color="#999999")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig7_geographic_prices(path):
    """Map of log listing prices (50,000-listing random subsample)."""
    df = data.load_analytical_sample()
    idx = RNG.choice(len(df), size=50_000, replace=False)
    sub = df.iloc[idx]

    fig, ax = plt.subplots(figsize=(15, 9))
    sc = ax.scatter(sub["longitude"], sub["latitude"], c=sub["ln_price"],
                    s=3.5, cmap="viridis", alpha=0.75, linewidths=0,
                    rasterized=True)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Log(Price)")
    cbar.outline.set_visible(False)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(False)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig8_regional_prices(path):
    """Distribution of log listing prices by Census region."""
    df = data.load_analytical_sample()
    order = ["Northeast", "Midwest", "South", "West"]
    groups = [df.loc[df["region"] == r, "ln_price"].values for r in order]
    labels_n = [f"{r}\n(N = {len(g):,})" for r, g in zip(order, groups)]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.boxplot(groups, tick_labels=labels_n, showfliers=False, patch_artist=True,
               widths=0.55,
               medianprops=dict(color=ACCENT, linewidth=1.8),
               boxprops=dict(facecolor=PRIMARY, alpha=0.75, edgecolor="#333333"),
               whiskerprops=dict(color="#333333", linewidth=1.0),
               capprops=dict(color="#333333", linewidth=1.0))
    ax.set_ylabel("Log(Price)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig9_marginal_effects(path):
    """Unconditional bivariate relationships: scatter plus binned means."""
    df = data.load_analytical_sample()
    specs = [
        ("living_area_sqft", "Living Area (sqft)", (0, 8000)),
        ("age", "Property Age (years)", (0, 150)),
        ("lot_size_sqft", "Lot Size (sqft)", (0, 45000)),
        ("bedrooms", "Bedrooms", None),
        ("bathrooms", "Bathrooms", (0, 8)),
        ("avg_school_rating", "Avg. School Rating", None),
    ]
    idx = RNG.choice(len(df), size=30_000, replace=False)
    sub = df.iloc[idx]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for k, (ax, (col, xlabel, xlim)) in enumerate(zip(axes.ravel(), specs)):
        x, y = sub[col].values, sub["ln_price"].values
        if xlim is not None:
            keep = (x >= xlim[0]) & (x <= xlim[1])
            x, y = x[keep], y[keep]
        ax.scatter(x, y, s=2.5, color=PRIMARY_LIGHT, alpha=0.2, linewidths=0,
                   rasterized=True)

        # Binned means computed on the full sample
        xf, yf = df[col].values, df["ln_price"].values
        if xlim is not None:
            keepf = (xf >= xlim[0]) & (xf <= xlim[1])
            xf, yf = xf[keepf], yf[keepf]
        bins = np.linspace(np.nanmin(xf), np.nanmax(xf), 25)
        which = np.digitize(xf, bins)
        centers = [xf[which == b].mean() for b in range(1, len(bins))
                   if (which == b).sum() > 50]
        means = [yf[which == b].mean() for b in range(1, len(bins))
                 if (which == b).sum() > 50]
        ax.plot(centers, means, color=ACCENT, linewidth=2.2, marker="o",
                markersize=4.5, label="Binned mean", zorder=3)

        ax.set_xlabel(xlabel)
        if k % 3 == 0:
            ax.set_ylabel("Log(Price)")
        if col in ("living_area_sqft", "lot_size_sqft"):
            ax.xaxis.set_major_formatter(
                plt.matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
        panel_title(ax, f"({chr(97 + k)}) {xlabel}")
        if k == 0:
            ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig10_shap_dependence(path):
    """SHAP dependence plots for the eight features shown in the paper."""
    sd, X = _shap_frame()
    shap_vals = sd["shap_values_xgb"]
    feats = ["ln_sqft", "bathrooms", "ln_lot", "avg_school_rating",
             "property_tax_rate", "age", "walk_score", "luxury_score"]

    fig, axes = plt.subplots(2, 4, figsize=(20, 9.5))
    for k, (ax, feat) in enumerate(zip(axes.ravel(), feats)):
        j = list(X.columns).index(feat)
        xv = X[feat].values
        ax.scatter(xv, shap_vals[:, j], c=xv, cmap=CMAP_DIV, s=4, alpha=0.6,
                   linewidths=0, rasterized=True)
        ax.axhline(0, color=NEUTRAL, linestyle="--", linewidth=1.1)
        ax.set_xlabel(label(feat))
        if k % 4 == 0:
            ax.set_ylabel("SHAP Value")
        panel_title(ax, f"({chr(97 + k)}) {label(feat)}")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig11_shap_interactions(path):
    """SHAP values of the two interaction features highlighted in the paper."""
    sd, X = _shap_frame()
    shap_vals = sd["shap_values_xgb"]
    panels = [
        ("waterfront_x_sqft", "(a) Waterfront $\\times$ Living Area Interaction"),
        ("pool_x_south", "(b) Pool $\\times$ South Region Interaction"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    for ax, (feat, title) in zip(axes, panels):
        j = list(X.columns).index(feat)
        ax.scatter(X[feat].values, shap_vals[:, j], s=6, color=PRIMARY,
                   alpha=0.35, linewidths=0, rasterized=True)
        ax.axhline(0, color=NEUTRAL, linestyle="--", linewidth=1.1)
        ax.set_xlabel(label(feat))
        ax.set_ylabel("SHAP Value")
        panel_title(ax, title)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


BUILDERS = {
    "fig1": fig1_price_distribution,
    "fig2": fig2_ols_diagnostics,
    "fig3": fig3_quantile_coefficients,
    "fig4": fig4_model_comparison,
    "fig5": fig5_shap_summary,
    "fig6": fig6_shap_importance,
    "fig7": fig7_geographic_prices,
    "fig8": fig8_regional_prices,
    "fig9": fig9_marginal_effects,
    "fig10": fig10_shap_dependence,
    "fig11": fig11_shap_interactions,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figs", nargs="+", choices=sorted(BUILDERS),
                        help="build only these figures")
    args = parser.parse_args()

    apply_paper_style()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    names = args.figs if args.figs else list(BUILDERS)
    for name in names:
        path = config.FIGURES_DIR / FILE_NAMES[name]
        print(f"Building {name} -> {path.relative_to(config.PROJECT_ROOT)}")
        BUILDERS[name](path)

    print("Done.")


if __name__ == "__main__":
    main()
