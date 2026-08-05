<div align="center">

# 🏠 Hedonic Housing Price Models for the United States

### A Multi-Method Comparison of Parametric, Quantile, and Machine Learning Approaches

**UQO Working Paper No. 3**

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LaTeX](https://img.shields.io/badge/LaTeX-pdflatex%20%2B%20BibTeX-008080?logo=latex&logoColor=white)](https://www.latex-project.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-EB5E28)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-9ACD32)](https://lightgbm.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.48.0-blueviolet)](https://shap.readthedocs.io/)
[![statsmodels](https://img.shields.io/badge/statsmodels-0.14.6-4051B5)](https://www.statsmodels.org/)
[![DuckDB](https://img.shields.io/badge/DuckDB-1.5.2-FFF000?logo=duckdb&logoColor=black)](https://duckdb.org/)

[![Paper](https://img.shields.io/badge/Paper-60%20pages%20PDF-B31B1B?logo=adobeacrobatreader&logoColor=white)](paper/main.pdf)
[![Sample](https://img.shields.io/badge/Sample-788%2C842%20listings-2B5C8A)](#-data)
[![Coverage](https://img.shields.io/badge/Coverage-50%20states%20%2B%20DC-2B5C8A)](#-data)
[![Figures](https://img.shields.io/badge/Figures-11%20publication--ready-A63E38)](#-figures)
[![References](https://img.shields.io/badge/References-70%20verified%20(OpenAlex%2FDOI)-2E8B57)](paper/references.bib)
[![Verified](https://img.shields.io/badge/Results-verified%20against%20stored%20artifacts-2E8B57)](#-verification--integrity)
[![Status](https://img.shields.io/badge/Status-Working%20Paper%20v1.1-orange)](#-citation)

**Author : [Simon-Pierre Boucher](mailto:contact@spboucher.ai)**
Département des sciences administratives — Université du Québec en Outaouais (UQO)
📧 **contact@spboucher.ai**

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Findings](#-key-findings)
- [Repository Structure](#-repository-structure)
- [Data](#-data)
- [Methodology at a Glance](#-methodology-at-a-glance)
- [The Analysis Pipeline](#-the-analysis-pipeline)
- [Installation & Setup](#-installation--setup)
- [Reproducing the Results](#-reproducing-the-results)
- [Building the Paper](#-building-the-paper)
- [Figures](#-figures)
- [Results Summary](#-results-summary)
- [Verification & Integrity](#-verification--integrity)
- [Project History & Provenance](#-project-history--provenance)
- [Citation](#-citation)
- [Contact & License](#-contact--license)

---

## 🔎 Overview

This repository contains the complete research compendium — analysis code, LaTeX
source, publication-ready figures, and machine-readable result tables — for
**UQO Working Paper No. 3**, which compares three frameworks for hedonic housing
valuation on a single large dataset of **788,842 active Zillow listings** spanning
all 50 U.S. states and the District of Columbia:

| Framework | Question it answers | Headline result |
|---|---|---|
| **Semi-log OLS** (62 regressors, HC3 SE) | What is the *average* conditional association between an attribute and log listing price? | $R^2 = 0.634$; price-to-area elasticity 0.63 |
| **Quantile regression** ($\tau \in \{0.10, 0.25, 0.50, 0.75, 0.90\}$) | How do attribute gradients *vary across the price distribution*? | Coefficient equality rejected for **11 of 13** variables (inter-quantile Wald tests) |
| **Gradient boosting + SHAP** (XGBoost, LightGBM) | How well can prices be *predicted*, and which features drive predictions? | $R^2 = 0.833$ (random split) vs. **0.425–0.547** (geographic holdout) |

The paper's core methodological contribution is a systematic study of
**spatial leakage** in hedonic model evaluation: random train/test splits let
geographically proximate listings appear on both sides of the split, inflating
performance metrics. Three complementary designs quantify the damage — a 10-state
geographic holdout, a 6-stage feature-ablation cascade, and a latitude/longitude
augmentation experiment.

> ⚠️ **Interpretation caveat.** All estimates are *listing-price capitalization
> gradients* — conditional associations between attributes and *asking* prices.
> They are not causal willingness-to-pay parameters and not transaction-price
> implicit prices.

---

## 🏆 Key Findings

1. **Geography does the heavy lifting in OLS.** Moving from 4 Census-region
   dummies to state fixed effects to 886 ZIP3 fixed effects raises out-of-sample
   $R^2$ from 0.630 → 0.678 → 0.725 — a 9.5 pp gain from spatial granularity alone.

2. **Strong, stable residual spatial autocorrelation.** Moran's $I$ on OLS
   residuals averages **0.2745** (SD 0.0076) across three independent 5,000-listing
   subsamples (row-standardized KNN weights, $k = 8$; all $p < 0.001$).

3. **Mean effects mask distributional heterogeneity.** The garage gradient is
   **10× larger** at $\tau = 0.10$ than at $\tau = 0.90$ ($z = 28.9$); the pool
   premium only emerges above the median; the age penalty is concentrated in
   lower-priced homes.

4. **Random validation flatters machine learning.** XGBoost reaches $R^2 = 0.833$
   under a random 80/20 split but only **0.425–0.547** when 10 entire states
   (438,315 listings) are held out.

5. **Geographic features can *hurt* generalization.** Removing all geographic
   features *improves* geographic-holdout $R^2$ from 0.425 to **0.519**, while
   adding raw lat/lon coordinates boosts random $R^2$ (+3.7 pp) but *degrades*
   geographic holdout. Region dummies memorize training-set price levels.

6. **SHAP rankings are model-stable but not structural.** Spearman rank
   correlations of mean $|\phi|$ across XGBoost / LightGBM / Random Forest range
   from **0.89 to 0.99**, with the same six features on top — yet these remain
   predictive decompositions, not implicit prices.

7. **Imputation matters.** The lot-size gradient **triples** (0.018 → 0.059) when
   state-median-imputed lot sizes are dropped — a warning for hedonic work on
   scraped listing data.

---

## 🗂 Repository Structure

```
wp3_uqo/
├── README.md                      ← you are here
├── AUDIT.md                       ← forensic audit of the original project archive
├── CHANGES.md                     ← every move/refactor/fix made during restructuring
├── requirements.txt               ← pinned Python dependencies (validated 2026-08-05)
│
├── data/                          ⚠ NOT in the Git repo (3.2 GB — see "Data" below)
│   ├── raw/
│   │   └── us_housing.duckdb      # raw Zillow extract (839,313 × 116 + 4 aux tables)
│   └── processed/
│       ├── analytical_sample.pkl  # final sample, 788,842 × 68 engineered columns
│       ├── model_data.pkl         # standardized X (788,842 × 63), y, OLS fit, residuals
│       └── shap_data.pkl          # TreeSHAP values (10,000 × 62) + explanation sample
│
├── src/wp3/                       # shared library code
│   ├── config.py                  # paths (relative, WP3_ROOT-overridable), constants,
│   │                              #   feature-block definitions, holdout states, seeds
│   ├── data.py                    # loaders + 62-regressor design-matrix builder
│   ├── models.py                  # XGBoost training helper (dual validation schemes)
│   └── plotting.py                # journal figure style + human-readable variable labels
│
├── scripts/                       # numbered pipeline entry points
│   ├── 01_spatial_robustness.py   # ZIP3 FE OLS · Moran's I ·  XGB ± geography · ablation
│   ├── 02_qr_imputation.py        # QR stability · inter-quantile Wald · imputation ·
│   │                              #   winsorization sensitivity
│   ├── 03_make_figures.py         # regenerates all 11 figures + verification stats
│   └── 04_export_tables.py        # exports paper tables → results/tables/*.csv
│
├── figures/                       # all 11 paper figures (PNG, 300 dpi, unified style)
│
├── results/
│   ├── *.pkl                      ⚠ NOT in the Git repo (up to 437 MB)
│   └── tables/                    # 13 CSVs mirroring the paper's tables (in repo)
│
└── paper/                         # LaTeX source
    ├── main.tex                   # preamble + metadata + \input skeleton
    ├── references.bib             # 37 BibTeX entries (natbib + apalike)
    ├── sections/                  # one file per section (IMRaD + robustness)
    │   ├── titlepage.tex          ├── introduction.tex   ├── literature.tex
    │   ├── data.tex               ├── methodology.tex    ├── results.tex
    │   ├── robustness.tex         ├── discussion.tex     ├── limitations.tex
    │   └── conclusion.tex
    ├── appendix/appendix.tex      # extra figures, full OLS table, reproducibility checklist
    ├── Makefile                   # latexmk targets (all / clean / distclean)
    ├── uq_logo.jpg
    └── main.pdf                   # compiled paper — 52 pages, zero warnings
```

---

## 💾 Data

### What the data are

- **Source:** Zillow active for-sale listings, 2025–2026 snapshot.
- **Raw:** 839,313 residential properties × 116 variables, stored in a DuckDB
  database (`properties`, `price_history`, `schools`, `tax_history`, `meta_columns`).
- **Analytical sample:** 788,842 listings after five sequential filters
  (valid price \$10K–\$10M, living area 200–20,000 sqft, 1–10 beds/baths,
  valid coordinates, US states + DC only — 6.0 % attrition).
- **68 engineered columns:** log transforms, age & age², ratios, amenity dummies,
  neighborhood scores, market-status flags, 6 interaction terms, and simplified
  categoricals (roof / construction / foundation / region).

### Why the data are not in this Git repository

GitHub rejects files above 100 MB. The DuckDB file is **1.6 GB** and the processed
pickles reach **867 MB**, so `data/` and `results/*.pkl` are excluded via
`.gitignore`. The canonical copies live in the local research archive:

| Artifact | Size | Location |
|---|---|---|
| `us_housing.duckdb` | 1.6 GB | `data/raw/` (local) |
| `analytical_sample.pkl` | 457 MB | `data/processed/` (local) |
| `model_data.pkl` | 867 MB | `data/processed/` (local) |
| `ml_results.pkl` | 437 MB | `results/` (local) |
| `shap_data.pkl`, `qr_results.pkl`, `extended_results.pkl`, `v3_*.pkl` | < 10 MB each | `results/` (local; small ones could be added on request) |

📬 **Data access:** available on request (subject to Zillow Terms of Service) —
**contact@spboucher.ai**.

---

## 🧪 Methodology at a Glance

| Component | Specification |
|---|---|
| Dependent variable | $\ln(\text{listing price})$ |
| OLS | 62 regressors, standardized + unstandardized variants, **HC3** robust SE |
| Fixed effects | Census region (baseline) → state (50) → ZIP3 (886) |
| Quantile regression | statsmodels `QuantReg`, $\tau \in \{0.10,\ldots,0.90\}$, 150,000-obs subsample, 10-seed stability check |
| Inter-quantile tests | Wald $z$-tests on $\hat\beta(0.10)-\hat\beta(0.90)$ |
| XGBoost | 1,000 trees, depth 8, lr 0.05, subsample/colsample 0.8, early stopping 50 |
| LightGBM / RF / Ridge / Lasso / Elastic Net | benchmark suite |
| Validation | random 80/20 (seed 42) **and** 10-state geographic holdout (CA NY TX FL OH CO NC WA IL GA — 438,315 test obs) |
| Ablation | 6 nested feature sets: structural → +lot → +amenities → +neighborhood → +market → full |
| Interpretation | TreeSHAP on 10,000 test obs; cross-model Spearman stability (XGB/LGB/RF) |
| Spatial diagnostics | Moran's $I$, row-standardized KNN ($k=8$), 3 × 5,000-obs subsamples, 999 permutations |
| Sensitivity | state-median-imputation drop-out, lot-size winsorization (p99.5), Bike Score cap |

---

## 🔁 The Analysis Pipeline

```
us_housing.duckdb  (raw, 839,313 listings)
        │
        │  [sample construction & feature engineering — original script lost;
        │   the stored analytical_sample.pkl is the canonical artifact]
        ▼
analytical_sample.pkl  (788,842 × 68)
        │
        ├─ [estimation — original scripts lost; stored artifacts are canonical]
        │      ├── model_data.pkl        (standardized X, OLS fit, residuals)
        │      ├── qr_results.pkl        (QR params at 5 quantiles)
        │      ├── ml_results.pkl        (fitted XGB/LGBM + test predictions)
        │      ├── shap_data.pkl         (TreeSHAP values, 10,000 × 62)
        │      ├── extended_results.pkl  (unstandardized OLS, geo holdout, RF)
        │      └── v3_shap_stability.pkl (XGB/LGB/RF SHAP rank correlations)
        │
        ├─ scripts/01_spatial_robustness.py ──► results/v3_spatial_results.pkl
        ├─ scripts/02_qr_imputation.py ───────► results/v3_qr_imputation_results.pkl
        ├─ scripts/03_make_figures.py ────────► figures/fig1…fig11 (+ verification)
        └─ scripts/04_export_tables.py ───────► results/tables/*.csv
                                                       │
                                                       ▼
                                          paper/main.tex ──► main.pdf (52 p.)
```

The bracketed upstream stages were produced before this restructuring by scripts
that no longer exist (see [Project History](#-project-history--provenance)); their
outputs are preserved and every downstream number is verified against them.

---

## ⚙️ Installation & Setup

**Prerequisites:** Python ≥ 3.11, a TeX Live distribution (with `newtx`, `natbib`,
`booktabs`, `threeparttable`), and ~4 GB of free disk for the data artifacts.

```bash
git clone https://github.com/spboucher-ai/wp3-hedonic-housing-us.git
cd wp3-hedonic-housing-us

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then place the data artifacts (request them at **contact@spboucher.ai**) into
`data/raw/`, `data/processed/`, and `results/` as shown in
[Repository Structure](#-repository-structure).

All paths resolve relative to the repository root. To point the code at another
location, set the environment variable:

```bash
export WP3_ROOT=/path/to/artifacts
```

---

## ▶️ Reproducing the Results

| Step | Command | Runtime* | Needs |
|---|---|---|---|
| Spatial & robustness suite | `python scripts/01_spatial_robustness.py` | ~1–2 h (trains 16 XGBoost models) | `analytical_sample.pkl` |
| QR & sensitivity suite | `python scripts/02_qr_imputation.py` | ~1–3 h (12 quantile regressions on 150k obs) | `model_data.pkl`, DuckDB |
| **All 11 figures** | `python scripts/03_make_figures.py` | ~5 min | processed pickles |
| Single figures | `python scripts/03_make_figures.py --figs fig2 fig4` | seconds–minutes | idem |
| **CSV table exports** | `python scripts/04_export_tables.py` | ~2 min | result pickles |

\* Apple Silicon, 32 GB RAM; scripts use all cores (`n_jobs=-1`).

Every script prints **verification statistics** comparing recomputed values with
the numbers reported in the paper (see [Verification](#-verification--integrity)).

---

## 📄 Building the Paper

```bash
cd paper
latexmk          # pdflatex ×N + bibtex, via Makefile/latexmkrc defaults
# or: make       # same thing
# or: make clean / make distclean
```

- Output: `paper/main.pdf` — **52 pages**, compiles with **zero errors, zero
  undefined references/citations**.
- Figures are pulled from `../figures/` via `\graphicspath`.
- Bibliography: `paper/references.bib` (37 entries), `natbib` + `apalike`.
- Metadata (title, abstract, keywords R31/C21/C45/C52, version) is centralized in
  `main.tex` `\WP*` macros.

---

## 🖼 Figures

All figures are regenerated from the stored artifacts by `scripts/03_make_figures.py`
in a unified publication style (serif typography matched to the paper's `newtx`
text font, muted colorblind-safe palette, no chart junk, 300 dpi).

| # | File | Content | Built from |
|---|---|---|---|
| 1 | `fig1_price_distribution.png` | Price & log-price histograms with medians | `analytical_sample.pkl` |
| 2 | `fig2_ols_diagnostics.png` | Residuals-vs-fitted, Q-Q, density, scale-location | `model_data.pkl` |
| 3 | `fig3_quantile_coefficients.png` | QR coefficient paths vs. OLS benchmark (8 panels) | `qr_results.pkl` + `model_data.pkl` |
| 4 | `fig4_model_comparison.png` | Predicted vs. actual: OLS / XGBoost / LightGBM | `ml_results.pkl` |
| 5 | `fig5_shap_summary.png` | SHAP beeswarm, top 20 features | `shap_data.pkl` |
| 6 | `fig6_shap_importance.png` | Mean \|SHAP\| bar chart, top 20 | `shap_data.pkl` |
| 7 | `fig7_geographic_prices.png` | 50,000-listing national price map (lat/lon, viridis) | `analytical_sample.pkl` |
| 8 | `fig8_regional_prices.png` | Log-price boxplots by Census region (with N) | `analytical_sample.pkl` |
| 9 | `fig9_marginal_effects.png` | Bivariate scatters + binned means (6 attributes) | `analytical_sample.pkl` |
| 10 | `fig10_shap_dependence.png` | SHAP dependence, 8 key features | `shap_data.pkl` |
| 11 | `fig11_shap_interactions.png` | SHAP for interaction terms (waterfront×sqft, pool×South) | `shap_data.pkl` |

---

## 📊 Results Summary

**Predictive performance (log-price scale):**

| Model | Random $R^2$ | Geo-holdout $R^2$ | RMSE (random) |
|---|---:|---:|---:|
| OLS (62 regressors) | 0.630 | < 0 | 0.491 |
| OLS + State FE | 0.678 | — | 0.459 |
| OLS + ZIP3 FE | 0.725 | — | 0.423 |
| Random Forest | 0.784 | 0.542 | 0.375 |
| LightGBM | 0.809 | — | 0.353 |
| **XGBoost** | **0.833** | **0.547** | **0.330** |
| XGBoost, no geographic features | 0.830 | **0.519** | — |
| XGBoost + lat/lon | 0.870 | 0.464 | — |
| XGBoost (ablation variant, full 62) | 0.833 | 0.425 | — |

**Ablation cascade (XGBoost):** structural 0.498 → +lot 0.546 → +amenities 0.638 →
+neighborhood **0.814 (+17.6 pp)** → +market 0.824 → full 0.833 (random $R^2$).

**Machine-readable versions** of these and all other paper tables are in
[`results/tables/`](results/tables/) (13 CSV files: OLS coefficients, QR paths,
inter-quantile Wald tests, stability CVs, Moran's I, imputation/winsorization
sensitivity, SHAP importance & cross-model correlations, ZIP3 FE, ablation).

---

## ✅ Verification & Integrity

The restructuring included a systematic audit reconciling the manuscript against
the stored result artifacts. Highlights:

- **Exact matches** — OLS residual diagnostics (skewness 0.18, kurtosis 5.55,
  Jarque–Bera 217,078), test-set $R^2$ (OLS 0.6301, XGBoost 0.8330,
  LightGBM 0.8088), Moran's $I$ subsample values and mean/SD, QR coefficients at
  all five quantiles, and the OLS reference line (0.312) in Figure 3.
- **End-to-end reproduction** — re-running the refactored Moran's $I$ pipeline
  reproduces the stored values to 4 decimal places (0.2742 / 0.2839 / 0.2652).
- **Manuscript corrections** — six internal inconsistencies were found and fixed
  (wrong KNN $k$ in the text, mislabeled inter-quantile sign convention,
  winsorization description not matching the actual procedure, a duplicated table
  row, mixed in/out-of-sample $R^2$ in the summary matrix). Every fix aligns the
  text to the stored results and is itemized in [`CHANGES.md`](CHANGES.md) §4.
  **No scientific result was altered.**
- **Completed placeholders** — 13 `[TODO]` table cells in the original manuscript
  were filled with the exact values from the result pickles.

See [`AUDIT.md`](AUDIT.md) for the full audit and [`CHANGES.md`](CHANGES.md) for
the change log.

---

## 🕰 Project History & Provenance

- The project was completed in **May 2026** in a flat working folder
  (`immo-wp3-spb-20260519`), then restructured into this compendium on
  **2026-08-05**. The original archive is preserved untouched.
- The scripts that built the analytical sample and estimated the primary models
  (OLS/QR/ML/SHAP) were **lost** prior to restructuring; only two robustness
  scripts survived and were refactored into `scripts/01` and `scripts/02`. The
  stored pickles are the canonical record of those stages and every downstream
  number is validated against them.
- Five paper figures had been deleted from the archive; they were rebuilt from the
  stored data, and subsequently **all 11 figures** were regenerated in a unified
  journal style.
- Three earlier monolithic drafts (`paper_hedonic_pricing{,_v2,_v3}.tex`) precede
  the sectioned `wp3/` source that became `paper/`.

---

## 📖 Citation

If you use this code, the figures, or the results, please cite:

```bibtex
@techreport{boucher2026hedonic,
  author      = {Boucher, Simon-Pierre},
  title       = {Hedonic Housing Price Models for the United States:
                 A Multi-Method Comparison of Parametric, Quantile,
                 and Machine Learning Approaches},
  institution = {Universit\'e du Qu\'ebec en Outaouais,
                 D\'epartement des sciences administratives},
  type        = {Working Paper},
  number      = {3},
  year        = {2026},
  month       = {May}
}
```

---

## 📬 Contact & License

**Simon-Pierre Boucher**
Département des sciences administratives
Université du Québec en Outaouais — Gatineau (Québec), Canada
📧 **contact@spboucher.ai** · 🎓 simon-pierre.boucher@uqo.ca

© 2026 Simon-Pierre Boucher. Analysis code is shared for **replication and review**
purposes. The underlying Zillow data are subject to Zillow's Terms of Service and
are **not redistributed** in this repository; the paper (`paper/main.pdf`) is a
working paper — please cite rather than redistribute.

<div align="center">
<sub>Built with Python · XGBoost · LightGBM · statsmodels · SHAP · DuckDB · LaTeX — reproducible from stored artifacts end to end.</sub>
</div>
