# AUDIT — UQO Working Paper 3 (Hedonic Housing Prices, US)

**Source auditée** : `~/Desktop/UQO/UQO_WP/immo-wp3-spb-20260519/` (3,2 Go, intacte — non modifiée)
**Nouvelle structure** : `~/Desktop/wp3_uqo/` (ce dossier)
**Date de l'audit** : 2026-08-05

---

## 1. Vue d'ensemble

Projet de recherche terminé : comparaison OLS / régression quantile / ML (XGBoost, LightGBM)
pour la valorisation hédonique de 788 842 annonces Zillow (50 États + DC).
Le papier final est le dossier `wp3/` (main.tex + sections), version « UQO Working Paper No. 3 », mai 2026.

## 2. Inventaire des fichiers source

### 2.1 Code Python (2 scripts seulement — voir §5 : scripts perdus)

| Script | Rôle | Produit |
|---|---|---|
| `run_v3_spatial.py` (19,6 Ko) | ZIP3 fixed-effects OLS ; Moran's I (3 sous-échantillons) ; XGBoost avec/sans géographie ; ablation (6 jeux de variables) | `v3_spatial_results.pkl` |
| `v3_qr_imputation_analysis.py` (14,4 Ko) | Stabilité QR (10 sous-échantillons τ=0.50) ; test inter-quantile τ=0.10 vs 0.90 ; sensibilité imputation (via duckdb) ; sensibilité winsorisation | `v3_qr_imputation_results.pkl` |

Les deux scripts ont un chemin codé en dur `BASE = '/Users/simon-pierreboucher/Desktop/immo-wp3-spb-20260519'`
qui **n'existe plus** (le projet a été déplacé dans `UQO/UQO_WP/`) — ils ne tournent plus tels quels.

### 2.2 Données

| Fichier | Taille | Contenu | Classement |
|---|---|---|---|
| `us_housing.duckdb` | 1,6 Go | Données brutes : `properties` (839 313 × 116), `price_history`, `schools`, `tax_history`, `meta_columns` | **data/raw** |
| `analytical_sample.pkl` | 457 Mo | Échantillon analytique final : DataFrame 788 842 × 68 (variables engineerées : ln_price, ln_sqft, interactions, dummies catégorielles, etc.) | **data/processed** |
| `model_data.pkl` | 867 Mo | Données prêtes pour modèles : `X_const` (788 842 × 63, standardisé + const), `y_clean`, `df_clean`, `feature_names`, `means`/`stds`, résultats OLS (`ols_results`), `residuals` | **data/processed** |
| `shap_data.pkl` | 7,5 Mo | Valeurs SHAP XGBoost (10 000 × 62), `X_shap`, `expected_value` | **data/processed** |

### 2.3 Résultats (pickles)

| Fichier | Contenu |
|---|---|
| `qr_results.pkl` | Régression quantile : params / p-values / pseudo-R² aux τ ∈ {0.10, 0.25, 0.50, 0.75, 0.90} |
| `ml_results.pkl` (437 Mo) | Modèles ML entraînés (XGBRegressor, LGBMRegressor), prédictions test (157 769), splits train/test (631 073 / 157 769), métriques r2/rmse/mae, importances |
| `extended_results.pkl` | OLS non standardisé (params/bse/p), holdout géographique (OLS/XGB/RF), Moran's I initial, stabilité QR (v2) |
| `v3_spatial_results.pkl` | Sorties de `run_v3_spatial.py` (ZIP3 FE, Moran robustesse, XGB ±géo, ablation) |
| `v3_qr_imputation_results.pkl` | Sorties de `v3_qr_imputation_analysis.py` |
| `v3_shap_stability.pkl` | Stabilité SHAP inter-modèles (XGB/LGB/RF), rangs, ρ de Spearman |

### 2.4 Figures

`figures/` (racine) et `wp3/figures/` sont **identiques octet pour octet** (duplication).
Présentes (6) : `fig1_price_distribution`, `fig3_quantile_coefficients`, `fig5_shap_summary`,
`fig6_shap_importance`, `fig10_shap_dependence`, `fig11_shap_interactions`.

**⚠ Manquantes (5)** — référencées par le papier mais absentes des deux dossiers
(supprimées vers le 9 juin 2026 d'après les timestamps des dossiers) :

| Figure | Légende dans le papier | Régénérable depuis |
|---|---|---|
| `fig2_ols_diagnostics.png` | Diagnostics OLS (résidus vs fitted, Q-Q, distribution, scale-location) | `model_data.pkl` (residuals + fitted) |
| `fig4_model_comparison.png` | Prédit vs observé : OLS / XGBoost / LightGBM | `ml_results.pkl` (y_test, y_pred_*) |
| `fig7_geographic_prices.png` | Carte lat/lon des log-prix (sous-échantillon 50 000) | `analytical_sample.pkl` |
| `fig8_regional_prices.png` | Distribution des prix par région Census | `analytical_sample.pkl` |
| `fig9_marginal_effects.png` | Relations bivariées brutes (scatter + moyennes binnées) | `analytical_sample.pkl` |

### 2.5 LaTeX — 4 versions successives

| Version | Fichiers | Statut |
|---|---|---|
| v1 | `paper_hedonic_pricing.tex` (71 Ko, monolithique) | Obsolète |
| v2 | `paper_hedonic_pricing_v2.tex` (82 Ko) | Obsolète |
| v3 | `paper_hedonic_pricing_v3.tex` (115 Ko) | Obsolète — contenu repris dans wp3/ |
| **wp3/** | `main.tex` + `sections/` (9 sections + titlepage + references) + `appendix/` + `Makefile` | **Version finale** |

Particularités de `wp3/` :
- `main_web.tex` = copie de `main.tex` + hack `\includegraphics` qui remplace toute figure
  manquante par une boîte « Figure indisponible ». Le PDF existant (`main_web.pdf`, 50 pages,
  13 juin) contient donc des **placeholders** à la place des 5 figures manquantes.
- Il n'existe **pas** de `main.pdf` compilé (la compilation de `main.tex` échouerait sur les figures manquantes).
- Bibliographie : `thebibliography` manuel dans `sections/references.tex` (43 entrées) — pas de `.bib`.
- `wp3/tables/` est vide (tables codées en dur dans les .tex).
- Fichiers auxiliaires LaTeX (aux/log/out/toc/fls/fdb_latexmk/synctex) présents à la racine et dans wp3/.

## 3. Chaîne de production (reconstituée)

```
us_housing.duckdb (raw)
   └─(script PERDU : filtrage + feature engineering)→ analytical_sample.pkl (788 842 × 68)
        ├─(script PERDU : standardisation + OLS)→ model_data.pkl
        ├─(script PERDU : QR aux 5 quantiles)→ qr_results.pkl
        ├─(script PERDU : XGB/LGB + split 80/20)→ ml_results.pkl
        ├─(script PERDU : SHAP sur 10 000 obs)→ shap_data.pkl
        ├─(script PERDU : holdout géo, RF, Moran v2)→ extended_results.pkl
        ├─(script PERDU : stabilité SHAP XGB/LGB/RF)→ v3_shap_stability.pkl
        ├─ run_v3_spatial.py → v3_spatial_results.pkl
        └─ v3_qr_imputation_analysis.py → v3_qr_imputation_results.pkl
   figures fig1–fig11 : script(s) de tracé PERDU(S)
```

## 4. Fichiers morts / dupliqués / auxiliaires

- Doublons : `figures/` ≡ `wp3/figures/`.
- Versions obsolètes : `paper_hedonic_pricing{,_v2,_v3}.{tex,aux,log,out,toc}`.
- Auxiliaires LaTeX régénérables : `*.aux, *.log, *.out, *.toc, *.fls, *.fdb_latexmk, *.synctex.gz`.
- `main_web.tex` : contournement temporaire, à retirer une fois les figures restaurées.
- `.DS_Store` divers.

## 5. ⚠ Problèmes à signaler (à vérifier par l'auteur)

1. **Scripts générateurs perdus** : les scripts de construction de l'échantillon, d'estimation
   (OLS, QR, ML, SHAP) et de tracé des figures n'existent plus — seuls 2 scripts « v3 » survivent.
   Les nouveaux scripts dans `scripts/` sont des **reconstructions** documentées, écrites à partir
   des pickles, du papier et des 2 scripts survivants ; ils sont vérifiés contre les résultats
   stockés quand c'est faisable (voir CHANGES.md).
2. **5 figures manquantes** (fig2, fig4, fig7, fig8, fig9) : régénérées depuis les pickles.
   Le **contenu** (données) est identique aux données originales, mais le **style** (couleurs,
   mise en page) peut différer des PNG perdus — à valider visuellement.
3. **Les 11 figures ont finalement toutes été régénérées** dans un style « journal » homogène
   (demande utilisateur en cours de session) ; le contenu de chacune est vérifié contre les
   pickles et les chiffres du papier (voir CHANGES.md §3). Les 6 PNG originaux restent
   disponibles dans l'archive source.
4. Chemins codés en dur dans les 2 scripts survivants (corrigés dans les versions refactorées).
5. Aucun changement aux résultats, chiffres ou affirmations scientifiques n'a été apporté.
