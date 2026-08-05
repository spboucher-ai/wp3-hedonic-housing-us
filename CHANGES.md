# CHANGES — Restructuration du projet WP3 (2026-08-05)

Original intact : `~/Desktop/UQO/UQO_WP/immo-wp3-spb-20260519/`.
Nouvelle structure : `~/Desktop/wp3_uqo/` (ce dossier). Voir `AUDIT.md` pour l'état des lieux initial.

---

## 1. Déplacements / réorganisation

| Origine (immo-wp3-spb-20260519/) | Destination (wp3_uqo/) |
|---|---|
| `us_housing.duckdb` | `data/raw/` |
| `analytical_sample.pkl`, `model_data.pkl`, `shap_data.pkl` | `data/processed/` |
| `qr_results.pkl`, `ml_results.pkl`, `extended_results.pkl`, `v3_*.pkl` | `results/` |
| `figures/fig*.png` (6 survivantes) | remplacées par les 11 régénérées dans `figures/` (originales conservées dans l'archive source) |
| `wp3/main.tex`, `wp3/sections/`, `wp3/appendix/`, `uq_logo.jpg`, `Makefile` | `paper/` |
| `run_v3_spatial.py` | refactoré → `scripts/01_spatial_robustness.py` |
| `v3_qr_imputation_analysis.py` | refactoré → `scripts/02_qr_imputation.py` |

Non repris (restent dans l'archive source uniquement) : `paper_hedonic_pricing{,_v2,_v3}.tex`
(anciennes versions), `wp3/main_web.tex` (contournement « Figure indisponible », devenu inutile),
fichiers auxiliaires LaTeX, `.DS_Store`.

## 2. Code refactoré

- **Nouveaux modules** `src/wp3/` : `config.py` (chemins relatifs, constantes, listes de variables),
  `data.py` (loaders + construction de la matrice de 62 régresseurs), `models.py` (entraînement
  XGBoost commun), `plotting.py` (style de figures + libellés lisibles).
- Les 2 scripts survivants sont réécrits en scripts numérotés avec docstrings, `argparse`,
  chemins relatifs (les chemins codés en dur `/Users/.../Desktop/immo-wp3-spb-20260519` ne
  fonctionnaient plus) et déduplication (le helper XGBoost et les listes de variables sont partagés).
- **Nouveaux scripts** : `03_make_figures.py` (11 figures depuis les pickles, avec statistiques de
  vérification), `04_export_tables.py` (tables du papier → `results/tables/*.csv`).
- `requirements.txt` épinglé sur l'environnement validé.
- En-tête `Author: Simon-Pierre Boucher — contact@spboucher.ai` ajouté à tous les fichiers de code
  et LaTeX (dans `references.bib`, l'e-mail est écrit sans `@` : BibTeX interprète `@` comme début d'entrée).

## 3. Figures

- **5 figures manquantes recréées** depuis les pickles (fig2 diagnostics OLS, fig4 prédit vs observé,
  fig7 carte, fig8 régions, fig9 relations bivariées) — les PNG originaux avaient été supprimés
  (~9 juin 2026) et le PDF existant contenait des boîtes « Figure indisponible ».
- **Les 11 figures régénérées dans un style homogène « journal »** (demande utilisateur) :
  typographie serif cohérente avec le papier, axes épurés (spines haut/droite retirés), grille
  discrète, palette sobre, libellés lisibles (« Log(Living Area) » au lieu de `ln_sqft`),
  titres de panneaux (a)/(b) alignés à gauche, 300 dpi.
- **Vérifications contenu = papier** (imprimées par `03_make_figures.py`) :
  skewness 0,18 / kurtosis 5,55 / JB 217 078 (fig2) ; R² OLS 0,6301 / XGB 0,8330 / LGB 0,8088 (fig4) ;
  OLS de référence ln_sqft 0,312 (fig3) — tous identiques aux valeurs du papier.
- Les 6 PNG originaux restent disponibles dans l'archive source pour comparaison.

## 4. Papier LaTeX (`paper/`)

Structure : `main.tex` + une section par fichier (déjà le cas), figures lues depuis `../figures/`
(`\graphicspath`), **bibliographie convertie du `thebibliography` manuel vers `references.bib`**
(37 entrées, clés identiques) + BibTeX/apalike. `sections/references.tex` supprimé.
Compilation : `latexmk` → **main.pdf, 52 pages, 0 erreur, 0 référence/citation indéfinie**
(l'archive originale n'avait pas de `main.pdf` compilable — figures manquantes).

### Complétions (placeholders `[TODO]` remplis avec les valeurs stockées dans les pickles)

- **Tableau des tests inter-quantiles** (results) : 8 lignes `[TODO: diff]/[TODO: z]` remplies depuis
  `v3_qr_imputation_results.pkl` (ln_lot −0,041/z=−9,49 ; luxury −0,054/−6,82 ; foreclosure −0,155/−4,18 ;
  Northeast −0,264/−21,43 ; West −0,086/−6,26 ; bedrooms −0,051/−5,95 ; age² +0,010/1,40 ;
  waterfront +0,222/1,86), lignes triées par |z|.
- **Tableau winsorisation** (robustness) : β ln_sqft 0,610 ; bathrooms 0,211 ; garage 0,111.
- **Tableau stabilité QR** : CV exacts (au lieu de « <7 ») et stabilité de signe 100 % pour
  waterfront/pool.
- **Annexe** : marqueurs `[TODO]` retirés de la liste « Robustness Agenda » (items non cochés =
  travaux futurs) ; checklist reproductibilité : « Code available: Yes (replication package) »,
  « Data available: On request ».

### Corrections d'incohérences (chiffres alignés sur les pickles — À VALIDER par l'auteur)

1. **En-tête du tableau inter-quantile** : disait β(0.90)−β(0.10) alors que les valeurs déjà
   présentes (garage +0,185, z=28,92…) correspondent à β(0.10)−β(0.90). En-tête corrigé.
2. **Moran's I** : le tableau affichait 0,268/0,278/0,277 avec z=42,4/44,0/43,8 et « k=10 » ;
   le pickle (`v3_spatial_results.pkl`) contient 0,2742/0,2839/0,2652 (moyenne 0,2745 et
   écart-type 0,0076 du papier ✓) calculés avec **k=8**. Valeurs et k corrigés ; z recalculés
   en ré-exécutant le pipeline refactoré (41,8/43,5/40,5) — la ré-exécution reproduit
   exactement les I du pickle (validation bout-en-bout du script 01). Méthodologie : k=10 → k=8.
3. **Description de la winsorisation** (data + robustness) : le texte disait « 1er/99e centile,
   toutes les variables continues » ; le code réel (script survivant + pickle) = lot au 99,5e
   centile (3 944 obs.) + Bike Score plafonné à 100 (38 obs.). Les chiffres du tableau
   (R² 0,640, β lot 0,058) proviennent bien de ce calcul. Description corrigée.
4. **« CV < 7 % »** → « CV < 8 % » (methodology, robustness, discussion, limitations) :
   bedrooms 7,3 % et foreclosure 7,1 % dépassent 7 %.
5. **Matrice de robustesse** : lignes FE mélangeaient R² in-sample (0,634) et out-of-sample
   (0,678/0,725) ; uniformisées en out-of-sample (0,630 → 0,678 (+4,8 pp) → 0,725 (+9,5 pp)),
   cohérent avec le tableau FE des résultats.
6. Le tableau QR-stabilité listait « Pool » à la fois comme stable (<7) et instable (33,4) ;
   doublon supprimé (CV réel 33,4 %).

**Aucun résultat, coefficient ou conclusion scientifique n'a été modifié** : toutes les
corrections ci-dessus alignent le texte sur les résultats effectivement stockés dans les pickles.

## 5. À vérifier par l'auteur

- Les 6 corrections numérotées ci-dessus (§4), en particulier Moran (k=8, nouvelles valeurs I/z).
- Le style des 11 figures régénérées (les 6 originales sont dans l'archive source si vous préférez
  l'ancien rendu pour certaines).
- Les scripts amont perdus (construction de l'échantillon depuis DuckDB, estimation OLS/QR/ML/SHAP)
  ne sont **pas** reconstruits — les artefacts stockés en tiennent lieu. Si une reconstruction
  complète raw→results est souhaitée, elle est possible mais devra être validée contre les pickles.
- `paper/main.tex` affiche « This version: \today » — figer la date avant soumission si besoin.
