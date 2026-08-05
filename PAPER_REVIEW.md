# PAPER_REVIEW — Critical assessment before the scholarly upgrade (2026-08-05)

Scope: `paper/` (52-page compiled working paper). Assessment based on a full read of
all sections, the analysis code (`src/`, `scripts/`), and the stored results.
**No results, data, or figures are questioned here — this is about framing, literature,
and argumentation only.**

---

## 1. Core contribution — is it clearly stated?

**What the paper claims:** a methodological + empirical comparison of OLS, quantile
regression, and gradient boosting on one large listing dataset, with a systematic
demonstration of *spatial leakage* in ML evaluation (geographic holdout, ablation,
lat/lon experiment).

**Assessment:** the contribution IS stated (intro, ¶5–6) and is genuinely interesting —
the spatial-leakage triad is the paper's strongest and most novel element. But:

- The intro *underplays* it: the three "persistent limitations" framing (linearity,
  mean-only, black-box) is generic and could open any of a hundred ML-hedonics papers.
  The leakage finding — that geographic features actively *harm* generalization — is
  the distinctive result and deserves to lead.
- The contribution statement is not benchmarked against the closest existing work
  (nothing tells the reader what the *delta* is vs. Bourassa et al.'s spatial-ML
  comparisons or vs. the spatial-CV literature imported from ecology).
- "First paper to X" claims are absent (good — none would survive), but the paper
  never says precisely *which combination* is new: national scale + listing data +
  three-way framework comparison + spatial-leakage quantification.

## 2. Literature review — gaps

Current: 37 references, 7 short subsections. Solid skeleton (Rosen/Lancaster
foundations, functional form, QR, ML/AVM, SHAP, spatial validation), but thin for a
journal submission in housing/urban economics. Specific gaps:

| Missing strand | Why it matters here | Representative works to add |
|---|---|---|
| **Pre-Rosen hedonic history** | Court/Griliches are cited but the agricultural origin (Waugh 1928) and the environmental-valuation lineage (Ridker & Henning 1967) anchor the method's breadth | Waugh (1928); Ridker & Henning (1967) |
| **Hedonic identification syntheses** | The paper leans on Ekeland et al. (2004) alone; the modern surveys of what hedonic coefficients can and cannot identify are absent | Kuminoff, Smith & Timmins (2013 JEL); Bishop et al. (2020); Bartik (1987); Epple (1987) |
| **Specification robustness / FE granularity** | The region→state→ZIP3 exercise begs for Kuminoff, Parmeter & Pope (2010), which asks exactly "which hedonic models can we trust" and finds spatial FE crucial | Kuminoff, Parmeter & Pope (2010 JEEM) |
| **Spatial hedonics beyond LeSage-Pace** | Only 3 spatial refs; no housing-specific spatial autocorrelation classics | Dubin (1998); Basu & Thibodeau (1998); Anselin (1988); Pace & Gilley (1997) |
| **Quantile methods depth** | Koenker & Bassett + three applications; missing the accessible survey and the housing-distribution literature | Koenker & Hallock (2001); McMillen (2008); Waltl (2016) |
| **Listing-price / search literature** | The dependent variable IS an asking price; only Knight (2002) cited. The strategic-pricing and loss-aversion literature directly supports the paper's central caveat | Genesove & Mayer (2001); Horowitz (1992); Han & Strange (2016); Haurin (1988) |
| **Capitalization of local public goods** | School rating and property-tax coefficients get counterintuitive signs; the boundary-discontinuity literature is the natural reference point | Black (1999); Bayer, Ferreira & McMillan (2007) |
| **ML-in-economics methodology** | Mullainathan & Spiess is alone; the econometrics-meets-ML canon is absent | Varian (2014); Athey & Imbens (2019); Breiman (2001); Friedman (2001) |
| **AVM evaluation practice** | The AVM framing (deployment, generalization) has its own metrics literature | Steurer, Hill & Pfeifer (2021); Pace & Hayunga (2020) |
| **Explainability debate** | SHAP caveats are well written but un-cited beyond Lundberg; the interpretability-vs-explanation debate strengthens them | Rudin (2019); Ribeiro et al. (2016); Molnar et al. (2020) — verify availability |
| **Spatial CV methods** | Roberts et al. + Meyer & Pebesma only; the blocking-methods and the *dissenting* literature are missing | Valavi et al. (2019); Ploton et al. (2020); Wadoux et al. (2021) — the last one *argues against* spatial CV and must be engaged, not ignored |
| **Climate risk pricing** | One reference (Baldauf et al.); this is now a large literature and the paper lists climate data as future work | Bernstein, Gustafson & Lewis (2019); Murfin & Spiegel (2020) |
| **Walkability premium** | Walk/Bike/Transit scores are regressors but no walkability-capitalization citation exists | Pivo & Fisher (2011) |
| **Moran's I primary source** | The statistic is used but Moran (1950) / Cliff & Ord are not cited | Moran (1950) |

**Suspect existing entries (to verify in Phase 2):**
- `bourassa2019machine` — dated 2019 but *JRER* 32(2), 139–159 is the **2010** volume.
- `meyer2019importance` — dated 2019 but *MEE* 12(9), 1620–1633 is **2021**; exact
  title/venue need confirmation.
- `chen2020housing` — "Expert Systems with Applications, 145:113142" needs
  author/title/article-number confirmation.
- `mak2010quantile` — plausible but verify volume/pages.

## 3. Weak argumentation / unsupported claims

1. **"Neighborhood-quality features … carry location-specific scale and meaning that
   may not transfer"** (discussion) — plausible mechanism, no citation, no test.
   Should be flagged as conjecture or supported (e.g., Walk Score's metro-relative
   construction).
2. **"Luxury properties are more frequently overpriced, distressed properties may be
   strategically underpriced"** (data section) — cited only to Knight (2002), which
   does not establish both claims; Genesove & Mayer / Han & Strange needed.
3. **Counterintuitive-signs subsection** — the school-rating and tax-rate explanations
   are multicollinearity narratives without references to the capitalization
   literature (Oates is cited elsewhere but not connected here; Black 1999 missing).
4. **"This is below unity, consistent with diminishing marginal returns to space"** —
   fine, but a comparison to the elasticity range in published meta-analyses
   (Sirmans et al. report living-area gradients) would ground it.
5. **The spatial-leakage argument never engages the counter-position** — Wadoux et al.
   (2021) argue spatial CV can be *pessimistically* biased under uniform sampling.
   Engaging this strengthens, not weakens, the paper's design (10-state holdout is an
   extrapolation task, where blocking is defensible).
6. **Abstract/intro report the ablation-variant geo R² (0.425)** while the headline
   model reaches 0.547 — internally explained (different training composition), but a
   reviewer will push; the discussion should own this more explicitly.

## 4. Underdeveloped sections

- **Introduction (31 lines):** no broader stakes (housing = ~$45T US asset class,
  AVM industry, algorithmic valuation policy debates); no explicit "contributions"
  enumeration tied to literature strands; roadmap is one flat sentence.
- **Literature (37 lines):** each subsection is 3–6 lines — closer to an annotated
  list than a review. No synthesis paragraphs, no explicit "gap" argument per strand
  (the Research Gap subsection does some of this but in generic terms).
- **Discussion (44 lines):** four "messages" are well structured but interpret results
  almost entirely *internally* — few comparisons with published magnitudes
  (elasticities, QR patterns vs. Zietz et al., ML gains vs. Bourassa et al.,
  geographic-transfer losses vs. ecology findings).
- **Limitations (37 lines):** good coverage, telegraphic style; several items could
  cite the literature that documents the problem (e.g., listing-vs-transaction:
  Genesove & Mayer; spatial CV design choice: Valavi/Wadoux).
- **Conclusion (19 lines):** adequate; can be sharpened with one paragraph on external
  validity and one on the research agenda, without overselling.

## 5. Positioning: current vs. recommended

**Current de facto positioning:** "a careful multi-method comparison with unusually
honest caveats" — reads like a very good methods-audit working paper.

**Recommended positioning:** "evidence on *when and why* ML predictive advantages in
hedonic valuation are real vs. artifacts of validation design, at national scale" —
i.e., lead with the spatial-leakage contribution, use the three-framework comparison
as the vehicle, and connect explicitly to (a) the hedonic-identification literature
(what the coefficients mean), (b) the spatial-CV literature (what the metrics mean),
and (c) the AVM-deployment literature (why practitioners should care).

## 6. What must NOT change

All numbers, tables, figures, estimation choices, and robustness results stay exactly
as they are. The upgrade is: framing, motivation, literature integration, discussion
depth, and citation grounding. Any tension found between the paper's findings and the
added literature is to be *reported* (UPGRADE_REPORT.md) and *discussed* in the text,
never resolved by altering results.
