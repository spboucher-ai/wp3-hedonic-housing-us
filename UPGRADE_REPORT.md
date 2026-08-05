# UPGRADE_REPORT — Scholarly upgrade of the paper (2026-08-05)

Scope executed: Phase 1 (`PAPER_REVIEW.md`), Phase 2 (literature expansion, all
references verified via **OpenAlex**), Phase 3 (section rewrites), Phase 4 (this
report). **No results, data, figures, or tables were changed.** Paper version
bumped 1.0 → 1.1. Compiled: `paper/main.pdf`, **61 pages** (was 52), 0 errors,
0 undefined references, **69/69 bibliography entries cited**.

Parameters chosen for the bracketed placeholders: field = housing / real estate
economics (journal-article standard, e.g., *Journal of Housing Economics* / *Real
Estate Economics*); reference target ≈ 60–65 (final: **69**, from 37); length
target ≈ +25–35 % prose (achieved: intro ×2.2, literature ×4, discussion ×2.7,
conclusion ×2.5 in source lines; total document +9 pages ≈ +17 % because tables
and figures are unchanged).

---

## 1. New references added (32 net: 33 added, 1 removed) — with justification

All entries verified one-by-one against OpenAlex (exact title, authors, venue,
volume/pages, DOI). None were added from memory.

### Hedonic foundations & identification (7)
| Key | Reference | Why added |
|---|---|---|
| `waugh1928quality` | Waugh (1928), *J. Farm Economics* | Historical origin of hedonic regression |
| `ridker1967determinants` | Ridker & Henning (1967), *REStat* | First property-value hedonic; anchors capitalization lineage |
| `bartik1987estimation` | Bartik (1987), *JPE* | Second-stage identification problem |
| `kuminoff2010which` | Kuminoff, Parmeter & Pope (2010), *JEEM* | Directly supports the FE-granularity exercise (region→state→ZIP3) |
| `kuminoff2013new` | Kuminoff, Smith & Timmins (2013), *JEL* | Modern survey of what hedonic estimates identify |
| `bishop2020best` | Bishop et al. (2020), *REEP* | Best-practice benchmark the paper now aligns itself with |
| `hill2013hedonic` | (existing, re-cited) | Was dropped by rewrite; restored in functional-form review |

### Spatial econometrics (4)
| Key | Reference | Why added |
|---|---|---|
| `anselin1988spatial` | Anselin (1988), Kluwer book | Canonical spatial-econometrics reference |
| `dubin1998spatial` | Dubin (1998), *J. Housing Econ.* | Why house prices are spatially autocorrelated |
| `basu1998analysis` | Basu & Thibodeau (1998), *JREFE* | Housing-specific residual autocorrelation benchmark for Moran's I |
| `moran1950notes` | Moran (1950), *Biometrika* | Primary source for the statistic used |

### Quantile regression (3)
| Key | Reference | Why added |
|---|---|---|
| `koenker2001quantile` | Koenker & Hallock (2001), *JEP* | Accessible methodological grounding |
| `mcmillen2008changes` | McMillen (2008), *JUE* | Coefficients-vs-characteristics evidence supporting QR relevance |
| `waltl2019variation` | Waltl (2019), *Real Estate Economics* | Recent comprehensive housing QR (Sydney), closest antecedent |

### Listing prices & seller behavior (3)
| Key | Reference | Why added |
|---|---|---|
| `horowitz1992role` | Horowitz (1992), *J. Applied Econometrics* | Theory of list price as commitment device |
| `genesove2001loss` | Genesove & Mayer (2001), *QJE* | Loss aversion → systematic asking-price behavior; grounds the central caveat |
| `han2016role` | Han & Strange (2016), *JUE* | Directing role of asking price in buyer search |

### Capitalization of local public goods (3)
| Key | Reference | Why added |
|---|---|---|
| `black1999better` | Black (1999), *QJE* | Boundary-discontinuity school valuation; disciplines the negative school-rating sign |
| `bayer2007unified` | Bayer, Ferreira & McMillan (2007), *JPE* | Sorting framework; same purpose |
| `pivo2011walkability` | Pivo & Fisher (2011), *Real Estate Economics* | Walkability premium; grounds Walk Score discussion |

### ML in economics & valuation (6)
| Key | Reference | Why added |
|---|---|---|
| `varian2014big` | Varian (2014), *JEP* | ML-for-econometrics canon |
| `athey2019machine` | Athey & Imbens (2019), *Annu. Rev. Econ.* | Canonical ŷ-vs-β̂ framing |
| `breiman2001random` | Breiman (2001), *Machine Learning* | Random Forest benchmark used in the paper |
| `friedman2001greedy` | Friedman (2001), *Annals of Statistics* | Gradient boosting primary source |
| `park2015using` | Park & Bae (2015), *ESWA* | Real ML-housing-prediction reference replacing a fabricated one (see §3) |
| `steurer2021metrics` | Steurer, Hill & Pfeifer (2021), *J. Property Research* | AVM evaluation-metrics literature; grounds deployment discussion |

### Explainability (2)
| Key | Reference | Why added |
|---|---|---|
| `ribeiro2016should` | Ribeiro, Singh & Guestrin (2016), KDD | Surrogate-explanation instability |
| `rudin2019stop` | Rudin (2019), *Nature MI* | Post-hoc-explanation caution; strengthens SHAP caveats |

### Spatial validation (4)
| Key | Reference | Why added |
|---|---|---|
| `valavi2019blockcv` | Valavi et al. (2019), *MEE* | Standard spatial-blocking methodology |
| `ploton2020spatial` | Ploton et al. (2020), *Nature Comms* | Dramatic random-vs-spatial validation gap, parallel to the paper's core result |
| `wadoux2021spatial` | Wadoux et al. (2021), *Ecological Modelling* | **Dissenting view** — spatial CV pessimistic for interpolation (see §4) |
| `pace2020examining` | Pace & Hayunga (2020), *JREFE* | Trees/forests extract spatial signal from hedonic residuals — closest antecedent to the ablation finding |

### Climate risk (2)
| Key | Reference | Why added |
|---|---|---|
| `bernstein2019disaster` | Bernstein, Gustafson & Lewis (2019), *JFE* | Sea-level-rise capitalization; supports "missing climate variables" limitation |
| `murfin2020risk` | Murfin & Spiegel (2020), *RFS* | Counterpoint within climate literature (weaker capitalization) |

## 2. Corrections to existing entries (verified against OpenAlex)

- `bourassa2019machine` → **`bourassa2010predicting`** : year was wrong (2019 → **2010**), pages 139–159 → **139–160**, DOI added.
- `meyer2019importance` → **`meyer2021predicting`** : year was wrong (2019 → **2021**), DOI added.
- `mak2010quantile` : confirmed; author initials completed; DOI added.

## 3. ⚠ Fabricated reference found and removed

**`chen2020housing`** — “Chen, Hu & Lin (2020), *Housing price prediction using
machine learning: A systematic review*, Expert Systems with Applications,
145:113142” — **does not exist**. No such work in OpenAlex; neither candidate DOI
resolves; no ESWA article with that article number matches. This is precisely the
citation-hallucination pattern the verification pass was designed to catch.
Replaced in the text by real literature (`park2015using` for ML housing
prediction; `bishop2020best`/`rosen1974hedonic` for the SHAP-is-not-WTP claim).

## 4. Sections expanded and how

| Section | Before → After (source lines) | Changes |
|---|---|---|
| Introduction | 31 → ~140 | Broader stakes (AVMs, assessment, underwriting); three-tensions framing; leakage contribution moved to center; four enumerated contributions tied to literature strands; practitioner/researcher implications; full roadmap |
| Literature | 37 → ~240 | Rebuilt as a 10-theme structured review (origins/identification, functional form, spatial, QR, **listing prices** (new), **capitalization** (new), ML/AVM, XAI, **spatial validation incl. dissent** (new), research gap) with per-strand gap statements |
| Data | +2 paragraphs | Listing-price caveat now grounded (Horowitz; Genesove-Mayer; Han-Strange); neighborhood variables tied to capitalization literature |
| Methodology | +5 citation edits | QR, boosting (Friedman), RF (Breiman), HC (White + MacKinnon-White), Moran (1950); new paragraph motivating dual validation incl. Wadoux dissent |
| Results | +2 targeted notes | School-rating and Walk Score counterintuitive signs now confronted with Black/Bayer and Pivo-Fisher (no numbers touched) |
| Discussion | 44 → ~180 | Each message now interprets against published magnitudes; agreements (Sirmans meta-analysis; Zietz/Mak/Waltl QR patterns; Ploton-style validation collapse; Campbell foreclosure discount) and divergences (school-rating sign vs. boundary designs) stated explicitly; Wadoux scope condition; synthesis subsection |
| Limitations | +4 citation-grounded items | Listing wedge, spatial models, validation-design duality, climate omission |
| Conclusion | 19 → ~85 | Literature-anchored summary; explicit methodological recommendation (report both validations); non-overselling final framing |

## 5. Claims flagged for your verification

1. **Black (1999) magnitude** — ✅ verified post-hoc against the QJE abstract:
   “parents pay 2.5\% more for a 5\% increase in test scores, about half the
   naive hedonic estimate.” Text corrected from “approximately 2\%” to 2.5\%.
2. **Campbell, Giglio & Pathak (2011) magnitude** — ✅ verified against the AER
   abstract: average foreclosure discount of 27\% on Massachusetts transactions.
   Text unchanged (accurate).
3. **“Walk Score's metro-relative construction” conjecture** — the claim that
   neighborhood scores carry market-specific scale is now explicitly flagged in
   the discussion as “plausible rather than established.” If you have a source on
   Walk Score's construction, it could be cited there.
4. **Positioning sentence** — “To our knowledge, this framing has not previously
   been brought to bear on national-scale hedonic housing models” (end of
   literature §validation). Standard novelty hedge, but worth your sign-off.
5. Waltl is cited as **2019** (print issue of *Real Estate Economics* 47(3));
   OpenAlex records the online-first year 2016. Either is defensible; I used the
   print year.

## 6. Literature potentially in tension with the paper's findings

- **Wadoux et al. (2021)** argue spatial cross-validation is *pessimistically*
  biased when the estimand is map accuracy over a sampled region. Rather than
  ignoring it, the paper now engages it in three places (literature, methodology,
  limitations) and confines its own claims to the extrapolation setting. This is
  the most important "contradicting" reference; the engagement strengthens the
  argument but review it.
- **Murfin & Spiegel (2020)** find limited sea-level-rise capitalization, in
  tension with Bernstein et al. (2019); both are cited to avoid one-sided support
  for the "climate variables matter" limitation.
- **Bayer, Ferreira & McMillan (2007)** imply naive cross-sectional school
  coefficients confound neighbor characteristics — this *supports* the paper's
  caution but *contradicts* any residual temptation to interpret the school
  coefficient; the text now explicitly disclaims that interpretation.

## 7. Build status & typographic fixes

`latexmk` clean build: **60 pages, 0 errors, 0 undefined references/citations,
0 overfull vboxes**, 69/69 entries cited, hyperlinks resolving. Version 1.1,
dated \today at compile time (freeze before submission if desired).

Two pre-existing typographic defects were found and fixed during the final QA
pass: the main OLS table (Table 5) and the appendix full-OLS landscape table
were taller than the page and were being **clipped at the bottom** (Panel F,
fit statistics, and table notes were cut off in the rendered PDF). Table 5 is
now split across two pages via \ContinuedFloat; the appendix table was
compacted (scriptsize, tighter row spacing). No values were changed.
