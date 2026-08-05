# Author: Simon-Pierre Boucher — contact@spboucher.ai
#
"""Project paths and analysis constants.

All paths are derived from the repository root so the pipeline can be run
from any working directory. Override the data location by setting the
environment variable ``WP3_ROOT`` before running any script.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("WP3_ROOT", Path(__file__).resolve().parents[2]))

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

DUCKDB_PATH = DATA_RAW / "us_housing.duckdb"
ANALYTICAL_SAMPLE = DATA_PROCESSED / "analytical_sample.pkl"
MODEL_DATA = DATA_PROCESSED / "model_data.pkl"
SHAP_DATA = DATA_PROCESSED / "shap_data.pkl"

# Reproducibility
RANDOM_STATE = 42
QR_STABILITY_SEEDS = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]
QR_SUBSAMPLE_SIZE = 150_000

# States held out entirely in the geographic-validation exercise
HOLDOUT_STATES = ["CA", "NY", "TX", "FL", "OH", "CO", "NC", "WA", "IL", "GA"]

# Feature blocks (raw, unstandardized names in analytical_sample)
STRUCTURAL_FEATS = ["ln_sqft", "bedrooms", "bathrooms", "age", "age_sq", "stories",
                    "bath_per_bed", "sqft_per_bed"]
LOT_FEATS = ["ln_lot"]
AMENITY_FEATS = ["has_pool", "has_spa", "has_basement", "has_fireplace", "has_garage",
                 "on_waterfront", "has_central_air", "has_forced_air", "has_hardwood",
                 "parking_spaces", "luxury_score"]
NEIGHBORHOOD_FEATS = ["walk_score", "bike_score", "transit_score", "avg_school_rating",
                      "school_count", "nearest_school_distance", "property_tax_rate"]
MARKET_FEATS = ["is_condo", "has_hoa", "ln_hoa", "tag_new_construction", "tag_foreclosure"]
INTERACTION_FEATS = ["sqft_x_age", "pool_x_south", "waterfront_x_sqft", "basement_x_north",
                     "condo_x_walkscore", "age_x_luxury"]

ALL_BASE_FEATS = (STRUCTURAL_FEATS + LOT_FEATS + AMENITY_FEATS + NEIGHBORHOOD_FEATS +
                  MARKET_FEATS + INTERACTION_FEATS)

CATEGORICAL_COLS = ["roof_cat", "construction_cat", "foundation_cat", "region"]

# Variables tracked in the quantile-regression robustness exercises
KEY_VARS = ["ln_sqft", "bedrooms", "bathrooms", "age", "age_sq",
            "ln_lot", "has_pool", "has_garage", "luxury_score",
            "tag_foreclosure", "on_waterfront", "region_Northeast", "region_West"]
