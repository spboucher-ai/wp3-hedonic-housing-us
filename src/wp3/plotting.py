# Author: Simon-Pierre Boucher — contact@spboucher.ai
#
"""Publication style shared by all paper figures.

Design goals (housing/urban economics journal standard):
  - serif typography consistent with the paper's newtx text font;
  - muted, colorblind-safe palette (no default-Stata / default-matplotlib look);
  - no chart junk: top/right spines removed, light dotted y-grid only;
  - human-readable variable labels instead of raw column names;
  - panel titles "(a) ..." left-aligned, no redundant figure-level titles
    (the LaTeX caption carries the title).
"""

import matplotlib.pyplot as plt

# Muted palette
PRIMARY = "#3B6B9B"      # deep steel blue (marks, bars, boxes)
PRIMARY_LIGHT = "#7FA5C6"  # scatter clouds
ACCENT = "#A63E38"       # muted brick red (reference lines, binned means)
NEUTRAL = "#6E6E6E"      # gray for zero lines / secondary elements
CMAP_SEQ = "Blues"       # densities (hexbin)
CMAP_DIV = "coolwarm"    # SHAP dependence coloring
DPI = 300

# Human-readable labels for model variables (used in axis labels and SHAP plots)
LABELS = {
    "ln_sqft": "Log(Living Area)",
    "ln_lot": "Log(Lot Size)",
    "ln_hoa": "Log(HOA Fee)",
    "ln_price": "Log(Price)",
    "bedrooms": "Bedrooms",
    "bathrooms": "Bathrooms",
    "age": "Property Age",
    "age_sq": "Property Age$^2$",
    "stories": "Stories",
    "bath_per_bed": "Baths per Bedroom",
    "sqft_per_bed": "Sqft per Bedroom",
    "parking_spaces": "Parking Spaces",
    "luxury_score": "Luxury Score",
    "walk_score": "Walk Score",
    "bike_score": "Bike Score",
    "transit_score": "Transit Score",
    "school_count": "School Count",
    "avg_school_rating": "Avg. School Rating",
    "nearest_school_distance": "Nearest School Dist.",
    "property_tax_rate": "Property Tax Rate",
    "has_pool": "Pool",
    "has_spa": "Spa",
    "has_basement": "Basement",
    "has_fireplace": "Fireplace",
    "has_garage": "Garage",
    "on_waterfront": "Waterfront",
    "has_central_air": "Central Air",
    "has_forced_air": "Forced Air",
    "has_hardwood": "Hardwood Floors",
    "is_condo": "Condo",
    "has_hoa": "HOA",
    "tag_new_construction": "New Construction",
    "tag_foreclosure": "Foreclosure",
    "region_Northeast": "Region: Northeast",
    "region_South": "Region: South",
    "region_West": "Region: West",
    "region_Midwest": "Region: Midwest",
    "sqft_x_age": "Sqft $\\times$ Age",
    "pool_x_south": "Pool $\\times$ South",
    "waterfront_x_sqft": "Waterfront $\\times$ Log(sqft)",
    "basement_x_north": "Basement $\\times$ North",
    "condo_x_walkscore": "Condo $\\times$ Walk Score",
    "age_x_luxury": "Age $\\times$ Luxury",
}


def label(var: str) -> str:
    """Human-readable label for a model variable (falls back to the raw name)."""
    return LABELS.get(var, var)


def apply_paper_style() -> None:
    """Set the rcParams used for every figure in the paper."""
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13.5,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        # Clean journal look
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#333333",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.linestyle": ":",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.45,
        "grid.color": "#999999",
        "legend.frameon": False,
        "figure.dpi": 100,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def panel_title(ax, text: str) -> None:
    """Left-aligned panel title, e.g. '(a) OLS'."""
    ax.set_title(text, loc="left", fontweight="regular", pad=8)
