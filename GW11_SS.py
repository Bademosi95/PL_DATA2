"""
Premier League Match Predictor - Streamlit App
-----------------------------------------------
Production version aligned with weekly_update.py pipeline.

Features:
- Three model views: Logistic, Poisson, Dixon-Coles
- Context adjustments for injuries/form/tactics
- Comprehensive team statistics
- Error handling and validation
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy.stats import poisson
from scipy.optimize import minimize
from pathlib import Path
from datetime import datetime, timezone
import json


def kelly_fraction(prob, odds):
    """Full Kelly fraction given win probability and decimal odds (binary market)."""
    b = odds - 1.0
    q = 1.0 - prob
    # classic Kelly for binary outcome; here we use 'prob' vs break-even
    return (prob * (b + 1.0) - 1.0) / b


def kelly_3way(probs, odds):
    """
    Proper Kelly optimisation for a 3-way mutually exclusive market (H/D/A).

    Args:
        probs: np.array of shape (3,) for [p_home, p_draw, p_away] (model probabilities)
        odds:  np.array of shape (3,) for decimal odds [o_home, o_draw, o_away]

    Returns:
        np.array of optimal Kelly fractions f (size 3), such that:
        - f_i >= 0
        - sum(f_i) <= 1
    """
    probs = np.asarray(probs, dtype=float)
    odds = np.asarray(odds, dtype=float)
    b = odds - 1.0  # net returns

    # If any probability or odds are degenerate, return zero fractions
    if np.any(probs < 0) or np.any(odds <= 1.0) or probs.sum() <= 0:
        return np.zeros(3)

    probs = probs / probs.sum()  # normalise just in case

    def neg_expected_log_growth(f):
        """
        Negative expected log growth to be minimised.

        Wealth factor if outcome i happens:
            W_i = 1 + f_i * b_i - sum_{j != i} f_j
        because we lose all stakes on other outcomes.
        """
        f = np.asarray(f)
        total_f = np.sum(f)
        # Enforce feasibility softly
        if total_f > 1.0 + 1e-8 or np.any(f < -1e-8):
            return 1e6

        expected_log = 0.0
        for i in range(3):
            gain = 1.0 + f[i] * b[i] - (total_f - f[i])
            # If gain <= 0, log undefined => huge penalty
            if gain <= 0:
                return 1e6
            expected_log += probs[i] * np.log(gain)
        return -expected_log

    # Constraints: sum(f) <= 1, f_i >= 0
    constraints = ({
        "type": "ineq",
        "fun": lambda f: 1.0 - np.sum(f)  # sum(f) <= 1
    },)

    bounds = [(0.0, 1.0)] * 3
    x0 = np.zeros(3)

    result = minimize(
        neg_expected_log_growth,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        return np.zeros(3)

    f_opt = np.maximum(result.x, 0.0)
    # tiny negatives to zero
    f_opt[f_opt < 1e-6] = 0.0
    return f_opt


def scaled_stake(kelly_fraction_base, bankroll, frac, max_pct):
    """
    Convert Kelly fraction (of bankroll) into a £ stake with:
    - fractional Kelly (frac in [0,1])
    - hard cap as % of bankroll (max_pct in [0,1])
    """
    k = max(0.0, float(kelly_fraction_base))  # no negative stakes
    raw = k * frac * bankroll
    cap = max_pct * bankroll
    return min(raw, cap)


def get_model_update_version():
    """Return update timestamp from metadata.json to bust Streamlit cache."""
    try:
        if Path("metadata.json").exists():
            with open("metadata.json", "r") as f:
                metadata = json.load(f)
                return metadata.get("update_time", "0")
    except Exception:
        pass
    return str(datetime.utcnow())

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="KB Premier League Match Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════
# CUSTOM STYLING
# ═══════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
    /* GLOBAL APP STYLING - Sofascore Light Vibe */
    body, .stApp {
        background: #F5F7FB;
        color: #111827;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", sans-serif;
    }

    .block-container {
        padding-top: 1.6rem !important;
        padding-bottom: 1.6rem !important;
        max-width: 1180px;
    }

    /* TITLES */
    .main-title {
        font-size: 40px;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin-bottom: 4px;
        color: #111827;
    }

    .sub-title {
        font-size: 17px;
        color: #4B5563;
        margin-bottom: 18px;
    }

    /* MATCH HEADER CARD */
    .headline-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 16px 22px 14px 22px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.05);
        margin-bottom: 16px;
    }

    .match-header {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
    }

    .match-teams {
        display: flex;
        align-items: center;
        gap: 14px;
        flex-wrap: wrap;
    }

    .team-block {
        display: flex;
        flex-direction: column;
        gap: 2px;
        min-width: 120px;
    }

    .team-name {
        font-size: 20px;
        font-weight: 700;
        color: #111827;
        white-space: nowrap;
    }

    .team-tag {
        font-size: 12px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #6B7280;
    }

    .vs-badge {
        font-size: 13px;
        font-weight: 700;
        color: #2563EB;
        padding: 4px 9px;
        border-radius: 999px;
        border: 1px solid #BFDBFE;
        background: #EFF6FF;
        text-transform: uppercase;
    }

    .match-meta {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 4px;
        min-width: 160px;
    }

    .meta-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #9CA3AF;
    }

    .meta-value {
        font-size: 13px;
        color: #1F2937;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-top: 1.0rem !important;
            padding-left: 0.8rem !important;
            padding-right: 0.8rem !important;
        }
        .match-header {
            flex-direction: column;
            align-items: flex-start;
        }
        .match-meta {
            align-items: flex-start;
            width: 100%;
            border-top: 1px solid #E5E7EB;
            padding-top: 8px;
            margin-top: 4px;
        }
        .team-name {
            font-size: 18px;
        }
        .model-card {
            margin-bottom: 12px;
        }
        /* Stack horizontal blocks (e.g. 3 model columns) on mobile */
        section[data-testid="stHorizontalBlock"] > div {
            flex-direction: column !important;
        }
    }

    /* GENERIC CARDS */
    .model-card, .stat-card {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 18px 20px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        color: #111827;
        height: 100%;
    }

    .model-card h4 {
        margin-top: 0;
        margin-bottom: 12px;
        font-size: 18px;
        font-weight: 700;
        color: #111827;
        border-bottom: 2px solid #F3F4F6;
        padding-bottom: 6px;
    }

    /* MODEL METRICS INSIDE CARDS */
    .model-metric-row {
        display: flex;
        justify-content: space-between;
        font-size: 15px;
        margin: 7px 0;
        color: #111827;
    }

    .model-metric-label {
        color: #4B5563;
    }

    .model-metric-value {
        font-weight: 600;
        color: #111827;
    }

    .metric-highlight {
        background: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        border-radius: 999px;
        padding: 3px 10px;
        font-size: 14px;
        font-weight: 600;
    }

    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background: #FFFFFF !important;
        border-right: 1px solid #E5E7EB;
    }

    [data-testid="stSidebar"] * {
        color: #111827 !important;
        font-size: 14px;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #111827 !important;
    }

    /* STREAMLIT METRIC TILES */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        padding: 10px 14px;
        box-shadow: 0 1px 5px rgba(15, 23, 42, 0.05);
    }

    [data-testid="stMetricValue"] {
        font-size: 22px;
        font-weight: 700;
        color: #111827;
    }

    [data-testid="stMetricLabel"] {
        font-size: 13px;
        color: #4B5563;
    }

    /* TABLES / DATAFRAMES */
    [data-testid="stDataFrame"] {
        background: #FFFFFF;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 5px rgba(15, 23, 42, 0.04);
    }

    [data-testid="stDataFrame"] table {
        font-size: 14px;
        color: #111827;
    }

    /* EXPANDERS */
    details {
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        background: #FFFFFF;
        padding: 6px 10px 12px 10px;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
        margin-bottom: 10px;
    }

    summary {
        font-size: 15px;
        font-weight: 600;
        color: #111827;
    }

    /* INFO / WARNING / SUCCESS BOXES */
    .info-box {
        background: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 10px 12px;
        border-radius: 8px;
        margin: 8px 0;
        color: #1E3A8A;
        font-size: 14px;
    }

    .warning-box {
        background: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 10px 12px;
        border-radius: 8px;
        margin: 8px 0;
        color: #92400E;
        font-size: 14px;
    }

    .success-box {
        background: #ECFDF5;
        border-left: 4px solid #10B981;
        padding: 10px 12px;
        border-radius: 8px;
        margin: 8px 0;
        color: #065F46;
        font-size: 14px;
    }

    /* FOOTER */
    .footer-note {
        margin-top: 18px;
        font-size: 13px;
        color: #4B5563;
        text-align: left;
        padding: 14px 0 0 0;
        border-top: 1px solid #E5E7EB;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# (TRUNCATED: For this downloadable version, we're including the header + styling +
#  Kelly functions; the rest of the app structure should be pasted from your
#  working version, as message length limits in this environment prevent sending
#  the entire script in one go.)
