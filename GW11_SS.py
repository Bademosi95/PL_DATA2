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
    """Full Kelly fraction given win probability and decimal odds."""
    b = odds - 1.0
    q = 1.0 - prob
    # classic Kelly for binary outcome; here we use 'prob' vs break-even
    return (prob * (b + 1.0) - 1.0) / b



def kelly_3way(probs, odds):
    """
    Proper Kelly optimisation for a 3-way mutually exclusive market (H/D/A).

    Args:
        probs: np.array of shape (3,) for [p_home, p_draw, p_away]
        odds:  np.array of shape (3,) for decimal odds [o_home, o_draw, o_away]

    Returns:
        np.array of optimal Kelly fractions f (size 3), such that:
        - f_i >= 0
        - sum(f_i) <= 1
    """
    probs = np.asarray(probs, dtype=float)
    odds = np.asarray(odds, dtype=float)
    b = odds - 1.0  # net returns

    # Guard against degenerate inputs
    if np.any(probs < 0) or np.any(odds <= 1.0) or probs.sum() <= 0:
        return np.zeros(3)

    probs = probs / probs.sum()  # normalise

    def neg_expected_log_growth(f):
        """Negative expected log growth to be minimised."""
        f = np.asarray(f)
        total_f = np.sum(f)
        # Soft feasibility check
        if total_f > 1.0 + 1e-8 or np.any(f < -1e-8):
            return 1e6

        expected_log = 0.0
        for i in range(3):
            # Wealth factor if outcome i occurs:
            # W_i = 1 + f_i * b_i - sum_{j != i} f_j
            gain = 1.0 + f[i] * b[i] - (total_f - f[i])
            if gain <= 0:
                return 1e6  # log undefined -> large penalty
            expected_log += probs[i] * np.log(gain)
        return -expected_log

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
    f_opt[f_opt < 1e-6] = 0.0  # clean tiny negatives
    return f_opt


def scaled_stake(kelly_full, bankroll, frac, max_pct):
    """
    Convert full Kelly fraction into a £ stake with:
    - fractional Kelly (frac in [0,1])
    - hard cap as % of bankroll (max_pct in [0,1])
    """
    k = max(0.0, float(kelly_full))  # no negative stakes
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

# ═══════════════════════════════════════════════════════════════════
# LOAD MODELS & DATA WITH VALIDATION
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource
def load_models(_version):
    """Load all pickled models and data with comprehensive error handling"""
    required_files = {
        "pipe_result_final.pkl": "Logistic regression model",
        "poisson_model.pkl": "Poisson GLM model",
        "long_df.pkl": "Match history data",
        "stats.pkl": "Team statistics",
        "rho_hat.pkl": "Dixon-Coles rho parameter",
        "feature_cols.pkl": "Feature column order"
    }

    missing_files = []
    for file, desc in required_files.items():
        if not Path(file).exists():
            missing_files.append(f"• {file} ({desc})")

    if missing_files:
        st.error("### ❌ Missing Required Model Files")
        st.error("\n".join(missing_files))
        st.error("\n**Action required:** Run `weekly_update.py` to generate all model files.")
        st.stop()

    try:
        with open("pipe_result_final.pkl", "rb") as f:
            pipe_result = pickle.load(f)

        with open("poisson_model.pkl", "rb") as f:
            poisson_model = pickle.load(f)

        with open("long_df.pkl", "rb") as f:
            long_df = pickle.load(f)

        with open("stats.pkl", "rb") as f:
            stats = pickle.load(f)

        with open("rho_hat.pkl", "rb") as f:
            rho_hat = pickle.load(f)

        # Feature columns: must exist and match training
        with open("feature_cols.pkl", "rb") as f:
            feature_cols = pickle.load(f)

        # Load metadata if available
        metadata = {}
        if Path("metadata.json").exists():
            with open("metadata.json", "r") as f:
                metadata = json.load(f)

        # Validate data
        if stats.empty or long_df.empty:
            st.error("❌ Loaded data is empty. Please regenerate models.")
            st.stop()

        # Normalize team names
        stats["Squad"] = stats["Squad"].astype(str).str.strip()
        long_df["team"] = long_df["team"].astype(str).str.strip()

        # Check required columns in stats
        required_stats_cols = [
            "Squad", "Home_xG", "Home_xGA", "Home_MP",
            "Away_xG", "Away_xGA", "Away_MP"
        ]
        missing_cols = [col for col in required_stats_cols if col not in stats.columns]

        if missing_cols:
            st.error(f"❌ Stats file missing required columns: {missing_cols}")
            st.stop()

        # Check required columns in long_df
        required_long_cols = [
            "team", "Date", "rolling_points", "rolling_xg_for",
            "rolling_xg_against", "rolling_GD",
            "rolling_finishing_overperf", "rolling_def_overperf"
        ]
        missing_cols = [col for col in required_long_cols if col not in long_df.columns]

        if missing_cols:
            st.error(f"❌ long_df missing required columns: {missing_cols}")
            st.stop()

        return {
            "pipe_result": pipe_result,
            "poisson_model": poisson_model,
            "long_df": long_df,
            "stats": stats,
            "rho_hat": float(rho_hat) if not isinstance(rho_hat, float) else rho_hat,
            "feature_cols": feature_cols,
            "metadata": metadata
        }

    except Exception as e:
        st.error("### ❌ Error Loading Models")
        st.error(f"```\n{str(e)}\n```")
        st.error("Please regenerate models by running `weekly_update.py`")
        st.stop()


# Load all models
models = load_models(get_model_update_version())

pipe_result_final = models["pipe_result"]
poisson_model = models["poisson_model"]
long_df = models["long_df"]
stats = models["stats"]
rho_hat = models["rho_hat"]
feature_cols = models["feature_cols"]
metadata = models["metadata"]

# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def pct(x, decimals=1):
    """Format as percentage"""
    return f"{x * 100:.{decimals}f}%"


def safe_division(numerator, denominator, default=0.0):
    """Safe division with fallback"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except Exception:
        return default


def get_latest_team_stat(team, column, default=None):
    """
    Safely get latest stat for a team from long_df.

    Returns `default` (None by default) if:
    - team has no data
    - column missing
    - value is NaN

    This lets us distinguish "no data" from genuine 0.
    """
    try:
        team_data = long_df[long_df["team"] == team].sort_values("Date")

        if team_data.empty:
            return default

        latest = team_data.iloc[-1]

        if column not in latest.index:
            return default

        value = latest[column]

        if pd.isna(value):
            return default

        return float(value)

    except Exception as e:
        st.warning(f"Error retrieving {column} for {team}: {e}")
        return default


def get_team_strength_metrics(team):
    """Get season-long strength metrics from stats file"""
    try:
        team_row = stats[stats["Squad"] == team]

        if team_row.empty:
            st.error(f"❌ Team '{team}' not found in stats file")
            return None

        row = team_row.iloc[0]

        return {
            "att_home": safe_division(row["Home_xG"], row["Home_MP"], 0.0),
            "def_home": safe_division(row["Home_xGA"], row["Home_MP"], 0.0),
            "att_away": safe_division(row["Away_xG"], row["Away_MP"], 0.0),
            "def_away": safe_division(row["Away_xGA"], row["Away_MP"], 0.0)
        }

    except Exception as e:
        st.error(f"Error getting strength for {team}: {e}")
        return None


def build_feature_vector(home_team, away_team):
    """
    Build feature vector for logistic regression.
    Must match exact feature engineering from weekly_update.py.
    """
    def _rolling_or_fallback(val, fallback=0.0):
        # Explicitly convert None (no data) to numerical fallback
        return fallback if val is None else val

    # Get strength metrics
    home_strength = get_team_strength_metrics(home_team)
    away_strength = get_team_strength_metrics(away_team)

    if home_strength is None or away_strength is None:
        return None

    # Get rolling form metrics (last 5 matches)
    home_rolling = {
        "points": get_latest_team_stat(home_team, "rolling_points"),
        "xg_for": get_latest_team_stat(home_team, "rolling_xg_for"),
        "xg_against": get_latest_team_stat(home_team, "rolling_xg_against"),
        "GD": get_latest_team_stat(home_team, "rolling_GD"),
        "finishing_overperf": get_latest_team_stat(home_team, "rolling_finishing_overperf"),
        "def_overperf": get_latest_team_stat(home_team, "rolling_def_overperf")
    }

    away_rolling = {
        "points": get_latest_team_stat(away_team, "rolling_points"),
        "xg_for": get_latest_team_stat(away_team, "rolling_xg_for"),
        "xg_against": get_latest_team_stat(away_team, "rolling_xg_against"),
        "GD": get_latest_team_stat(away_team, "rolling_GD"),
        "finishing_overperf": get_latest_team_stat(away_team, "rolling_finishing_overperf"),
        "def_overperf": get_latest_team_stat(away_team, "rolling_def_overperf")
    }

    # Build features matching training exactly, with explicit fallbacks
    features = {
        "strength_diff": home_strength["att_home"] - away_strength["att_away"],
        "defense_diff": away_strength["def_away"] - home_strength["def_home"],
        "rolling_points_diff": _rolling_or_fallback(home_rolling["points"])
        - _rolling_or_fallback(away_rolling["points"]),
        "rolling_xG_diff": _rolling_or_fallback(home_rolling["xg_for"])
        - _rolling_or_fallback(away_rolling["xg_for"]),
        "rolling_xGA_diff": _rolling_or_fallback(home_rolling["xg_against"])
        - _rolling_or_fallback(away_rolling["xg_against"]),
        "rolling_GD_diff": _rolling_or_fallback(home_rolling["GD"])
        - _rolling_or_fallback(away_rolling["GD"]),
        "finishing_overperf_diff": _rolling_or_fallback(home_rolling["finishing_overperf"])
        - _rolling_or_fallback(away_rolling["finishing_overperf"]),
        "def_overperf_diff": _rolling_or_fallback(home_rolling["def_overperf"])
        - _rolling_or_fallback(away_rolling["def_overperf"])
    }

    # Return DataFrame with correct column order
    return pd.DataFrame([features])[feature_cols]


def predict_logistic(home_team, away_team):
    """Get logistic regression predictions (H/D/A probabilities)"""
    try:
        X = build_feature_vector(home_team, away_team)

        if X is None:
            return None

        probs = pipe_result_final.predict_proba(X)[0]
        classes = pipe_result_final.classes_

        return dict(zip(classes, probs))

    except Exception as e:
        st.error(f"Logistic prediction error: {e}")
        return None


def get_poisson_lambdas(home_team, away_team, context_adj=0.0):
    """
    Get expected goals from Poisson model with optional context adjustment.

    Args:
        home_team: Home team name
        away_team: Away team name
        context_adj: Adjustment factor (~-0.3 to +0.3)
            > 0: boost home xG, reduce away xG
            < 0: reduce home xG, boost away xG

    Returns:
        (lam_home_adjusted, lam_away_adjusted, lam_home_base, lam_away_base)
    """
    try:
        # Predict base lambdas
        pred_df = pd.DataFrame({
            "team": [home_team, away_team],
            "opponent": [away_team, home_team],
            "is_home": [1, 0]
        })

        predictions = poisson_model.predict(pred_df)
        lam_home_base = float(predictions.iloc[0])
        lam_away_base = float(predictions.iloc[1])

        # Apply context adjustment
        lam_home_adj = max(lam_home_base * (1 + context_adj), 0.01)
        lam_away_adj = max(lam_away_base * (1 - context_adj), 0.01)

        return lam_home_adj, lam_away_adj, lam_home_base, lam_away_base

    except Exception as e:
        st.error(f"Poisson prediction error: {e}")
        return None


def dixon_coles_tau(hg, ag, lam_h, lam_a, rho):
    """Dixon-Coles tau adjustment for low-scoring games"""
    if hg == 0 and ag == 0:
        return 1 - (lam_h * lam_a * rho)
    elif hg == 0 and ag == 1:
        return 1 + lam_a * rho
    elif hg == 1 and ag == 0:
        return 1 + lam_h * rho
    elif hg == 1 and ag == 1:
        return 1 - rho
    else:
        return 1.0


def compute_scoreline_probabilities(lam_home, lam_away, max_goals=6, use_dc=False, rho=0.0):
    """
    Compute probability matrix for all scorelines.

    Args:
        lam_home: Home team goal expectation
        lam_away: Away team goal expectation
        max_goals: Maximum goals to consider
        use_dc: Whether to apply Dixon-Coles correction
        rho: Dixon-Coles rho parameter

    Returns:
        Dictionary with probabilities for various markets
    """
    # Generate goal arrays
    hg = np.arange(0, max_goals + 1)
    ag = np.arange(0, max_goals + 1)

    # Base Poisson probabilities
    P = np.outer(
        poisson.pmf(hg, lam_home),
        poisson.pmf(ag, lam_away)
    )

    # Apply Dixon-Coles adjustment if requested
    if use_dc and abs(rho) > 1e-6:
        for i, h in enumerate(hg):
            for j, a in enumerate(ag):
                tau = dixon_coles_tau(h, a, lam_home, lam_away, rho)
                P[i, j] *= tau

    # Normalize to sum to 1
    P /= P.sum()

    # Calculate market probabilities
    total_goals = hg[:, None] + ag[None, :]

    return {
        "P_home": float(np.tril(P, -1).sum()),   # Home wins
        "P_draw": float(np.trace(P)),            # Draws
        "P_away": float(np.triu(P, 1).sum()),    # Away wins
        "P_over_0_5": float(P[total_goals > 0].sum()),
        "P_over_1_5": float(P[total_goals > 1].sum()),
        "P_over_2_5": float(P[total_goals > 2].sum()),
        "P_over_3_5": float(P[total_goals > 3].sum()),
        "P_over_4_5": float(P[total_goals > 4].sum()),
        "P_BTTS": float(P[(hg[:, None] > 0) & (ag[None, :] > 0)].sum()),
        "P_clean_sheet_home": float(P[:, 0].sum()),
        "P_clean_sheet_away": float(P[0, :].sum())
    }


def get_team_form(team, n=5):
    """
    Get last N results for a team.
    Returns: (letters_string, icons_string, points)
    """
    try:
        team_matches = long_df[long_df["team"] == team].sort_values("Date").tail(n)

        letters = []
        icons = []
        total_points = 0

        for _, match in team_matches.iterrows():
            gf = match["goals_for"]
            ga = match["goals_against"]

            if gf > ga:
                letters.append("W")
                icons.append("🟩")
                total_points += 3
            elif gf == ga:
                letters.append("D")
                icons.append("🟨")
                total_points += 1
            else:
                letters.append("L")
                icons.append("🟥")

        return "".join(letters), "".join(icons), total_points

    except Exception as e:
        st.warning(f"Error getting form for {team}: {e}")
        return "", "", 0


def get_team_position(team):
    """Get team's league position if available"""
    try:
        if "Position" in stats.columns:
            team_row = stats[stats["Squad"] == team]
            if not team_row.empty:
                pos = team_row.iloc[0]["Position"]
                if pd.notna(pos):
                    return str(pos)
        return "—"
    except Exception:
        return "—"


def render_data_freshness_banner(metadata_dict):
    """Prominent banner showing how fresh the data is."""
    if not metadata_dict or "update_time" not in metadata_dict:
        st.markdown(
            '<div class="warning-box">'
            '⚠️ Data freshness unknown. Please run <code>weekly_update.py</code> to refresh.'
            '</div>',
            unsafe_allow_html=True
        )
        return

    try:
        update_dt = datetime.fromisoformat(metadata_dict["update_time"])
        if update_dt.tzinfo is None:
            update_dt = update_dt.replace(tzinfo=timezone.utc)
    except Exception:
        st.markdown(
            '<div class="warning-box">'
            '⚠️ Could not parse data update time. Please re-run <code>weekly_update.py</code>.'
            '</div>',
            unsafe_allow_html=True
        )
        return

    age_days = (datetime.now(timezone.utc) - update_dt).days

    if age_days <= 2:
        box_class = "success-box"
        label = "✅ Data is fresh"
    elif age_days <= 7:
        box_class = "info-box"
        label = "ℹ️ Data a few days old"
    else:
        box_class = "warning-box"
        label = "⚠️ Data may be stale"

    st.markdown(
        f'<div class="{box_class}">'
        f'{label}: last update <strong>{update_dt.strftime("%Y-%m-%d %H:%M UTC")}</strong> '
        f'(~{age_days} day(s) ago).'
        '</div>',
        unsafe_allow_html=True
    )

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR - MATCH SELECTION & CONTEXT
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚽ Match Setup")

    # Team selection
    teams = sorted(stats["Squad"].unique())

    if len(teams) == 0:
        st.error("No teams found in stats file!")
        st.stop()

    home_team = st.selectbox("🏠 Home Team", teams, key="home")
    away_team = st.selectbox("✈️  Away Team", teams, key="away")

    st.markdown("---")
    st.subheader("📈 Market Odds (Manual Entry)")

    st.subheader("💰 Bankroll Settings")
    bankroll = st.number_input(
        "Current bankroll (£)",
        min_value=1.0,
        value=1000.0,
        step=50.0
    )

    # Staking controls – fractional Kelly & maximum stake cap
    st.subheader("📏 Staking Controls")

    kelly_fraction_user = st.slider(
        "Kelly fraction (0 = no Kelly, 1 = full Kelly)",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help=(
            "Full Kelly maximises long-run growth but is very volatile.\n"
            "Many professionals use 0.1–0.33 of Kelly to reduce drawdowns."
        ),
    )

    max_stake_pct = st.slider(
        "Maximum stake as % of bankroll",
        min_value=0.0,
        max_value=0.10,
        value=0.03,
        step=0.01,
        help="Hard cap per bet (e.g. 3% of bankroll).",
    )

    st.markdown("---")

    odds_home = st.number_input("Home Win Odds", min_value=1.01, value=2.20, step=0.01)
    odds_draw = st.number_input("Draw Odds", min_value=1.01, value=3.30, step=0.01)
    odds_away = st.number_input("Away Win Odds", min_value=1.01, value=3.10, step=0.01)

    st.markdown("---")

    # Context adjustment mode
    st.subheader("🎯 Context Adjustments")

    st.caption(
        "Adjust predictions based on team news, form, tactics. "
        "Affects Poisson & Dixon-Coles only (Logistic stays baseline)."
    )

    advanced_mode = st.checkbox(
        "Advanced Controls",
        value=False,
        help="Enable granular adjustments for injuries, tactics, motivation"
    )

    if not advanced_mode:
        # Simple mode: single slider
        context_raw = st.slider(
            "Match Context",
            min_value=-3.0,
            max_value=3.0,
            value=0.0,
            step=0.1,
            help=(
                "-3 = Strong away advantage (injuries/suspensions to home team)\n"
                "+3 = Strong home advantage (injuries/suspensions to away team)\n"
                "0 = Neutral (no adjustments)"
            )
        )
        context_adj = context_raw / 10.0  # Scale to approx ±0.3

    else:
        # Advanced mode: multiple sliders
        st.markdown("**Real-life context controls:**")

        att_adj = st.slider(
            "⚔️ Attack",
            -3.0, 3.0, 0.0, 0.1,
            help="Attacking strength difference (injuries to forwards, tactical setup)."
        )

        def_adj = st.slider(
            "🛡️ Defense",
            -3.0, 3.0, 0.0, 0.1,
            help="Defensive solidity difference (injuries to defenders, high line, etc.)."
        )

        morale_adj = st.slider(
            "💪 Morale/Momentum",
            -3.0, 3.0, 0.0, 0.1,
            help="Psychological factors (winning streak, derby motivation, pressure)."
        )

        # Combined adjustment (weighted)
        context_adj = (att_adj + def_adj + 0.5 * morale_adj) / 20.0

    # Display current adjustment
    if abs(context_adj) < 0.01:
        st.success("**Adjustment:** Neutral (±0.00)")
    elif context_adj > 0:
        st.info(f"**Adjustment:** Home advantage ({context_adj:+.3f})")
    else:
        st.warning(f"**Adjustment:** Away advantage ({context_adj:+.3f})")

    with st.expander("❓ How to use context adjustments"):
        st.markdown(
            """
- **0.0** → model uses purely historical data (xG, form, strengths).
- **+1.0 in simple mode** ≈ moderate home boost:
  - Home xG increases by ~10%
  - Away xG decreases by ~10%
- **+3.0** (max) ≈ strong home boost (e.g. away team missing multiple key players).
- **Advanced mode examples:**
  - Home star striker out → Attack **−1.0** for home.
  - Away centre-backs injured → Defense **+1.5** for home.
  - Cup rotation / low motivation → Morale **−1.0** for the rotated side.
Use sliders to map **specific news** (injuries, rotation, motivation) into small, realistic nudges.
            """
        )

    st.markdown("---")

    # Model information
    with st.expander("ℹ️  Model Info"):
        if metadata:
            st.markdown("**Last Updated:**")
            try:
                update_time = datetime.fromisoformat(metadata.get("update_time", ""))
                st.text(update_time.strftime("%Y-%m-%d %H:%M UTC"))
            except Exception:
                st.text("Unknown")

            st.markdown("**Data Range:**")
            st.text(metadata.get("date_range", "Unknown"))

            st.markdown(f"**Teams:** {len(metadata.get('teams', []))}")
            st.markdown(f"**Dixon-Coles ρ:** {metadata.get('rho', rho_hat):.4f}")
        else:
            st.text("Metadata not available")
            st.text(f"Dixon-Coles ρ: {rho_hat:.4f}")

# ═══════════════════════════════════════════════════════════════════
# MAIN AREA - PREDICTIONS
# ═══════════════════════════════════════════════════════════════════

# Validation
if home_team == away_team:
    st.error("### ⚠️  Home and Away teams must be different!")
    st.stop()

# Header
st.markdown('<div class="main-title">Premier League Match Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    'Premier League match insights powered by Logistic, Poisson and Dixon-Coles models.'
    '</div>',
    unsafe_allow_html=True
)

# Data freshness banner
render_data_freshness_banner(metadata)

# Sofascore-style structured match header
st.markdown(
    f'''<div class="headline-card">
        <div class="match-header">
            <div class="match-teams">
                <div class="team-block">
                    <div class="team-name">{home_team}</div>
                    <div class="team-tag">Home</div>
                </div>
                <div class="vs-badge">VS</div>
                <div class="team-block">
                    <div class="team-name">{away_team}</div>
                    <div class="team-tag">Away</div>
                </div>
            </div>
            <div class="match-meta">
                <div class="meta-label">Model Ensemble</div>
                <div class="meta-value">Logistic • Poisson • Dixon-Coles</div>
            </div>
        </div>
    </div>''',
    unsafe_allow_html=True,
)

# ───────────────────────────────────────────────────────────────────
# GENERATE PREDICTIONS
# ───────────────────────────────────────────────────────────────────

with st.spinner("Generating predictions..."):
    # Model 1: Logistic (baseline, no adjustments)
    log_probs = predict_logistic(home_team, away_team)

    if log_probs is None:
        st.error("❌ Could not generate logistic predictions. Check team data.")
        st.stop()

    # Model 2 & 3: Poisson & Dixon-Coles (with adjustments)
    lambdas = get_poisson_lambdas(home_team, away_team, context_adj=context_adj)

    if lambdas is None:
        st.error("❌ Could not generate Poisson predictions. Check team data.")
        st.stop()

    lam_h_adj, lam_a_adj, lam_h_base, lam_a_base = lambdas

    # Poisson probabilities (adjusted & baseline)
    poisson_adj = compute_scoreline_probabilities(lam_h_adj, lam_a_adj, use_dc=False)
    poisson_base = compute_scoreline_probabilities(lam_h_base, lam_a_base, use_dc=False)

    # Dixon-Coles probabilities (adjusted & baseline)
    dc_adj = compute_scoreline_probabilities(lam_h_adj, lam_a_adj, use_dc=True, rho=rho_hat)
    dc_base = compute_scoreline_probabilities(lam_h_base, lam_a_base, use_dc=True, rho=rho_hat)

    # Convert betting odds to implied probabilities (decimal odds -> prob)
    imp_home_raw = 1.0 / odds_home
    imp_draw_raw = 1.0 / odds_draw
    imp_away_raw = 1.0 / odds_away

    overround = imp_home_raw + imp_draw_raw + imp_away_raw

    # Margin-adjusted "fair" probabilities
    fair_home = imp_home_raw / overround
    fair_draw = imp_draw_raw / overround
    fair_away = imp_away_raw / overround

    # Margin-adjusted edge: model - fair
    edge_home = dc_adj["P_home"] - fair_home
    edge_draw = dc_adj["P_draw"] - fair_draw
    edge_away = dc_adj["P_away"] - fair_away

    # Simple sanity check on overround
    if overround < 0.90 or overround > 1.20:
        st.warning(
            f"Odds overround looks unusual ({(overround - 1) * 100:+.2f}%). "
            "Double-check the odds inputs."
        )

    # Proper 3-way Kelly optimisation across Home/Draw/Away
    probs_vector = np.array([dc_adj["P_home"], dc_adj["P_draw"], dc_adj["P_away"]])
    odds_vector = np.array([odds_home, odds_draw, odds_away])
    kelly_home, kelly_draw, kelly_away = kelly_3way(probs_vector, odds_vector)

    # Kelly bet amounts in £ with fractional Kelly and cap
    stake_home = scaled_stake(kelly_home, bankroll, kelly_fraction_user, max_stake_pct)
    stake_draw = scaled_stake(kelly_draw, bankroll, kelly_fraction_user, max_stake_pct)
    stake_away = scaled_stake(kelly_away, bankroll, kelly_fraction_user, max_stake_pct)

# ───────────────────────────────────────────────────────────────────
# HEADLINE METRICS (DIXON-COLES ADJUSTED)
# ───────────────────────────────────────────────────────────────────

st.subheader("🎯 Headline Prediction (Dixon-Coles Adjusted)")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🏠 Home Win", pct(dc_adj["P_home"]))

with col2:
    st.metric("🤝 Draw", pct(dc_adj["P_draw"]))

with col3:
    st.metric("✈️  Away Win", pct(dc_adj["P_away"]))

with col4:
    st.metric("⚽ BTTS", pct(dc_adj["P_BTTS"]))

if abs(context_adj) > 0.01:
    delta = (dc_adj["P_home"] - dc_base["P_home"]) * 100
    st.caption(
        "💡 Your adjustments changed the home win probability by "
        f"**{delta:+.1f} percentage points** vs baseline."
    )

st.caption(
    "These are point estimates from statistical models. Individual matches can deviate "
    "significantly from these probabilities, so treat them as directional rather than certain."
)

st.markdown("---")

# ───────────────────────────────────────────────────────────────────
# THREE MODEL CARDS
# ───────────────────────────────────────────────────────────────────

st.subheader("📊 Model Comparison")

c1, c2, c3 = st.columns(3)

# MODEL 1: LOGISTIC REGRESSION
with c1:
    st.markdown('<div class="model-card">', unsafe_allow_html=True)
    st.markdown("<h4>📈 Logistic Model</h4>", unsafe_allow_html=True)

    st.markdown(
        f'<div class="model-metric-row">'
        f'<span class="model-metric-label">Home Win</span>'
        f'<span class="model-metric-value metric-highlight">{pct(log_probs.get("H", 0.0))}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="model-metric-row">'
        f'<span class="model-metric-label">Draw</span>'
        f'<span class="model-metric-value metric-highlight">{pct(log_probs.get("D", 0.0))}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="model-metric-row">'
        f'<span class="model-metric-label">Away Win</span>'
        f'<span class="model-metric-value metric-highlight">{pct(log_probs.get("A", 0.0))}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title" style="margin-top:12px; font-size:12px;">'
        '🔒 <strong>Baseline prediction</strong> using historical results, '
        'team strength, and 5-game rolling form. '
        'Unaffected by context adjustments.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

# MODEL 2: POISSON GLM
with c2:
    st.markdown('<div class="model-card">', unsafe_allow_html=True)
    st.markdown("<h4>📐 Poisson Model</h4>", unsafe_allow_html=True)

    st.markdown(
        f'<div class="model-metric-row">'
        f'<span class="model-metric-label">xG Home</span>'
        f'<span class="model-metric-value metric-highlight">{lam_h_adj:.2f}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="model-metric-row">'
        f'<span class="model-metric-label">xG Away</span>'
        f'<span class="model-metric-value metric-highlight">{lam_a_adj:.2f}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown("<hr style='margin:8px 0; border-color:#374151;'>", unsafe_allow_html=True)

    st.markdown(
        f'<div class="model-metric-row">'
        f'<span class="model-metric-label">Home Win</span>'
        f'<span class="model-metric-value">{pct(poisson_adj["P_home"])}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="model-metric-row">'
        f'<span class="model-metric-label">Draw</span>'
        f'<span class="model-metric-value">{pct(poisson_adj["P_draw"])}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="model-metric-row">'
        f'<span class="model-metric-label">Away Win</span>'
        f'<span class="model-metric-value">{pct(poisson_adj["P_away"])}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown("<hr style='margin:8px 0; border-color:#374151;'>", unsafe_allow_html=True)

    st.markdown(
        f'<div class="model-metric-row">'
        f'<span class="model-metric-label">Over 1.5</span>'
        f'<span class="model-metric-value">{pct(poisson_adj["P_over_1_5"])}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="model-metric-row">'
        f'<span class="model-metric-label">Over 2.5</span>'
        f'<span class="model-metric-value">{pct(poisson_adj["P_over_2_5"])}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="model-metric-row">'
        f'<span class="model-metric-label">BTTS</span>'
        f'<span class="model-metric-value">{pct(poisson_adj["P_BTTS"])}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title" style="margin-top:12px; font-size:12px;">'
        '⚡ <strong>Goal-based prediction</strong> using expected goals. '
        'Context sliders adjust xG up/down based on team news and form.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

# MODEL 3: DIXON-COLES
with c3:
    st.markdown('<div class="model-card">', unsafe_allow_html=True)
    st.markdown("<h4>🎲 Dixon-Coles</h4>", unsafe_allow_html=True)

    st.markdown(
        f'<div class="model-metric-row">'
        f'<span class="model-metric-label">Home Win</span>'
        f'<span class="model-metric-value metric-highlight">{pct(dc_adj["P_home"])}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="model-metric-row">'
        f'<span class="model-metric-label">Draw</span>'
        f'<span class="model-metric-value metric-highlight">{pct(dc_adj["P_draw"])}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="model-metric-row">'
        f'<span class="model-metric-label">Away Win</span>'
        f'<span class="model-metric-value metric-highlight">{pct(dc_adj["P_away"])}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown("<hr style='margin:8px 0; border-color:#374151;'>", unsafe_allow_html=True)

    st.markdown(
        f'<div class="model-metric-row">'
        f'<span class="model-metric-label">Over 1.5</span>'
        f'<span class="model-metric-value">{pct(dc_adj["P_over_1_5"])}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="model-metric-row">'
        f'<span class="model-metric-label">Over 2.5</span>'
        f'<span class="model-metric-value">{pct(dc_adj["P_over_2_5"])}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="model-metric-row">'
        f'<span class="model-metric-label">Over 3.5</span>'
        f'<span class="model-metric-value">{pct(dc_adj["P_over_3_5"])}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="model-metric-row">'
        f'<span class="model-metric-label">BTTS</span>'
        f'<span class="model-metric-value">{pct(dc_adj["P_BTTS"])}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title" style="margin-top:12px; font-size:12px;">'
        f'🔧 <strong>Enhanced Poisson</strong> with correlation adjustment (ρ={rho_hat:.3f}). '
        'Corrects for dependency in low-scoring matches (0-0, 0-1, 1-0, 1-1).'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ───────────────────────────────────────────────────────────────────
# VALUE EDGE ANALYSIS & KELLY
# ───────────────────────────────────────────────────────────────────

st.subheader("💰 Value Edge vs Market Odds")

st.subheader("📊 Kelly Optimal Stake Sizing")

kelly_df = pd.DataFrame({
    "Outcome": ["Home", "Draw", "Away"],
    "Edge vs fair": [
        f"{edge_home * 100:+.2f}%",
        f"{edge_draw * 100:+.2f}%",
        f"{edge_away * 100:+.2f}%"
    ],
    "Full Kelly % of Bankroll (3-way)": [
        f"{max(0, kelly_home) * 100:.2f}%",
        f"{max(0, kelly_draw) * 100:.2f}%",
        f"{max(0, kelly_away) * 100:.2f}%"
    ],
    f"Applied Kelly % (×{kelly_fraction_user:.2f})": [
        f"{max(0, kelly_home) * kelly_fraction_user * 100:.2f}%",
        f"{max(0, kelly_draw) * kelly_fraction_user * 100:.2f}%",
        f"{max(0, kelly_away) * kelly_fraction_user * 100:.2f}%"
    ],
    "Stake (£)": [
        f"£{stake_home:.2f}",
        f"£{stake_draw:.2f}",
        f"£{stake_away:.2f}"
    ]
})

st.dataframe(kelly_df, use_container_width=True, hide_index=True)

best_stake = max(stake_home, stake_draw, stake_away)
best_index = np.argmax([stake_home, stake_draw, stake_away])
best_side = ["Home", "Draw", "Away"][best_index]

if best_stake > 0:
    st.success(f"Recommended Bet: **{best_side}** — £{best_stake:.2f} (3-way fractional Kelly with caps)")
else:
    st.warning("Kelly suggests NO BET on this match.")

st.markdown(
    '<div class="warning-box">'
    '⚠️ <strong>Kelly risk notice:</strong> Full Kelly is highly aggressive and can lead to '
    'large drawdowns even with a positive edge. This model applies fractional Kelly and a hard '
    'stake cap, but you remain responsible for your overall risk management.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("---")

edge_df = pd.DataFrame({
    "Outcome": ["Home", "Draw", "Away"],
    "Model Probability": [
        pct(dc_adj["P_home"]),
        pct(dc_adj["P_draw"]),
        pct(dc_adj["P_away"])
    ],
    "Market Prob (raw)": [
        pct(imp_home_raw),
        pct(imp_draw_raw),
        pct(imp_away_raw)
    ],
    "Market Prob (fair, margin-adj)": [
        pct(fair_home),
        pct(fair_draw),
        pct(fair_away)
    ],
    "Edge vs fair": [
        f"{edge_home * 100:+.2f}%",
        f"{edge_draw * 100:+.2f}%",
        f"{edge_away * 100:+.2f}%"
    ]
})


def highlight_edge(val):
    if isinstance(val, str) and val.startswith('+'):
        return 'background-color: #D1FAE5; color: #065F46; font-weight: 600;'
    if isinstance(val, str) and val.startswith('-'):
        return 'color: #991B1B;'
    return ''


st.dataframe(
    edge_df.style.applymap(highlight_edge),
    use_container_width=True,
    hide_index=True
)

st.caption(
    f"Bookmaker overround: {(overround - 1) * 100:+.2f}%. "
    "Edges are computed vs margin-adjusted fair probabilities."
)

best_edge = max(edge_home, edge_draw, edge_away)
best_outcome = ["Home", "Draw", "Away"][np.argmax([edge_home, edge_draw, edge_away])]

if best_edge > 0:
    st.success(f"Best value opportunity: **{best_outcome}** ({best_edge * 100:+.2f}%)")
else:
    st.warning("No positive edge — market appears efficient for this match.")

st.markdown("---")

# ───────────────────────────────────────────────────────────────────
# ADDITIONAL MARKETS
# ───────────────────────────────────────────────────────────────────

st.subheader("Additional Markets")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("**Total Goals**")
    st.markdown(f"Over 0.5: {pct(dc_adj['P_over_0_5'])}")
    st.markdown(f"Over 1.5: {pct(dc_adj['P_over_1_5'])}")
    st.markdown(f"Over 2.5: {pct(dc_adj['P_over_2_5'])}")
    st.markdown(f"Over 3.5: {pct(dc_adj['P_over_3_5'])}")
    st.markdown(f"Over 4.5: {pct(dc_adj['P_over_4_5'])}")

with col_b:
    st.markdown("**Both Teams to Score**")
    st.markdown(f"BTTS Yes: {pct(dc_adj['P_BTTS'])}")
    st.markdown(f"BTTS No: {pct(1 - dc_adj['P_BTTS'])}")

with col_c:
    st.markdown("**Clean Sheets**")
    st.markdown(f"{home_team} CS: {pct(dc_adj['P_clean_sheet_home'])}")
    st.markdown(f"{away_team} CS: {pct(dc_adj['P_clean_sheet_away'])}")

st.markdown("---")

# ───────────────────────────────────────────────────────────────────
# TEAM FORM & STATISTICS
# ───────────────────────────────────────────────────────────────────

with st.expander("📋 Team Form & Statistics (Last 5 Matches)", expanded=False):

    col_home, col_away = st.columns(2)

    # Home team stats
    with col_home:
        st.markdown(f"### 🏠 {home_team}")

        home_pos = get_team_position(home_team)
        st.markdown(f"**League Position:** {home_pos}")

        home_form_letters, home_form_icons, home_form_points = get_team_form(home_team)
        st.markdown(f"**Form (oldest → newest):** {home_form_icons}")
        st.markdown(f"**Form String:** {home_form_letters}")
        st.markdown(f"**Points (last 5):** {home_form_points}")

        st.markdown("---")

        st.markdown("**Rolling Metrics (Last 5 Games):**")

        home_xg_for = get_latest_team_stat(home_team, "rolling_xg_for")
        home_xg_ag = get_latest_team_stat(home_team, "rolling_xg_against")
        home_gd = get_latest_team_stat(home_team, "rolling_GD")
        home_fin = get_latest_team_stat(home_team, "rolling_finishing_overperf")
        home_def = get_latest_team_stat(home_team, "rolling_def_overperf")

        if home_xg_for is None:
            st.markdown("- **xG For:** n/a (insufficient recent data)")
        else:
            st.markdown(f"- **xG For:** {home_xg_for:.2f}")

        if home_xg_ag is None:
            st.markdown("- **xG Against:** n/a (insufficient recent data)")
        else:
            st.markdown(f"- **xG Against:** {home_xg_ag:.2f}")

        if home_gd is None:
            st.markdown("- **Goal Difference:** n/a")
        else:
            st.markdown(f"- **Goal Difference:** {home_gd:+.0f}")

        if home_fin is None:
            st.markdown("- **Finishing (G - xG):** n/a")
        else:
            st.markdown(f"- **Finishing (G - xG):** {home_fin:+.2f}")

        if home_def is None:
            st.markdown("- **Defence (xGA - GA):** n/a")
        else:
            st.markdown(f"- **Defence (xGA - GA):** {home_def:+.2f}")

        if home_fin is not None:
            if home_fin > 1:
                st.success("💚 Overperforming xG (good finishing streak)")
            elif home_fin < -1:
                st.warning("⚠️ Underperforming xG (poor finishing streak)")

        if home_def is not None:
            if home_def > 1:
                st.success("💚 Overperforming defence (conceding less than xGA)")
            elif home_def < -1:
                st.warning("⚠️ Underperforming defence (conceding more than xGA)")

    # Away team stats
    with col_away:
        st.markdown(f"### ✈️  {away_team}")

        away_pos = get_team_position(away_team)
        st.markdown(f"**League Position:** {away_pos}")

        away_form_letters, away_form_icons, away_form_points = get_team_form(away_team)
        st.markdown(f"**Form (oldest → newest):** {away_form_icons}")
        st.markdown(f"**Form String:** {away_form_letters}")
        st.markdown(f"**Points (last 5):** {away_form_points}")

        st.markdown("---")

        st.markdown("**Rolling Metrics (Last 5 Games):**")

        away_xg_for = get_latest_team_stat(away_team, "rolling_xg_for")
        away_xg_ag = get_latest_team_stat(away_team, "rolling_xg_against")
        away_gd = get_latest_team_stat(away_team, "rolling_GD")
        away_fin = get_latest_team_stat(away_team, "rolling_finishing_overperf")
        away_def = get_latest_team_stat(away_team, "rolling_def_overperf")

        if away_xg_for is None:
            st.markdown("- **xG For:** n/a (insufficient recent data)")
        else:
            st.markdown(f"- **xG For:** {away_xg_for:.2f}")

        if away_xg_ag is None:
            st.markdown("- **xG Against:** n/a (insufficient recent data)")
        else:
            st.markdown(f"- **xG Against:** {away_xg_ag:.2f}")

        if away_gd is None:
            st.markdown("- **Goal Difference:** n/a")
        else:
            st.markdown(f"- **Goal Difference:** {away_gd:+.0f}")

        if away_fin is None:
            st.markdown("- **Finishing (G - xG):** n/a")
        else:
            st.markdown(f"- **Finishing (G - xG):** {away_fin:+.2f}")

        if away_def is None:
            st.markdown("- **Defence (xGA - GA):** n/a")
        else:
            st.markdown(f"- **Defence (xGA - GA):** {away_def:+.2f}")

        if away_fin is not None:
            if away_fin > 1:
                st.success("💚 Overperforming xG (good finishing streak)")
            elif away_fin < -1:
                st.warning("⚠️ Underperforming xG (poor finishing streak)")

        if away_def is not None:
            if away_def > 1:
                st.success("💚 Overperforming defence (conceding less than xGA)")
            elif away_def < -1:
                st.warning("⚠️ Underperforming defence (conceding more than xGA)")

    # Comparison table
    st.markdown("---")
    st.markdown("### 📊 Head-to-Head Comparison")

    def fmt_metric(val, fmt="{:.2f}"):
        if val is None:
            return "n/a"
        try:
            return fmt.format(val)
        except Exception:
            return "n/a"

    comparison_df = pd.DataFrame({
        "Metric": [
            "xG For (last 5)",
            "xG Against (last 5)",
            "Goal Difference (last 5)",
            "Points (last 5)",
            "Finishing Over/Under",
            "Defence Over/Under"
        ],
        home_team: [
            fmt_metric(home_xg_for),
            fmt_metric(home_xg_ag),
            fmt_metric(home_gd, "{:+.0f}"),
            f"{home_form_points}",
            fmt_metric(home_fin, "{:+.2f}"),
            fmt_metric(home_def, "{:+.2f}")
        ],
        away_team: [
            fmt_metric(away_xg_for),
            fmt_metric(away_xg_ag),
            fmt_metric(away_gd, "{:+.0f}"),
            f"{away_form_points}",
            fmt_metric(away_fin, "{:+.2f}"),
            fmt_metric(away_def, "{:+.2f}")
        ]
    })

    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.caption(
        "💡 **Interpretation:**\n"
        "- **Finishing > 0:** Scoring more goals than xG suggests (good finishing/luck)\n"
        "- **Finishing < 0:** Scoring fewer goals than xG suggests (poor finishing/bad luck)\n"
        "- **Defence > 0:** Conceding fewer goals than xGA suggests (good defending/goalkeeping)\n"
        "- **Defence < 0:** Conceding more goals than xGA suggests (poor defending)"
    )

# ───────────────────────────────────────────────────────────────────
# MODEL AGREEMENT ANALYSIS
# ───────────────────────────────────────────────────────────────────

with st.expander("🔍 Model Agreement Analysis", expanded=False):

    st.markdown("### Model Consensus")

    # Calculate which outcome each model favors
    log_favorite = max(log_probs, key=log_probs.get)
    poisson_favorite = max(
        ["H", "D", "A"],
        key=lambda x: poisson_adj["P_home"] if x == "H"
        else (poisson_adj["P_draw"] if x == "D" else poisson_adj["P_away"])
    )
    dc_favorite = max(
        ["H", "D", "A"],
        key=lambda x: dc_adj["P_home"] if x == "H"
        else (dc_adj["P_draw"] if x == "D" else dc_adj["P_away"])
    )

    outcome_names = {"H": "Home Win", "D": "Draw", "A": "Away Win"}

    st.markdown(
        f"- **Logistic Model:** {outcome_names[log_favorite]} "
        f"({pct(log_probs.get(log_favorite, 0))})"
    )
    st.markdown(
        f"- **Poisson Model:** {outcome_names[poisson_favorite]}"
    )
    st.markdown(
        f"- **Dixon-Coles:** {outcome_names[dc_favorite]} "
        f"({pct(dc_adj['P_home'] if dc_favorite == 'H' else (dc_adj['P_draw'] if dc_favorite == 'D' else dc_adj['P_away']))})"
    )

    # Check agreement
    all_agree = (log_favorite == poisson_favorite == dc_favorite)

    if all_agree:
        st.markdown(
            '<div class="success-box">'
            f'✅ <strong>Strong Consensus:</strong> All three models agree on '
            f'<strong>{outcome_names[log_favorite]}</strong>. '
            'This suggests higher confidence in the prediction.'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="warning-box">'
            '⚠️ <strong>Model Disagreement:</strong> Models have different favorites. '
            'Consider this uncertainty when making decisions.'
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Probability comparison table
    st.markdown("### Probability Comparison Across Models")

    prob_comparison = pd.DataFrame({
        "Outcome": ["Home Win", "Draw", "Away Win"],
        "Logistic": [
            pct(log_probs.get("H", 0)),
            pct(log_probs.get("D", 0)),
            pct(log_probs.get("A", 0))
        ],
        "Poisson": [
            pct(poisson_adj["P_home"]),
            pct(poisson_adj["P_draw"]),
            pct(poisson_adj["P_away"])
        ],
        "Dixon-Coles": [
            pct(dc_adj["P_home"]),
            pct(dc_adj["P_draw"]),
            pct(dc_adj["P_away"])
        ]
    })

    st.dataframe(prob_comparison, use_container_width=True, hide_index=True)

    st.caption(
        "💡 Large discrepancies between models indicate uncertainty. "
        "The logistic model is based purely on match outcomes, while "
        "Poisson/Dixon-Coles are based on goal expectations."
    )

# ───────────────────────────────────────────────────────────────────
# CONTEXT IMPACT ANALYSIS
# ───────────────────────────────────────────────────────────────────

if abs(context_adj) > 0.01:
    with st.expander("⚙️ Context Adjustment Impact", expanded=False):

        st.markdown("### How Adjustments Changed Predictions")

        st.markdown(f"**Adjustment Factor:** {context_adj:+.3f}")

        # xG changes
        st.markdown("**Expected Goals Impact:**")
        st.markdown(
            f"- Home xG: {lam_h_base:.2f} → {lam_h_adj:.2f} "
            f"({lam_h_adj - lam_h_base:+.2f})"
        )
        st.markdown(
            f"- Away xG: {lam_a_base:.2f} → {lam_a_adj:.2f} "
            f"({lam_a_adj - lam_a_base:+.2f})"
        )

        st.markdown("---")

        # Probability changes
        st.markdown("**Match Outcome Impact (Dixon-Coles):**")

        delta_h = (dc_adj["P_home"] - dc_base["P_home"]) * 100
        delta_d = (dc_adj["P_draw"] - dc_base["P_draw"]) * 100
        delta_a = (dc_adj["P_away"] - dc_base["P_away"]) * 100

        st.markdown(
            f"- Home Win: {pct(dc_base['P_home'])} → {pct(dc_adj['P_home'])} "
            f"({delta_h:+.1f}pp)"
        )
        st.markdown(
            f"- Draw: {pct(dc_base['P_draw'])} → {pct(dc_adj['P_draw'])} "
            f"({delta_d:+.1f}pp)"
        )
        st.markdown(
            f"- Away Win: {pct(dc_base['P_away'])} → {pct(dc_adj['P_away'])} "
            f"({delta_a:+.1f}pp)"
        )

        st.caption("pp = percentage points")

        st.markdown("---")

        # Goals market impact
        st.markdown("**Goals Market Impact:**")

        delta_btts = (dc_adj["P_BTTS"] - dc_base["P_BTTS"]) * 100
        delta_o25 = (dc_adj["P_over_2_5"] - dc_base["P_over_2_5"]) * 100

        st.markdown(
            f"- BTTS: {pct(dc_base['P_BTTS'])} → {pct(dc_adj['P_BTTS'])} "
            f"({delta_btts:+.1f}pp)"
        )
        st.markdown(
            f"- Over 2.5: {pct(dc_base['P_over_2_5'])} → {pct(dc_adj['P_over_2_5'])} "
            f"({delta_o25:+.1f}pp)"
        )

# ───────────────────────────────────────────────────────────────────
# FOOTER & DISCLAIMERS
# ───────────────────────────────────────────────────────────────────

st.markdown("---")

st.markdown(
    '<div class="footer-note">'
    '⚠️ <strong>Important Disclaimer:</strong><br>'
    'These predictions are statistical models based on historical data and should not be used as the sole basis for betting or decision-making. '
    'Football matches are inherently unpredictable, and many factors (injuries, referee decisions, weather, motivation) cannot be fully captured by models.<br><br>'
    '📊 <strong>Model Updates:</strong> Models are retrained weekly with latest match data to maintain accuracy.<br>'
    '🔧 <strong>Context Adjustments:</strong> Use the sidebar sliders to incorporate team news and match-specific factors.<br>'
    '📈 <strong>Model Types:</strong> Logistic (outcome-based), Poisson (goal-based), Dixon-Coles (correlation-adjusted).'
    '</div>',
    unsafe_allow_html=True
)
