"""
Premier League Match Predictor - Enhanced Version
-------------------------------------------------
Enhanced with performance tracking, confidence metrics, and improved betting analysis.

New Features:
- Model performance dashboard
- Prediction confidence indicators
- Enhanced value betting analysis
- Kelly criterion validation
- Export functionality
- Risk management controls
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy.stats import poisson
from pathlib import Path
from datetime import datetime
import json

def kelly_fraction(prob, odds):
    b = odds - 1
    q = 1 - prob
    if b <= 0:
        return 0
    return (prob * (b + 1) - 1) / b


def get_model_update_version():
    """Return update timestamp from metadata.json to bust Streamlit cache."""
    try:
        if Path("metadata.json").exists():
            with open("metadata.json", "r") as f:
                metadata = json.load(f)
                return metadata.get("update_time", "0")
    except:
        pass
    return str(datetime.utcnow())


def calculate_prediction_confidence(p_home, p_draw, p_away):
    """
    Calculate confidence based on entropy and margin between outcomes.
    Returns: confidence_score (0-100), confidence_label, color
    """
    # Shannon entropy (normalized)
    probs = [p_home, p_draw, p_away]
    entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probs)
    max_entropy = np.log2(3)  # Maximum for 3 outcomes
    normalized_entropy = entropy / max_entropy
    
    # Confidence: inverse of entropy
    confidence = (1 - normalized_entropy) * 100
    
    # Margin between top two outcomes
    sorted_probs = sorted(probs, reverse=True)
    margin = (sorted_probs[0] - sorted_probs[1]) * 100
    
    # Combined confidence score (60% margin, 40% entropy-based)
    combined_confidence = (0.6 * margin + 0.4 * confidence)
    
    if combined_confidence >= 25:
        label = "High"
        color = "#10B981"
    elif combined_confidence >= 15:
        label = "Medium"
        color = "#F59E0B"
    else:
        label = "Low"
        color = "#EF4444"
    
    return combined_confidence, label, color


# ═══════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="KB Premier League Match Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════
# CUSTOM STYLING
# ═══════════════════════════════════════════════════════════════

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
        .match-meta {
            align-items: flex-start;
        }
        .team-name {
            font-size: 18px;
        }
    }

    @media (max-width: 480px) {
        .team-name {
            font-size: 16px;
        }
        .vs-badge {
            font-size: 11px;
            padding: 3px 7px;
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

# ═══════════════════════════════════════════════════════════════
# LOAD MODELS & DATA WITH VALIDATION
# ═══════════════════════════════════════════════════════════════

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
        
        if Path("feature_cols.pkl").exists():
            with open("feature_cols.pkl", "rb") as f:
                feature_cols = pickle.load(f)
        else:
            feature_cols = [
                "strength_diff", "defense_diff",
                "rolling_points_diff", "rolling_xG_diff", "rolling_xGA_diff",
                "rolling_GD_diff", "finishing_overperf_diff", "def_overperf_diff"
            ]
        
        metadata = {}
        if Path("metadata.json").exists():
            with open("metadata.json", "r") as f:
                metadata = json.load(f)
        
        if stats.empty or long_df.empty:
            st.error("❌ Loaded data is empty. Please regenerate models.")
            st.stop()
        
        stats["Squad"] = stats["Squad"].astype(str).str.strip()
        long_df["team"] = long_df["team"].astype(str).str.strip()
        
        required_stats_cols = ["Squad", "Home_xG", "Home_xGA", "Home_MP", 
                               "Away_xG", "Away_xGA", "Away_MP"]
        missing_cols = [col for col in required_stats_cols if col not in stats.columns]
        
        if missing_cols:
            st.error(f"❌ Stats file missing required columns: {missing_cols}")
            st.stop()
        
        required_long_cols = ["team", "Date", "rolling_points", "rolling_xg_for", 
                              "rolling_xg_against", "rolling_GD", 
                              "rolling_finishing_overperf", "rolling_def_overperf"]
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
        st.error(f"### ❌ Error Loading Models")
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

# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def pct(x, decimals=1):
    """Format as percentage"""
    return f"{x * 100:.{decimals}f}%"


def safe_division(numerator, denominator, default=0.0):
    """Safe division with fallback"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default


def get_latest_team_stat(team, column, default=None):
    """Safely get latest stat for a team from long_df"""
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
    
    home_strength = get_team_strength_metrics(home_team)
    away_strength = get_team_strength_metrics(away_team)
    
    if home_strength is None or away_strength is None:
        return None
    
    home_rolling = {
        "points": get_latest_team_stat(home_team, "rolling_points", 0.0),
        "xg_for": get_latest_team_stat(home_team, "rolling_xg_for", 0.0),
        "xg_against": get_latest_team_stat(home_team, "rolling_xg_against", 0.0),
        "GD": get_latest_team_stat(home_team, "rolling_GD", 0.0),
        "finishing_overperf": get_latest_team_stat(home_team, "rolling_finishing_overperf", 0.0),
        "def_overperf": get_latest_team_stat(home_team, "rolling_def_overperf", 0.0)
    }
    
    away_rolling = {
        "points": get_latest_team_stat(away_team, "rolling_points", 0.0),
        "xg_for": get_latest_team_stat(away_team, "rolling_xg_for", 0.0),
        "xg_against": get_latest_team_stat(away_team, "rolling_xg_against", 0.0),
        "GD": get_latest_team_stat(away_team, "rolling_GD", 0.0),
        "finishing_overperf": get_latest_team_stat(away_team, "rolling_finishing_overperf", 0.0),
        "def_overperf": get_latest_team_stat(away_team, "rolling_def_overperf", 0.0)
    }
    
    # Check for missing critical data
    critical_values = [
        home_rolling["points"], home_rolling["xg_for"],
        away_rolling["points"], away_rolling["xg_for"]
    ]
    
    if any(v is None for v in critical_values):
        st.warning(f"⚠️ Incomplete recent form data for {home_team} or {away_team}. Using fallback values.")
    
    features = {
        "strength_diff": home_strength["att_home"] - away_strength["att_away"],
        "defense_diff": away_strength["def_away"] - home_strength["def_home"],
        "rolling_points_diff": home_rolling["points"] - away_rolling["points"],
        "rolling_xG_diff": home_rolling["xg_for"] - away_rolling["xg_for"],
        "rolling_xGA_diff": home_rolling["xg_against"] - away_rolling["xg_against"],
        "rolling_GD_diff": home_rolling["GD"] - away_rolling["GD"],
        "finishing_overperf_diff": home_rolling["finishing_overperf"] - away_rolling["finishing_overperf"],
        "def_overperf_diff": home_rolling["def_overperf"] - away_rolling["def_overperf"]
    }
    
    try:
        return pd.DataFrame([features])[feature_cols]
    except KeyError as e:
        st.error(f"❌ Feature mismatch: {e}. Model may need retraining.")
        return None


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
    """
    try:
        pred_df = pd.DataFrame({
            "team": [home_team, away_team],
            "opponent": [away_team, home_team],
            "is_home": [1, 0]
        })
        
        predictions = poisson_model.predict(pred_df)
        lam_home_base = float(predictions.iloc[0])
        lam_away_base = float(predictions.iloc[1])
        
        lam_home_adj = max(lam_home_base * (1 + context_adj), 0.01)
        lam_away_adj = max(lam_away_base * (1 - context_adj), 0.01)
        
        return lam_home_adj, lam_away_adj, lam_home_base, lam_away_base
    
    except Exception as e:
        st.error(f"Poisson prediction error: {e}")
        return None


def compute_scoreline_probabilities(lam_home, lam_away, max_goals=6, use_dc=False, rho=0.0):
    """
    Compute probability matrix for all scorelines.
    """
    
    hg = np.arange(0, max_goals + 1)
    ag = np.arange(0, max_goals + 1)
    
    P = np.outer(
        poisson.pmf(hg, lam_home),
        poisson.pmf(ag, lam_away)
    )
    
    if use_dc and abs(rho) > 1e-6:
        for i, h in enumerate(hg):
            for j, a in enumerate(ag):
                tau = dixon_coles_tau(h, a, lam_home, lam_away, rho)
                P[i, j] *= tau
    
    P /= P.sum()
    
    total_goals = hg[:, None] + ag[None, :]
    
    return {
        "P_home": float(np.tril(P, -1).sum()),
        "P_draw": float(np.trace(P)),
        "P_away": float(np.triu(P, 1).sum()),
        "P_over_0_5": float(P[total_goals > 0].sum()),
        "P_over_1_5": float(P[total_goals > 1].sum()),
        "P_over_2_5": float(P[total_goals > 2].sum()),
        "P_over_3_5": float(P[total_goals > 3].sum()),
        "P_over_4_5": float(P[total_goals > 4].sum()),
        "P_BTTS": float(P[(hg[:, None] > 0) & (ag[None, :] > 0)].sum()),
        "P_clean_sheet_home": float(P[:, 0].sum()),
        "P_clean_sheet_away": float(P[0, :].sum())
    }


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
    except:
        return "—"


# ═══════════════════════════════════════════════════════════════
# SIDEBAR - MATCH SELECTION & CONTEXT
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚽ Match Setup")
    
    teams = sorted(stats["Squad"].unique())
    
    if len(teams) == 0:
        st.error("No teams found in stats file!")
        st.stop()
    
    home_team = st.selectbox("🏠 Home Team", teams, key="home")
    away_team = st.selectbox("✈️ Away Team", teams, key="away")
    
    # Data freshness indicator
    if metadata:
        try:
            update_time = datetime.fromisoformat(metadata.get("update_time", ""))
            hours_old = (datetime.utcnow() - update_time).total_seconds() / 3600
            
            if hours_old < 24:
                st.success(f"✅ Data updated {hours_old:.1f} hours ago")
            elif hours_old < 72:
                st.warning(f"⚠️ Data is {hours_old/24:.1f} days old")
            else:
                st.error(f"❌ Data is {hours_old/24:.1f} days old - predictions may be unreliable")
        except:
            st.warning("⚠️ Unable to verify data freshness")
    
    st.markdown("---")
    st.subheader("📈 Market Odds (Manual Entry)")

    odds_home = st.number_input("Home Win Odds", min_value=1.01, value=2.20, step=0.01)
    odds_draw = st.number_input("Draw Odds", min_value=1.01, value=3.30, step=0.01)
    odds_away = st.number_input("Away Win Odds", min_value=1.01, value=3.10, step=0.01)
    
    st.markdown("---")
    st.subheader("💰 Bankroll Settings")
    
    bankroll = st.number_input("Current bankroll (£)", min_value=1.0, value=1000.0, step=50.0)
    
    kelly_fraction_multiplier = st.slider(
        "Kelly Fraction (risk adjustment)",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Quarter Kelly (0.25) recommended for most users. Full Kelly (1.0) is very aggressive."
    )
    
    if kelly_fraction_multiplier >= 0.5:
        st.warning("⚠️ Higher Kelly fractions increase variance significantly.")