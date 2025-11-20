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
from pathlib import Path
from datetime import datetime
import json

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
    body, .main { color: #F3F4F6; }
    .main-title { font-size: 36px; font-weight: 800; margin-bottom: 4px; color: #FFFFFF; }
    .sub-title { font-size: 16px; color: #D1D5DB; margin-bottom: 20px; }
    .model-card { border-radius: 12px; padding: 20px; background-color: #111827;
                  color: #F9FAFB; border: 1px solid #374151; box-shadow: 0 2px 6px rgba(0,0,0,0.4); }
    .model-card h4 { font-size: 20px; font-weight: 700; color:#FFFFFF; border-bottom:2px solid #4B5563;
                     padding-bottom:8px; margin-bottom:14px; }
    .model-metric-row { display:flex; justify-content:space-between; font-size:16px; margin:8px 0; }
    .model-metric-label { color:#E5E7EB; }
    .model-metric-value { font-weight:700; color:#FFFFFF; }
    .metric-highlight { background-color:#1E293B; padding:3px 10px; border-radius:6px; font-size:16px; }
    .info-box { background-color:#1E3A8A; border-left:4px solid #3B82F6; padding:12px; border-radius:6px;
                margin:10px 0; color:#F3F4F6; font-size:15px; }
    .warning-box { background-color:#78350F; border-left:4px solid #FBBF24; padding:12px; border-radius:6px;
                   margin:10px 0; color:#FEF3C7; font-size:15px; }
    .success-box { background-color:#065F46; border-left:4px solid #34D399; padding:12px; border-radius:6px;
                   margin:10px 0; color:#D1FAE5; font-size:15px; }
    .footer-note { margin-top:30px; font-size:13px; color:#D1D5DB; text-align:center;
                   padding:16px; border-top:1px solid #4B5563; line-height:1.6; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# LOAD MODELS & DATA WITH VALIDATION
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
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
        
        # Load feature columns if available (ensures correct order)
        if Path("feature_cols.pkl").exists():
            with open("feature_cols.pkl", "rb") as f:
                feature_cols = pickle.load(f)
        else:
            # Fallback to default order
            feature_cols = [
                "strength_diff", "defense_diff",
                "rolling_points_diff", "rolling_xG_diff", "rolling_xGA_diff",
                "rolling_GD_diff", "finishing_overperf_diff", "def_overperf_diff"
            ]
        
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
        required_stats_cols = ["Squad", "Home_xG", "Home_xGA", "Home_MP", 
                               "Away_xG", "Away_xGA", "Away_MP"]
        missing_cols = [col for col in required_stats_cols if col not in stats.columns]
        
        if missing_cols:
            st.error(f"❌ Stats file missing required columns: {missing_cols}")
            st.stop()
        
        # Check required columns in long_df
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
models = load_models()

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
    except:
        return default


def get_latest_team_stat(team, column, default=0.0):
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
    
    # Build features matching training exactly
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
        context_adj: Adjustment factor (-1 to +1)
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
        "P_home": float(np.tril(P, -1).sum()),  # Home wins
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
        context_adj = context_raw / 10.0  # Scale to ±0.3
        
    else:
        # Advanced mode: multiple sliders
        st.markdown("**Real life context Controls:**")
        
        att_adj = st.slider(
            "⚔️ Attack",
            -3.0, 3.0, 0.0, 0.1,
            help="Attacking strength difference (injuries to forwards, tactics)"
        )
        
        def_adj = st.slider(
            "🛡️ Defense", 
            -3.0, 3.0, 0.0, 0.1,
            help="Defensive solidity difference (injuries to defenders, high line tactics)"
        )
        
        morale_adj = st.slider(
            "💪 Morale/Momentum",
            -3.0, 3.0, 0.0, 0.1,
            help="Psychological factors (winning streak, derby motivation, pressure)"
        )
        
        # Combined adjustment (weighted)
        context_adj = (att_adj + def_adj + 0.5 * morale_adj) / 20.0
    
    # Display current adjustment
    if abs(context_adj) < 0.01:
        st.success(f"**Adjustment:** Neutral (±0.00)")
    elif context_adj > 0:
        st.info(f"**Adjustment:** Home advantage ({context_adj:+.3f})")
    else:
        st.warning(f"**Adjustment:** Away advantage ({context_adj:+.3f})")
    
    st.markdown("---")
    
    # Model information
    with st.expander("ℹ️  Model Info"):
        if metadata:
            st.markdown("**Last Updated:**")
            try:
                update_time = datetime.fromisoformat(metadata.get("update_time", ""))
                st.text(update_time.strftime("%Y-%m-%d %H:%M UTC"))
            except:
                st.text("Unknown")
            
            st.markdown(f"**Data Range:**")
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
    'Three model ensemble: Logistic Regression • Poisson GLM • Dixon-Coles'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(f"## {home_team} vs {away_team}")
st.markdown("---")

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
        f"💡 Your adjustments changed the home win probability by "
        f"**{delta:+.1f} percentage points** vs baseline."
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
        
        pos = get_team_position(home_team)
        st.markdown(f"**League Position:** {pos}")
        
        form_letters, form_icons, form_points = get_team_form(home_team)
        st.markdown(f"**Form (oldest → newest):** {form_icons}")
        st.markdown(f"**Form String:** {form_letters}")
        st.markdown(f"**Points (last 5):** {form_points}")
        
        st.markdown("---")
        
        st.markdown("**Rolling Metrics (Last 5 Games):**")
        
        home_xg_for = get_latest_team_stat(home_team, "rolling_xg_for")
        home_xg_ag = get_latest_team_stat(home_team, "rolling_xg_against")
        home_gd = get_latest_team_stat(home_team, "rolling_GD")
        home_fin = get_latest_team_stat(home_team, "rolling_finishing_overperf")
        home_def = get_latest_team_stat(home_team, "rolling_def_overperf")
        
        st.markdown(f"- **xG For:** {home_xg_for:.2f}")
        st.markdown(f"- **xG Against:** {home_xg_ag:.2f}")
        st.markdown(f"- **Goal Difference:** {home_gd:+.0f}")
        st.markdown(f"- **Finishing (G - xG):** {home_fin:+.2f}")
        st.markdown(f"- **Defence (xGA - GA):** {home_def:+.2f}")
        
        if home_fin > 1:
            st.success("💚 Overperforming xG (good finishing streak)")
        elif home_fin < -1:
            st.warning("⚠️ Underperforming xG (poor finishing streak)")
        
        if home_def > 1:
            st.success("💚 Overperforming defence (conceding less than xGA)")
        elif home_def < -1:
            st.warning("⚠️ Underperforming defence (conceding more than xGA)")
    
    # Away team stats
    with col_away:
        st.markdown(f"### ✈️  {away_team}")
        
        pos = get_team_position(away_team)
        st.markdown(f"**League Position:** {pos}")
        
        form_letters, form_icons, form_points = get_team_form(away_team)
        st.markdown(f"**Form (oldest → newest):** {form_icons}")
        st.markdown(f"**Form String:** {form_letters}")
        st.markdown(f"**Points (last 5):** {form_points}")
        
        st.markdown("---")
        
        st.markdown("**Rolling Metrics (Last 5 Games):**")
        
        away_xg_for = get_latest_team_stat(away_team, "rolling_xg_for")
        away_xg_ag = get_latest_team_stat(away_team, "rolling_xg_against")
        away_gd = get_latest_team_stat(away_team, "rolling_GD")
        away_fin = get_latest_team_stat(away_team, "rolling_finishing_overperf")
        away_def = get_latest_team_stat(away_team, "rolling_def_overperf")
        
        st.markdown(f"- **xG For:** {away_xg_for:.2f}")
        st.markdown(f"- **xG Against:** {away_xg_ag:.2f}")
        st.markdown(f"- **Goal Difference:** {away_gd:+.0f}")
        st.markdown(f"- **Finishing (G - xG):** {away_fin:+.2f}")
        st.markdown(f"- **Defence (xGA - GA):** {away_def:+.2f}")
        
        if away_fin > 1:
            st.success("💚 Overperforming xG (good finishing streak)")
        elif away_fin < -1:
            st.warning("⚠️ Underperforming xG (poor finishing streak)")
        
        if away_def > 1:
            st.success("💚 Overperforming defence (conceding less than xGA)")
        elif away_def < -1:
            st.warning("⚠️ Underperforming defence (conceding more than xGA)")
    
    # Comparison table
    st.markdown("---")
    st.markdown("### 📊 Head-to-Head Comparison")
    
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
            f"{home_xg_for:.2f}",
            f"{home_xg_ag:.2f}",
            f"{home_gd:+.0f}",
            f"{form_points}",
            f"{home_fin:+.2f}",
            f"{home_def:+.2f}"
        ],
        away_team: [
            f"{away_xg_for:.2f}",
            f"{away_xg_ag:.2f}",
            f"{away_gd:+.0f}",
            f"{form_points}",
            f"{away_fin:+.2f}",
            f"{away_def:+.2f}"
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
    poisson_favorite = max(["H", "D", "A"], 
                          key=lambda x: poisson_adj["P_home"] if x == "H" 
                                       else (poisson_adj["P_draw"] if x == "D" 
                                             else poisson_adj["P_away"]))
    dc_favorite = max(["H", "D", "A"],
                     key=lambda x: dc_adj["P_home"] if x == "H"
                                  else (dc_adj["P_draw"] if x == "D"
                                        else dc_adj["P_away"]))
    
    outcome_names = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
    
    st.markdown(f"- **Logistic Model:** {outcome_names[log_favorite]} ({pct(log_probs.get(log_favorite, 0))})")
    st.markdown(f"- **Poisson Model:** {outcome_names[poisson_favorite]}")
    st.markdown(f"- **Dixon-Coles:** {outcome_names[dc_favorite]} ({pct(dc_adj['P_home'] if dc_favorite == 'H' else (dc_adj['P_draw'] if dc_favorite == 'D' else dc_adj['P_away']))})")
    
    # Check agreement
    all_agree = (log_favorite == poisson_favorite == dc_favorite)
    
    if all_agree:
        st.markdown(
            '<div class="success-box">'
            f'✅ <strong>Strong Consensus:</strong> All three models agree on <strong>{outcome_names[log_favorite]}</strong>. '
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
        st.markdown(f"- Home xG: {lam_h_base:.2f} → {lam_h_adj:.2f} ({lam_h_adj - lam_h_base:+.2f})")
        st.markdown(f"- Away xG: {lam_a_base:.2f} → {lam_a_adj:.2f} ({lam_a_adj - lam_a_base:+.2f})")
        
        st.markdown("---")
        
        # Probability changes
        st.markdown("**Match Outcome Impact (Dixon-Coles):**")
        
        delta_h = (dc_adj["P_home"] - dc_base["P_home"]) * 100
        delta_d = (dc_adj["P_draw"] - dc_base["P_draw"]) * 100
        delta_a = (dc_adj["P_away"] - dc_base["P_away"]) * 100
        
        st.markdown(f"- Home Win: {pct(dc_base['P_home'])} → {pct(dc_adj['P_home'])} ({delta_h:+.1f}pp)")
        st.markdown(f"- Draw: {pct(dc_base['P_draw'])} → {pct(dc_adj['P_draw'])} ({delta_d:+.1f}pp)")
        st.markdown(f"- Away Win: {pct(dc_base['P_away'])} → {pct(dc_adj['P_away'])} ({delta_a:+.1f}pp)")
        
        st.caption("pp = percentage points")
        
        st.markdown("---")
        
        # Goals market impact
        st.markdown("**Goals Market Impact:**")
        
        delta_btts = (dc_adj["P_BTTS"] - dc_base["P_BTTS"]) * 100
        delta_o25 = (dc_adj["P_over_2_5"] - dc_base["P_over_2_5"]) * 100
        
        st.markdown(f"- BTTS: {pct(dc_base['P_BTTS'])} → {pct(dc_adj['P_BTTS'])} ({delta_btts:+.1f}pp)")
        st.markdown(f"- Over 2.5: {pct(dc_base['P_over_2_5'])} → {pct(dc_adj['P_over_2_5'])} ({delta_o25:+.1f}pp)")

# ───────────────────────────────────────────────────────────────────
# FOOTER & DISCLAIMERS
# ───────────────────────────────────────────────────────────────────

st.markdown("---")

st.markdown(
    '<div class="footer-note">'
    '⚠️ <strong>Important Disclaimer:</strong><br>'
    'These predictions are statistical models based on historical data and should not be used as the sole basis for betting or decision-making. '
    'As you know, football matches are inherently unpredictable, and many factors (injuries, referee decisions, weather, motivation) cannot be fully captured by models.<br><br>'
    '📊 <strong>Model Updates:</strong> Models are retrained weekly with latest match data to maintain accuracy.<br>'
    '🔧 <strong>Context Adjustments:</strong> Use the sidebar sliders to incorporate team news and match-specific factors.<br>'
    '📈 <strong>Model Types:</strong> Logistic (outcome-based), Poisson (goal-based), Dixon-Coles (correlation-adjusted).'
    '</div>',
    unsafe_allow_html=True
)