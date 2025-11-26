import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy.stats import poisson
from pathlib import Path
from datetime import datetime, timedelta
import json
import plotly.graph_objects as go # Added for visuals

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
    
    .block-container { max-width: 1180px; }

    /* MATCH HEADER CARD */
    .headline-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 20px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.05);
        margin-bottom: 20px;
        text-align: center;
    }
    
    .vs-badge {
        font-size: 14px; font-weight: 800; color: #2563EB;
        background: #EFF6FF; padding: 4px 12px; border-radius: 20px;
        display: inline-block; margin: 0 15px;
    }
    
    .team-name-header { font-size: 28px; font-weight: 800; color: #111827; }
    
    /* CARDS */
    .metric-card {
        background: white; border: 1px solid #E5E7EB; 
        border-radius: 12px; padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
    }

    /* FRESHNESS BADGES */
    .badge-fresh { background-color: #d1fae5; color: #065f46; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }
    .badge-warn { background-color: #fef3c7; color: #92400e; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }
    .badge-stale { background-color: #fee2e2; color: #991b1b; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }

</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def pct(x, decimals=1):
    """Format as percentage."""
    return f"{x * 100:.{decimals}f}%"

def safe_float(val):
    """Safely convert numpy types to python float."""
    try:
        return float(val)
    except:
        return 0.0

def kelly_fraction_safe(prob, odds, fraction=0.25):
    """
    Calculate Fractional Kelly Criterion.
    Args:
        prob: Model probability (0-1)
        odds: Decimal odds
        fraction: Kelly multiplier (0.25, 0.5, 1.0)
    """
    if odds <= 1: return 0.0
    b = odds - 1
    q = 1 - prob
    f_star = (prob * (b + 1) - 1) / b
    return max(0.0, f_star * fraction)

def check_data_freshness(metadata):
    """Analyze metadata to return freshness status."""
    try:
        update_str = metadata.get("update_time", str(datetime.now()))
        # Handle various date formats if necessary, assuming ISO for now
        if "T" in update_str:
            update_dt = datetime.fromisoformat(update_str)
        else:
            update_dt = datetime.strptime(update_str, "%Y-%m-%d %H:%M:%S")
            
        age = datetime.now() - update_dt
        
        if age.days < 3:
            return "Fresh", "badge-fresh", f"Updated {age.days} days ago"
        elif age.days < 7:
            return "Warning", "badge-warn", f"Updated {age.days} days ago"
        else:
            return "Stale", "badge-stale", f"⚠️ Data is {age.days} days old"
    except:
        return "Unknown", "badge-warn", "Update time unknown"

# ═══════════════════════════════════════════════════════════════════
# LOAD MODELS & DATA
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    """Load pickled models with strict feature validation."""
    required = ["pipe_result_final.pkl", "poisson_model.pkl", "long_df.pkl", "stats.pkl", "rho_hat.pkl", "feature_cols.pkl"]
    
    # Check existence
    missing = [f for f in required if not Path(f).exists()]
    if missing:
        st.error(f"❌ Critical files missing: {', '.join(missing)}")
        st.stop()

    try:
        data = {}
        for f in required:
            with open(f, "rb") as file:
                data[f.replace(".pkl", "")] = pickle.load(file)
        
        # Load Metadata safely
        if Path("metadata.json").exists():
            with open("metadata.json", "r") as f:
                data["metadata"] = json.load(f)
        else:
            data["metadata"] = {}

        return data
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

models = load_models()
pipe_result_final = models["pipe_result_final"]
poisson_model = models["poisson_model"]
long_df = models["long_df"]
stats = models["stats"]
rho_hat = safe_float(models["rho_hat"])
feature_cols = models["feature_cols"] # Strict ordering
metadata = models["metadata"]

# Normalize names
stats["Squad"] = stats["Squad"].astype(str).str.strip()
long_df["team"] = long_df["team"].astype(str).str.strip()

# ═══════════════════════════════════════════════════════════════════
# CORE CALCULATION LOGIC
# ═══════════════════════════════════════════════════════════════════

def get_latest_team_stat(team, column):
    """
    Returns np.nan if data is missing, rather than 0.0.
    This prevents 'Zero' being interpreted as 'Average'.
    """
    try:
        team_data = long_df[long_df["team"] == team].sort_values("Date")
        if team_data.empty: return np.nan
        val = team_data.iloc[-1].get(column, np.nan)
        return float(val) if pd.notna(val) else np.nan
    except:
        return np.nan

def build_feature_vector(home, away):
    """Build feature vector enforcing strict column order."""
    try:
        # Strength Metrics
        h_stats = stats[stats["Squad"] == home].iloc[0]
        a_stats = stats[stats["Squad"] == away].iloc[0]
        
        att_h = h_stats["Home_xG"] / h_stats["Home_MP"]
        def_h = h_stats["Home_xGA"] / h_stats["Home_MP"]
        att_a = a_stats["Away_xG"] / a_stats["Away_MP"]
        def_a = a_stats["Away_xGA"] / a_stats["Away_MP"]

        # Rolling Metrics
        def get_roll(t, k): return get_latest_team_stat(t, k)
        
        # Calculate diffs
        feats = {}
        # NOTE: We use .get() with fillna(0) ONLY for the model prediction step 
        # because Sklearn cannot handle NaNs.
        feats["strength_diff"] = att_h - att_a
        feats["defense_diff"] = def_a - def_h # Inverted as per standard logic
        feats["rolling_points_diff"] = (get_roll(home, "rolling_points") or 0) - (get_roll(away, "rolling_points") or 0)
        feats["rolling_xG_diff"] = (get_roll(home, "rolling_xg_for") or 0) - (get_roll(away, "rolling_xg_for") or 0)
        feats["rolling_xGA_diff"] = (get_roll(home, "rolling_xg_against") or 0) - (get_roll(away, "rolling_xg_against") or 0)
        feats["rolling_GD_diff"] = (get_roll(home, "rolling_GD") or 0) - (get_roll(away, "rolling_GD") or 0)
        feats["finishing_overperf_diff"] = (get_roll(home, "rolling_finishing_overperf") or 0) - (get_roll(away, "rolling_finishing_overperf") or 0)
        feats["def_overperf_diff"] = (get_roll(home, "rolling_def_overperf") or 0) - (get_roll(away, "rolling_def_overperf") or 0)

        # Enforce Order
        df = pd.DataFrame([feats])
        return df[feature_cols] # Will error if columns missing, which is GOOD (safety)
    except Exception as e:
        st.error(f"Feature Engineering Error: {e}")
        return None

def compute_probs(lam_h, lam_a, rho):
    """Vectorized Dixon-Coles Probability Matrix."""
    max_g = 6
    hg, ag = np.meshgrid(np.arange(max_g+1), np.arange(max_g+1), indexing='ij')
    P = poisson.pmf(hg, lam_h) * poisson.pmf(ag, lam_a)
    
    # DC Adjustment
    if abs(rho) > 0:
        tau = np.ones_like(P)
        tau[0,0] = 1 - (lam_h * lam_a * rho)
        tau[0,1] = 1 + (lam_h * rho)
        tau[1,0] = 1 + (lam_a * rho)
        tau[1,1] = 1 - rho
        P = P * tau
        P = np.maximum(P, 0) # Clip negatives
        P = P / P.sum() # Renormalize
        
    return {
        "H": np.sum(np.tril(P, -1)),
        "D": np.trace(P),
        "A": np.sum(np.triu(P, 1)),
        "Matrix": P
    }

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    # 1. DATA FRESHNESS INDICATOR
    status, badge_class, msg = check_data_freshness(metadata)
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
        <span style="font-weight:bold;">Data Status:</span>
        <span class="{badge_class}">{msg}</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.header("⚽ Match Setup")
    teams = sorted(stats["Squad"].unique())
    home_team = st.selectbox("🏠 Home Team", teams, index=0)
    # 7. Validation: remove home team from away list
    away_teams_filtered = [t for t in teams if t != home_team]
    away_team = st.selectbox("✈️ Away Team", away_teams_filtered, index=0)

    st.divider()
    
    # 2. IMPROVED BETTING INPUTS
    st.subheader("💰 Bankroll Management")
    bankroll = st.number_input("Bankroll (£)", value=1000.0, step=50.0)
    
    # Max Stake Safety Cap
    max_stake_pct = st.slider("Max Stake Cap (%)", 1, 10, 5, help="Never bet more than X% of bankroll on one game.")
    
    # Fractional Kelly
    kelly_mode = st.selectbox(
        "Kelly Aggressiveness", 
        ["Quarter (Recommended)", "Half", "Full"],
        index=0,
        help="Full Kelly is mathematically optimal for growth but highly volatile. Quarter Kelly reduces variance."
    )
    kelly_multiplier = {"Quarter (Recommended)": 0.25, "Half": 0.5, "Full": 1.0}[kelly_mode]

    st.caption("Market Odds (Decimal)")
    c1, c2, c3 = st.columns(3)
    odds_h = c1.number_input("Home", 2.20)
    odds_d = c2.number_input("Draw", 3.30)
    odds_a = c3.number_input("Away", 3.10)
    
    st.divider()

    # 3. CONTEXT EXPLANATION
    st.subheader("🎯 Context Adjustments")
    with st.expander("❓ What do these sliders do?"):
        st.markdown("""
        **Adjust xG (Expected Goals) based on news:**
        * **+0.1:** Minor advantage (e.g., Home team rested).
        * **+0.2:** Moderate (e.g., Star striker returns).
        * **+0.5:** Major (e.g., Opponent playing youth team).
        * **Negative:** Shift advantage to the Away team.
        """)
        
    context_raw = st.slider("Home Advantage Shift", -0.5, 0.5, 0.0, 0.05)

# ═══════════════════════════════════════════════════════════════════
# MAIN PREDICTION ENGINE
# ═══════════════════════════════════════════════════════════════════

# Header
st.markdown(f"""
<div class="headline-card">
    <span class="team-name-header">{home_team}</span>
    <span class="vs-badge">VS</span>
    <span class="team-name-header">{away_team}</span>
</div>
""", unsafe_allow_html=True)

if st.button("Generate Prediction", type="primary", use_container_width=True):
    
    # A. LOGISTIC MODEL
    feat_vec = build_feature_vector(home_team, away_team)
    if feat_vec is not None:
        log_probs = pipe_result_final.predict_proba(feat_vec)[0]
        log_res = dict(zip(pipe_result_final.classes_, log_probs))
    else:
        st.error("Could not build features.")
        st.stop()

    # B. POISSON/DC MODEL
    p_df = pd.DataFrame({"team": [home_team, away_team], "opponent": [away_team, home_team], "is_home": [1, 0]})
    base_lambdas = poisson_model.predict(p_df)
    lam_h = float(base_lambdas.iloc[0]) * np.exp(context_raw) # Exponential scaling for safety
    lam_a = float(base_lambdas.iloc[1]) * np.exp(-context_raw)
    
    dc_res = compute_probs(lam_h, lam_a, rho_hat)

    # C. MONTE CARLO SIMULATION (For Uncertainty)
    sim_h = poisson.rvs(lam_h, size=1000)
    sim_a = poisson.rvs(lam_a, size=1000)
    sim_diff = sim_h - sim_a
    sim_home_win = np.mean(sim_diff > 0)
    
    # ═══════════════════════════════════════════════════════════════════
    # RESULTS DISPLAY
    # ═══════════════════════════════════════════════════════════════════
    
    # 1. PROBABILITY CARDS
    st.subheader("📊 Model Probabilities")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Home Win", pct(dc_res["H"]), delta=f"{pct(dc_res['H'] - (1/odds_h))} Edge" if dc_res['H'] > 1/odds_h else None)
    with col2:
        st.metric("Draw", pct(dc_res["D"]))
    with col3:
        st.metric("Away Win", pct(dc_res["A"]), delta=f"{pct(dc_res['A'] - (1/odds_a))} Edge" if dc_res['A'] > 1/odds_a else None)

    # 2. BETTING ANALYSIS (KELLY & MARGIN)
    st.divider()
    st.subheader("💰 Value Analysis")
    
    # Calculate Bookie Margin
    bookie_margin = (1/odds_h + 1/odds_d + 1/odds_a) - 1
    
    k_col1, k_col2 = st.columns(2)
    
    with k_col1:
        st.info(f"🏦 **Bookmaker Margin:** {pct(bookie_margin)}\n\nThis is the 'house edge' you must overcome.")

    # Kelly Calculation
    best_edge = -999
    best_bet = None
    best_stake = 0
    
    for outcome, prob, odds in [("Home", dc_res["H"], odds_h), ("Draw", dc_res["D"], odds_d), ("Away", dc_res["A"], odds_a)]:
        edge = prob - (1/odds)
        if edge > best_edge:
            best_edge = edge
            best_bet = outcome
            f_kelly = kelly_fraction_safe(prob, odds, kelly_multiplier)
            raw_stake = f_kelly * bankroll
            # Apply Max Cap
            max_allowed = bankroll * (max_stake_pct / 100.0)
            best_stake = min(raw_stake, max_allowed)

    with k_col2:
        if best_edge > 0 and best_stake > 0:
            st.success(f"**Recommended Bet:** {best_bet} @ {odds:.2f}\n\n"
                       f"**Stake:** £{best_stake:.2f} ({best_stake/bankroll:.1%} of bank)\n"
                       f"**Edge:** {pct(best_edge)}")
        else:
            st.warning("**No Value Bet Found**\n\nMarket odds are too efficient given your model's confidence.")

    # 3. STATS & VISUALS (HANDLING MISSING DATA)
    st.divider()
    
    tab1, tab2 = st.tabs(["📈 Team Stats", "🎲 Simulation Spread"])
    
    with tab1:
        def format_stat(val, is_pct=False):
            if pd.isna(val): return "N/A"
            return f"{val:+.2f}"
            
        stats_data = {
            "Metric": ["xG (Roll 5)", "xGA (Roll 5)", "Goal Diff (Roll 5)", "Finishing Luck"],
            home_team: [
                format_stat(get_latest_team_stat(home_team, "rolling_xg_for")),
                format_stat(get_latest_team_stat(home_team, "rolling_xg_against")),
                format_stat(get_latest_team_stat(home_team, "rolling_GD")),
                format_stat(get_latest_team_stat(home_team, "rolling_finishing_overperf"))
            ],
            away_team: [
                format_stat(get_latest_team_stat(away_team, "rolling_xg_for")),
                format_stat(get_latest_team_stat(away_team, "rolling_xg_against")),
                format_stat(get_latest_team_stat(away_team, "rolling_GD")),
                format_stat(get_latest_team_stat(away_team, "rolling_finishing_overperf"))
            ]
        }
        st.table(pd.DataFrame(stats_data))
        st.caption("*N/A indicates missing historical data. 'Finishing Luck' > 0 means team scores more than xG implies.*")

    with tab2:
        #         # Monte Carlo Visualization
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=sim_h, name=home_team, opacity=0.75, marker_color='blue'))
        fig.add_trace(go.Histogram(x=sim_a, name=away_team, opacity=0.75, marker_color='red'))
        fig.update_layout(barmode='overlay', title="Simulated Goal Distribution (1000 Matches)", 
                          xaxis_title="Goals Scored", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("This chart helps visualize the uncertainty. Overlap indicates a higher chance of a Draw.")

else:
    st.info("👈 Adjust settings in the sidebar and click **Generate Prediction**.")