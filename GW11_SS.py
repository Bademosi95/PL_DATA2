"""
Premier League Match Predictor - Enhanced Version
--------------------------------------------------
Optimized for performance, visuals, and UX.
Includes Vectorized Dixon-Coles, Heatmaps, and Fractional Kelly.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy.stats import poisson
from pathlib import Path
from datetime import datetime
import json
import plotly.express as px

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="KB Premier League Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════
# CUSTOM STYLING (Sofascore Vibe)
# ═══════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
    /* GLOBAL APP STYLING */
    body, .stApp {
        background: #F5F7FB;
        color: #111827;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", sans-serif;
    }
    
    .block-container {
        max-width: 1180px;
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
    }

    /* CARDS & CONTAINERS */
    .headline-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 20px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 24px;
    }
    
    .model-card {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 20px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        height: 100%;
        transition: transform 0.2s;
    }
    
    .model-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    }

    /* TYPOGRAPHY */
    .main-title { font-size: 32px; font-weight: 800; color: #111827; letter-spacing: -0.02em; }
    .sub-title { font-size: 16px; color: #6B7280; margin-bottom: 24px; }
    
    .team-name { font-size: 24px; font-weight: 800; color: #111827; }
    .team-tag { font-size: 12px; font-weight: 600; text-transform: uppercase; color: #9CA3AF; letter-spacing: 0.05em; }
    
    .vs-badge {
        background: #EFF6FF; color: #2563EB;
        font-weight: 800; font-size: 14px;
        padding: 6px 12px; border-radius: 20px;
        border: 1px solid #DBEAFE;
    }

    /* METRICS */
    .metric-value { font-size: 28px; font-weight: 800; color: #111827; }
    .metric-label { font-size: 13px; color: #6B7280; font-weight: 500; }
    
    /* UTILS */
    .divider { margin: 24px 0; border-top: 1px solid #E5E7EB; }
    
    /* SIDEBAR */
    [data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E5E7EB; }
    
    /* DATAFRAMES */
    [data-testid="stDataFrame"] { border: 1px solid #E5E7EB; border-radius: 12px; overflow: hidden; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    """Load models with robust error handling."""
    required_files = {
        "pipe_result_final.pkl": "Logistic Model",
        "poisson_model.pkl": "Poisson Model",
        "long_df.pkl": "Match History",
        "stats.pkl": "Team Stats",
        "rho_hat.pkl": "Dixon-Coles Parameter"
    }
    
    # Check existence
    missing = [f for f in required_files if not Path(f).exists()]
    if missing:
        st.error(f"❌ Missing files: {', '.join(missing)}")
        st.stop()
        
    try:
        data = {}
        for filename in required_files:
            with open(filename, "rb") as f:
                data[filename.split(".")[0]] = pickle.load(f)
                
        # Optional: Load feature columns order if exists, else use default
        if Path("feature_cols.pkl").exists():
            with open("feature_cols.pkl", "rb") as f:
                data["feature_cols"] = pickle.load(f)
        else:
            data["feature_cols"] = [
                "strength_diff", "defense_diff", "rolling_points_diff", 
                "rolling_xG_diff", "rolling_xGA_diff", "rolling_GD_diff", 
                "finishing_overperf_diff", "def_overperf_diff"
            ]
            
        # Metadata
        if Path("metadata.json").exists():
            with open("metadata.json", "r") as f:
                data["metadata"] = json.load(f)
        else:
            data["metadata"] = {}

        return data

    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

# Load Data
models = load_models()
pipe_result_final = models["pipe_result_final"]
poisson_model = models["poisson_model"]
long_df = models["long_df"]
stats = models["stats"]
rho_hat = models["rho_hat"] if isinstance(models["rho_hat"], float) else float(models["rho_hat"])
feature_cols = models["feature_cols"]
metadata = models["metadata"]

# Normalize team names immediately
stats["Squad"] = stats["Squad"].astype(str).str.strip()
long_df["team"] = long_df["team"].astype(str).str.strip()

# ═══════════════════════════════════════════════════════════════════
# MATHEMATICAL FUNCTIONS (OPTIMIZED)
# ═══════════════════════════════════════════════════════════════════

def kelly_fraction(prob, odds):
    """Calculate Kelly Criterion fraction."""
    if odds <= 1: return 0.0
    b = odds - 1
    q = 1 - prob
    return (prob * (b + 1) - 1) / b

def compute_probs_vectorized(lam_h, lam_a, rho, max_goals=6):
    """
    Vectorized calculation of probability matrix (Home Goals x Away Goals).
    Much faster than nested loops.
    """
    # Create grid of goals
    h_goals, a_goals = np.meshgrid(
        np.arange(max_goals + 1), 
        np.arange(max_goals + 1), 
        indexing='ij'
    )
    
    # Base Poisson Probability Matrix
    pmf_h = poisson.pmf(h_goals, lam_h)
    pmf_a = poisson.pmf(a_goals, lam_a)
    P = pmf_h * pmf_a
    
    # Dixon-Coles Adjustment (Vectorized)
    if abs(rho) > 1e-6:
        tau = np.ones_like(P)
        
        # Apply corrections to specific scorelines
        # 0-0
        tau[0, 0] = 1 - (lam_h * lam_a * rho)
        # 0-1
        tau[0, 1] = 1 + lam_h * rho
        # 1-0
        tau[1, 0] = 1 + lam_a * rho
        # 1-1
        tau[1, 1] = 1 - rho
        
        # Apply and re-normalize
        P = P * tau
        P[P < 0] = 0  # Safety floor
        P = P / P.sum()
        
    return P

def extract_markets_from_matrix(P):
    """Extract standard betting markets from probability matrix."""
    # Create masks for outcomes
    h_idx, a_idx = np.indices(P.shape)
    
    return {
        "P_home": P[h_idx > a_idx].sum(),
        "P_draw": np.trace(P),
        "P_away": P[h_idx < a_idx].sum(),
        "P_over_1_5": P[(h_idx + a_idx) > 1.5].sum(),
        "P_over_2_5": P[(h_idx + a_idx) > 2.5].sum(),
        "P_BTTS": P[(h_idx > 0) & (a_idx > 0)].sum(),
        "P_CS_Home": P[:, 0].sum(),
        "P_CS_Away": P[0, :].sum(),
        "Matrix": P # Return raw matrix for heatmap
    }

def get_latest_team_stat(team, column, default=0.0):
    try:
        rows = long_df[long_df["team"] == team]
        return float(rows.iloc[-1][column]) if not rows.empty else default
    except:
        return default

# ═══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════

def build_feature_vector(home_team, away_team):
    """Reconstructs the exact feature vector used during training."""
    try:
        # 1. Strength Metrics from Stats
        h_stats = stats[stats["Squad"] == home_team].iloc[0]
        a_stats = stats[stats["Squad"] == away_team].iloc[0]
        
        att_h = h_stats["Home_xG"] / h_stats["Home_MP"]
        def_h = h_stats["Home_xGA"] / h_stats["Home_MP"]
        att_a = a_stats["Away_xG"] / a_stats["Away_MP"]
        def_a = a_stats["Away_xGA"] / a_stats["Away_MP"]
        
        # 2. Rolling Metrics from Long DF
        def get_rolling(team):
            return {
                "pts": get_latest_team_stat(team, "rolling_points"),
                "xg": get_latest_team_stat(team, "rolling_xg_for"),
                "xga": get_latest_team_stat(team, "rolling_xg_against"),
                "gd": get_latest_team_stat(team, "rolling_GD"),
                "fin": get_latest_team_stat(team, "rolling_finishing_overperf"),
                "def": get_latest_team_stat(team, "rolling_def_overperf")
            }
            
        h_roll = get_rolling(home_team)
        a_roll = get_rolling(away_team)
        
        # 3. Combine into Diff Features
        features = {
            "strength_diff": att_h - att_a,
            "defense_diff": def_a - def_h,
            "rolling_points_diff": h_roll["pts"] - a_roll["pts"],
            "rolling_xG_diff": h_roll["xg"] - a_roll["xg"],
            "rolling_xGA_diff": h_roll["xga"] - a_roll["xga"],
            "rolling_GD_diff": h_roll["gd"] - a_roll["gd"],
            "finishing_overperf_diff": h_roll["fin"] - a_roll["fin"],
            "def_overperf_diff": h_roll["def"] - a_roll["def"]
        }
        
        return pd.DataFrame([features])[feature_cols]
        
    except Exception as e:
        st.error(f"Error building features: {e}")
        return None

# ═══════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def plot_score_heatmap(matrix, h_team, a_team):
    """Create a Plotly heatmap for correct scores."""
    # Clip matrix to 4x4 for cleaner display, or keep full
    display_matrix = matrix[:5, :5] # Show up to 4-4
    
    fig = px.imshow(
        display_matrix,
        labels=dict(x=f"{a_team} Goals", y=f"{h_team} Goals", color="Prob"),
        x=[str(i) for i in range(5)],
        y=[str(i) for i in range(5)],
        text_auto='.1%',
        color_continuous_scale="Blues",
        aspect="auto"
    )
    fig.update_layout(
        title={
            'text': "Correct Score Probabilities",
            'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'
        },
        margin=dict(l=20, r=20, t=40, b=20),
        height=350,
        coloraxis_showscale=False
    )
    fig.update_traces(textfont_size=12)
    return fig

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Match Setup")
    
    # Dynamic Team Selection (Prevents selecting same team)
    all_teams = sorted(stats["Squad"].unique())
    home_team = st.selectbox("🏠 Home Team", all_teams, index=0)
    
    remaining_teams = [t for t in all_teams if t != home_team]
    away_team = st.selectbox("✈️ Away Team", remaining_teams, index=0)
    
    st.divider()
    
    st.subheader("💰 Betting Config")
    bankroll = st.number_input("Bankroll (£)", value=1000.0, step=50.0)
    
    col_odds1, col_odds2, col_odds3 = st.columns(3)
    odds_h = col_odds1.number_input("Home", value=2.20, min_value=1.01, step=0.05)
    odds_d = col_odds2.number_input("Draw", value=3.30, min_value=1.01, step=0.05)
    odds_a = col_odds3.number_input("Away", value=3.10, min_value=1.01, step=0.05)
    
    kelly_mode = st.radio("Kelly Aggressiveness", ["Full", "Half (Recommended)", "Quarter"], index=1)
    kelly_multiplier = {"Full": 1.0, "Half": 0.5, "Quarter": 0.25}[kelly_mode]
    
    st.divider()
    
    st.subheader("🎛️ Context Adjustments")
    st.caption("Adjust for injuries, tactics, or morale.")
    
    adv_mode = st.checkbox("Advanced Inputs", value=False)
    
    if not adv_mode:
        raw_adj = st.slider("Home Advantage Shift", -2.0, 2.0, 0.0, 0.1, 
                           help="Positive = Boost Home, Negative = Boost Away")
        context_factor = raw_adj * 0.1 # Scale down
    else:
        att = st.slider("Attack Strength", -1.0, 1.0, 0.0, 0.1)
        defn = st.slider("Defense Strength", -1.0, 1.0, 0.0, 0.1)
        mot = st.slider("Motivation", -1.0, 1.0, 0.0, 0.1)
        context_factor = (att + defn + mot) / 10.0

# ═══════════════════════════════════════════════════════════════════
# MAIN LOGIC
# ═══════════════════════════════════════════════════════════════════

# Title Section
st.markdown('<div class="main-title">Premier League Predictor</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">Matchday Analysis: {home_team} vs {away_team}</div>', unsafe_allow_html=True)

# 1. GENERATE PREDICTIONS
with st.spinner("Crunching numbers..."):
    
    # A. Logistic (Baseline)
    X_feat = build_feature_vector(home_team, away_team)
    if X_feat is not None:
        log_probs = pipe_result_final.predict_proba(X_feat)[0]
        log_res = dict(zip(pipe_result_final.classes_, log_probs))
    else:
        st.stop()
        
    # B. Poisson & Dixon-Coles (Goal Based)
    # Get base lambdas
    pred_df = pd.DataFrame({"team": [home_team, away_team], "opponent": [away_team, home_team], "is_home": [1, 0]})
    base_preds = poisson_model.predict(pred_df)
    lam_h_base, lam_a_base = float(base_preds.iloc[0]), float(base_preds.iloc[1])
    
    # Apply Context (Using Exponential for safety)
    # If context > 0, Home UP, Away DOWN.
    lam_h_adj = lam_h_base * np.exp(context_factor)
    lam_a_adj = lam_a_base * np.exp(-context_factor)
    
    # Calculate Matrices
    P_matrix_base = compute_probs_vectorized(lam_h_base, lam_a_base, rho_hat)
    P_matrix_adj  = compute_probs_vectorized(lam_h_adj, lam_a_adj, rho_hat)
    
    # Extract Markets
    mkts_base = extract_markets_from_matrix(P_matrix_base)
    mkts_adj = extract_markets_from_matrix(P_matrix_adj)

# 2. HEADLINE DISPLAY
cols = st.columns([2, 1, 1])
with cols[0]:
    st.markdown(
        f'''<div class="headline-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div class="team-tag">HOME</div>
                    <div class="team-name">{home_team}</div>
                    <div style="font-size:24px; font-weight:700; color:#3B82F6;">{mkts_adj['P_home']:.1%}</div>
                </div>
                <div class="vs-badge">VS</div>
                <div style="text-align:right;">
                    <div class="team-tag">AWAY</div>
                    <div class="team-name">{away_team}</div>
                    <div style="font-size:24px; font-weight:700; color:#EF4444;">{mkts_adj['P_away']:.1%}</div>
                </div>
            </div>
            <div style="text-align:center; margin-top:10px; color:#6B7280; font-size:14px;">
                Draw: <strong>{mkts_adj['P_draw']:.1%}</strong>
            </div>
        </div>''', unsafe_allow_html=True
    )

# 3. DETAILED MODELS
st.subheader("📊 Model Breakdown")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        f'''<div class="model-card">
            <h4>📈 Logistic (Baseline)</h4>
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <span>Home</span><span style="font-weight:700">{log_res.get('H',0):.1%}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <span>Draw</span><span style="font-weight:700">{log_res.get('D',0):.1%}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span>Away</span><span style="font-weight:700">{log_res.get('A',0):.1%}</span>
            </div>
            <div style="margin-top:15px; font-size:12px; color:#6B7280;">
                Based on historical form and team strength only. No adjustments.
            </div>
        </div>''', unsafe_allow_html=True
    )

with c2:
    st.markdown(
        f'''<div class="model-card">
            <h4>📐 Exp. Goals (xG)</h4>
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <span>{home_team}</span>
                <span style="font-weight:700; color:#3B82F6">{lam_h_adj:.2f}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <span>{away_team}</span>
                <span style="font-weight:700; color:#EF4444">{lam_a_adj:.2f}</span>
            </div>
             <div style="margin-top:15px; font-size:12px; color:#6B7280;">
                Includes context adjustments applied to base Poisson numbers.
            </div>
        </div>''', unsafe_allow_html=True
    )

with c3:
    st.plotly_chart(
        plot_score_heatmap(mkts_adj['Matrix'], home_team, away_team), 
        use_container_width=True
    )

# 4. VALUE & BETTING
st.divider()
st.subheader("💰 Value Analysis & Staking")

# Calculate Edges & Kelly
probs = {"H": mkts_adj['P_home'], "D": mkts_adj['P_draw'], "A": mkts_adj['P_away']}
odds_map = {"H": odds_h, "D": odds_d, "A": odds_a}

bet_data = []
for res in ["H", "D", "A"]:
    prob = probs[res]
    odds = odds_map[res]
    edge = prob - (1/odds)
    kf = kelly_fraction(prob, odds)
    stake = max(0, kf * bankroll * kelly_multiplier)
    
    bet_data.append({
        "Outcome": {"H": home_team, "D": "Draw", "A": away_team}[res],
        "Model Prob": f"{prob:.1%}",
        "Implied Prob": f"{(1/odds):.1%}",
        "Edge": edge,
        "Kelly Stake": stake
    })

# Display Betting Cards
b1, b2, b3 = st.columns(3)
for idx, (col, data) in enumerate(zip([b1, b2, b3], bet_data)):
    with col:
        is_pos = data["Edge"] > 0
        bg_color = "#ECFDF5" if is_pos else "#F9FAFB" # Green tint if value
        border = "#10B981" if is_pos else "#E5E7EB"
        
        st.markdown(
            f'''<div style="background:{bg_color}; border:1px solid {border}; border-radius:12px; padding:16px;">
                <div style="font-weight:700; margin-bottom:8px;">{data['Outcome']}</div>
                <div style="font-size:13px; display:flex; justify-content:space-between;">
                    <span>Edge:</span>
                    <span style="color:{'#059669' if is_pos else '#DC2626'}">{data['Edge']:.1%}</span>
                </div>
                <div style="font-size:20px; font-weight:800; margin-top:8px;">
                    £{data['Kelly Stake']:.2f}
                </div>
                <div style="font-size:11px; color:#6B7280;">Recommended Stake</div>
            </div>''', unsafe_allow_html=True
        )

# 5. STATS COMPARISON
with st.expander("📋 Detailed Team Stats Comparison"):
    
    # Helper to get stat safely
    def get_stat(team, metric): return get_latest_team_stat(team, metric)
    
    stats_df = pd.DataFrame({
        "Metric": ["xG For (Last 5)", "xG Against (Last 5)", "Goal Diff (Last 5)", "Finishing Luck"],
        home_team: [
            f"{get_stat(home_team, 'rolling_xg_for'):.2f}",
            f"{get_stat(home_team, 'rolling_xg_against'):.2f}",
            f"{get_stat(home_team, 'rolling_GD'):+.1f}",
            f"{get_stat(home_team, 'rolling_finishing_overperf'):+.2f}"
        ],
        away_team: [
            f"{get_stat(away_team, 'rolling_xg_for'):.2f}",
            f"{get_stat(away_team, 'rolling_xg_against'):.2f}",
            f"{get_stat(away_team, 'rolling_GD'):+.1f}",
            f"{get_stat(away_team, 'rolling_finishing_overperf'):+.2f}"
        ]
    })
    st.dataframe(stats_df, hide_index=True, use_container_width=True)
    st.caption("Finishing Luck > 0 implies the team is scoring more than their xG suggests.")

# Footer
st.markdown("---")
st.caption("Disclaimer: This tool is for informational purposes only. Models: Logistic Regression + Dixon-Coles Poisson Ensemble.")