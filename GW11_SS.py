"""
Premier League Match Predictor - Streamlit App
-----------------------------------------------
Modern, sleek UI with Apple/Google design aesthetic.
Production version aligned with weekly_update.py pipeline.
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
    page_title="Premier League Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════
# MODERN DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* Header Section */
    .hero-section {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 3rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    .hero-title {
        font-size: 42px;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .hero-subtitle {
        font-size: 16px;
        color: #64748b;
        font-weight: 500;
        margin-top: 0.5rem;
    }
    
    .match-header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 2rem;
        margin-top: 2rem;
        padding: 1.5rem;
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 16px;
    }
    
    .team-name {
        font-size: 28px;
        font-weight: 700;
        color: #1e293b;
    }
    
    .vs-divider {
        font-size: 20px;
        font-weight: 600;
        color: #94a3b8;
        padding: 0 1rem;
    }
    
    /* Card System */
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 1.75rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.5);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 60px rgba(0,0,0,0.15);
    }
    
    .card-title {
        font-size: 18px;
        font-weight: 700;
        color: #1e293b;
        margin: 0 0 1.25rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .card-icon {
        font-size: 20px;
    }
    
    /* Metrics */
    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 0;
        border-bottom: 1px solid #f1f5f9;
    }
    
    .metric-row:last-child {
        border-bottom: none;
    }
    
    .metric-label {
        font-size: 14px;
        font-weight: 500;
        color: #64748b;
    }
    
    .metric-value {
        font-size: 16px;
        font-weight: 700;
        color: #1e293b;
        padding: 0.25rem 0.75rem;
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 8px;
    }
    
    .metric-value-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.35rem 1rem;
        border-radius: 10px;
        font-weight: 700;
    }
    
    /* Headline Metrics */
    .headline-metric {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
        border: 1px solid rgba(255,255,255,0.5);
        transition: all 0.3s ease;
    }
    
    .headline-metric:hover {
        transform: scale(1.05);
        box-shadow: 0 15px 50px rgba(0,0,0,0.12);
    }
    
    .headline-label {
        font-size: 13px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    .headline-value {
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(226, 232, 240, 0.5);
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #1e293b !important;
        font-weight: 600 !important;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 24px;
        font-weight: 700;
        color: white;
        margin: 2rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, rgba(255,255,255,0.3) 0%, transparent 100%);
        margin: 1.5rem 0;
        border: none;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 12px;
        font-weight: 600;
        color: #1e293b;
        border: 1px solid rgba(255,255,255,0.5);
    }
    
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 0 0 12px 12px;
        border: 1px solid rgba(255,255,255,0.5);
        border-top: none;
    }
    
    /* Status Indicators */
    .status-badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    
    .badge-success {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    
    .badge-warning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
    }
    
    .badge-neutral {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
    }
    
    /* Form Indicators */
    .form-icon {
        font-size: 18px;
        display: inline-block;
        margin: 0 2px;
    }
    
    /* Footer */
    .footer-section {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 2rem;
        margin-top: 3rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.5);
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
    }
    
    .footer-text {
        font-size: 13px;
        color: #64748b;
        line-height: 1.8;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .glass-card, .headline-metric {
        animation: fadeIn 0.5s ease;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
        }
        
        .hero-section {
            padding: 2rem 1.5rem;
        }
        
        .hero-title {
            font-size: 32px;
        }
        
        .match-header {
            flex-direction: column;
            gap: 1rem;
        }
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# LOAD MODELS
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
    return f"{x * 100:.{decimals}f}%"

def safe_division(numerator, denominator, default=0.0):
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default

def get_latest_team_stat(team, column, default=0.0):
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
        return default

def get_team_strength_metrics(team):
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
    home_strength = get_team_strength_metrics(home_team)
    away_strength = get_team_strength_metrics(away_team)
    
    if home_strength is None or away_strength is None:
        return None
    
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
    
    return pd.DataFrame([features])[feature_cols]

def predict_logistic(home_team, away_team):
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
        return "", "", 0

def get_team_position(team):
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
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚽ Match Setup")
    st.markdown("")
    
    teams = sorted(stats["Squad"].unique())
    
    if len(teams) == 0:
        st.error("No teams found in stats file!")
        st.stop()
    
    home_team = st.selectbox("🏠 Home Team", teams, key="home")
    away_team = st.selectbox("✈️ Away Team", teams, key="away")
    
    st.markdown("---")
    
    st.markdown("## 🎯 Match Context")
    st.caption("Adjust for injuries, form, tactics")
    
    advanced_mode = st.checkbox("Advanced Controls", value=False)
    
    if not advanced_mode:
        context_raw = st.slider(
            "Context Adjustment",
            min_value=-3.0,
            max_value=3.0,
            value=0.0,
            step=0.1,
            help="Negative = Away advantage | Positive = Home advantage"
        )
        context_adj = context_raw / 10.0
    else:
        st.markdown("**Granular Controls:**")
        att_adj = st.slider("⚔️ Attack", -3.0, 3.0, 0.0, 0.1)
        def_adj = st.slider("🛡️ Defense", -3.0, 3.0, 0.0, 0.1)
        morale_adj = st.slider("💪 Morale", -3.0, 3.0, 0.0, 0.1)
        context_adj = (att_adj + def_adj + 0.5 * morale_adj) / 20.0
    
    if abs(context_adj) < 0.01:
        st.markdown('<span class="status-badge badge-neutral">⚖️ Neutral</span>', unsafe_allow_html=True)
    elif context_adj > 0:
        st.markdown(f'<span class="status-badge badge-success">🏠 Home +{context_adj:.2f}</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="status-badge badge-warning">✈️ Away {context_adj:.2f}</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    with st.expander("ℹ️ Model Info"):
        if metadata:
            try:
                update_time = datetime.fromisoformat(metadata.get("update_time", ""))
                st.text(f"Updated: {update_time.strftime('%Y-%m-%d %H:%M')}")
            except:
                st.text("Updated: Unknown")
            st.text(f"Data: {metadata.get('date_range', 'Unknown')}")
            st.text(f"Teams: {len(metadata.get('teams', []))}")
            st.text(f"DC rho: {metadata.get('rho', rho_hat):.4f}")
        else:
            st.text(f"DC rho: {rho_hat:.4f}")

# ═══════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════

if home_team == away_team:
    st.error("### ⚠️ Please select different teams")
    st.stop()

# Hero Section
st.markdown('''
<div class="hero-section">
    <div class="hero-title">Premier League Predictor</div>
    <div class="hero-subtitle">AI-powered match analysis • Three model ensemble</div>
    <div class="match-header">
        <span class="team-name">{}</span>
        <span class="vs-divider">VS</span>
        <span class="team-name">{}</span>
    </div>
</div>
'''.format(home_team, away_team), unsafe_allow_html=True)

# Generate Predictions
with st.spinner("🔮 Generating predictions..."):
    log_probs = predict_logistic(home_team, away_team)
    
    if log_probs is None:
        st.error("❌ Could not generate predictions. Check team data.")
        st.stop()
    
    lambdas = get_poisson_lambdas(home_team, away_team, context_adj=context_adj)
    
    if lambdas is None:
        st.error("❌ Could not generate Poisson predictions.")
        st.stop()
    
    lam_h_adj, lam_a_adj, lam_h_base, lam_a_base = lambdas
    
    poisson_adj = compute_scoreline_probabilities(lam_h_adj, lam_a_adj, use_dc=False)
    poisson_base = compute_scoreline_probabilities(lam_h_base, lam_a_base, use_dc=False)
    
    dc_adj = compute_scoreline_probabilities(lam_h_adj, lam_a_adj, use_dc=True, rho=rho_hat)
    dc_base = compute_scoreline_probabilities(lam_h_base, lam_a_base, use_dc=True, rho=rho_hat)

# Headline Metrics
st.markdown('<div class="section-header">🎯 Headline Prediction</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f'''
    <div class="headline-metric">
        <div class="headline-label">🏠 Home Win</div>
        <div class="headline-value">{pct(dc_adj["P_home"])}</div>
    </div>
    ''', unsafe_allow_html=True)

with col2:
    st.markdown(f'''
    <div class="headline-metric">
        <div class="headline-label">🤝 Draw</div>
        <div class="headline-value">{pct(dc_adj["P_draw"])}</div>
    </div>
    ''', unsafe_allow_html=True)

with col3:
    st.markdown(f'''
    <div class="headline-metric">
        <div class="headline-label">✈️ Away Win</div>
        <div class="headline-value">{pct(dc_adj["P_away"])}</div>
    </div>
    ''', unsafe_allow_html=True)

with col4:
    st.markdown(f'''
    <div class="headline-metric">
        <div class="headline-label">⚽ BTTS</div>
        <div class="headline-value">{pct(dc_adj["P_BTTS"])}</div>
    </div>
    ''', unsafe_allow_html=True)

if abs(context_adj) > 0.01:
    delta = (dc_adj["P_home"] - dc_base["P_home"]) * 100
    st.markdown(f"<p style='text-align:center; color:white; margin-top:1rem;'>💡 Context adjustment changed home win by <strong>{delta:+.1f}pp</strong></p>", unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# Model Cards
st.markdown('<div class="section-header">📊 Model Comparison</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f'''
    <div class="glass-card">
        <div class="card-title"><span class="card-icon">📈</span> Logistic Model</div>
        <div class="metric-row">
            <span class="metric-label">Home Win</span>
            <span class="metric-value-highlight">{pct(log_probs.get("H", 0.0))}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Draw</span>
            <span class="metric-value-highlight">{pct(log_probs.get("D", 0.0))}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Away Win</span>
            <span class="metric-value-highlight">{pct(log_probs.get("A", 0.0))}</span>
        </div>
        <p style="margin-top:1rem; font-size:12px; color:#64748b; line-height:1.6;">
            Historical result-based prediction using team strength and 5-game form. Context-independent baseline.
        </p>
    </div>
    ''', unsafe_allow_html=True)

with c2:
    st.markdown(f'''
    <div class="glass-card">
        <div class="card-title"><span class="card-icon">📐</span> Poisson Model</div>
        <div class="metric-row">
            <span class="metric-label">xG Home</span>
            <span class="metric-value">{lam_h_adj:.2f}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">xG Away</span>
            <span class="metric-value">{lam_a_adj:.2f}</span>
        </div>
        <hr style="margin:0.75rem 0; border:none; border-top:1px solid #f1f5f9;">
        <div class="metric-row">
            <span class="metric-label">Home Win</span>
            <span class="metric-value">{pct(poisson_adj["P_home"])}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Draw</span>
            <span class="metric-value">{pct(poisson_adj["P_draw"])}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Away Win</span>
            <span class="metric-value">{pct(poisson_adj["P_away"])}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Over 2.5</span>
            <span class="metric-value">{pct(poisson_adj["P_over_2_5"])}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">BTTS</span>
            <span class="metric-value">{pct(poisson_adj["P_BTTS"])}</span>
        </div>
        <p style="margin-top:1rem; font-size:12px; color:#64748b; line-height:1.6;">
            Goal expectation model. Context adjustments affect xG values.
        </p>
    </div>
    ''', unsafe_allow_html=True)

with c3:
    st.markdown(f'''
    <div class="glass-card">
        <div class="card-title"><span class="card-icon">🎲</span> Dixon-Coles</div>
        <div class="metric-row">
            <span class="metric-label">Home Win</span>
            <span class="metric-value-highlight">{pct(dc_adj["P_home"])}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Draw</span>
            <span class="metric-value-highlight">{pct(dc_adj["P_draw"])}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Away Win</span>
            <span class="metric-value-highlight">{pct(dc_adj["P_away"])}</span>
        </div>
        <hr style="margin:0.75rem 0; border:none; border-top:1px solid #f1f5f9;">
        <div class="metric-row">
            <span class="metric-label">Over 1.5</span>
            <span class="metric-value">{pct(dc_adj["P_over_1_5"])}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Over 2.5</span>
            <span class="metric-value">{pct(dc_adj["P_over_2_5"])}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Over 3.5</span>
            <span class="metric-value">{pct(dc_adj["P_over_3_5"])}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">BTTS</span>
            <span class="metric-value">{pct(dc_adj["P_BTTS"])}</span>
        </div>
        <p style="margin-top:1rem; font-size:12px; color:#64748b; line-height:1.6;">
            Enhanced Poisson with correlation adjustment (rho={rho_hat:.3f}). Corrects low-scoring dependencies.
        </p>
    </div>
    ''', unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# Additional Markets
st.markdown('<div class="section-header">🎰 Betting Markets</div>', unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown(f'''
    <div class="glass-card">
        <div class="card-title"><span class="card-icon">⚽</span> Total Goals</div>
        <div class="metric-row">
            <span class="metric-label">Over 0.5</span>
            <span class="metric-value">{pct(dc_adj["P_over_0_5"])}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Over 1.5</span>
            <span class="metric-value">{pct(dc_adj["P_over_1_5"])}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Over 2.5</span>
            <span class="metric-value">{pct(dc_adj["P_over_2_5"])}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Over 3.5</span>
            <span class="metric-value">{pct(dc_adj["P_over_3_5"])}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Over 4.5</span>
            <span class="metric-value">{pct(dc_adj["P_over_4_5"])}</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)

with col_b:
    st.markdown(f'''
    <div class="glass-card">
        <div class="card-title"><span class="card-icon">🎯</span> Both Teams to Score</div>
        <div class="metric-row">
            <span class="metric-label">BTTS Yes</span>
            <span class="metric-value-highlight">{pct(dc_adj["P_BTTS"])}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">BTTS No</span>
            <span class="metric-value">{pct(1 - dc_adj["P_BTTS"])}</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)

with col_c:
    st.markdown(f'''
    <div class="glass-card">
        <div class="card-title"><span class="card-icon">🛡️</span> Clean Sheets</div>
        <div class="metric-row">
            <span class="metric-label">{home_team}</span>
            <span class="metric-value">{pct(dc_adj["P_clean_sheet_home"])}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">{away_team}</span>
            <span class="metric-value">{pct(dc_adj["P_clean_sheet_away"])}</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# Team Form & Statistics
with st.expander("📋 Team Form & Statistics", expanded=False):
    col_home, col_away = st.columns(2)
    
    with col_home:
        st.markdown(f"### 🏠 {home_team}")
        
        pos = get_team_position(home_team)
        form_letters, form_icons, form_points = get_team_form(home_team)
        
        st.markdown(f"**Position:** {pos}")
        st.markdown(f"**Form:** {form_icons} ({form_letters})")
        st.markdown(f"**Points (L5):** {form_points}")
        
        st.markdown("---")
        st.markdown("**Last 5 Matches:**")
        
        home_xg_for = get_latest_team_stat(home_team, "rolling_xg_for")
        home_xg_ag = get_latest_team_stat(home_team, "rolling_xg_against")
        home_gd = get_latest_team_stat(home_team, "rolling_GD")
        home_fin = get_latest_team_stat(home_team, "rolling_finishing_overperf")
        home_def = get_latest_team_stat(home_team, "rolling_def_overperf")
        
        st.markdown(f"• xG For: **{home_xg_for:.2f}**")
        st.markdown(f"• xG Against: **{home_xg_ag:.2f}**")
        st.markdown(f"• Goal Diff: **{home_gd:+.0f}**")
        st.markdown(f"• Finishing: **{home_fin:+.2f}**")
        st.markdown(f"• Defence: **{home_def:+.2f}**")
        
        if home_fin > 1:
            st.success("💚 Hot finishing streak")
        elif home_fin < -1:
            st.warning("⚠️ Cold finishing streak")
        
        if home_def > 1:
            st.success("💚 Solid defensive form")
        elif home_def < -1:
            st.warning("⚠️ Leaky defence")
    
    with col_away:
        st.markdown(f"### ✈️ {away_team}")
        
        pos = get_team_position(away_team)
        form_letters, form_icons, form_points = get_team_form(away_team)
        
        st.markdown(f"**Position:** {pos}")
        st.markdown(f"**Form:** {form_icons} ({form_letters})")
        st.markdown(f"**Points (L5):** {form_points}")
        
        st.markdown("---")
        st.markdown("**Last 5 Matches:**")
        
        away_xg_for = get_latest_team_stat(away_team, "rolling_xg_for")
        away_xg_ag = get_latest_team_stat(away_team, "rolling_xg_against")
        away_gd = get_latest_team_stat(away_team, "rolling_GD")
        away_fin = get_latest_team_stat(away_team, "rolling_finishing_overperf")
        away_def = get_latest_team_stat(away_team, "rolling_def_overperf")
        
        st.markdown(f"• xG For: **{away_xg_for:.2f}**")
        st.markdown(f"• xG Against: **{away_xg_ag:.2f}**")
        st.markdown(f"• Goal Diff: **{away_gd:+.0f}**")
        st.markdown(f"• Finishing: **{away_fin:+.2f}**")
        st.markdown(f"• Defence: **{away_def:+.2f}**")
        
        if away_fin > 1:
            st.success("💚 Hot finishing streak")
        elif away_fin < -1:
            st.warning("⚠️ Cold finishing streak")
        
        if away_def > 1:
            st.success("💚 Solid defensive form")
        elif away_def < -1:
            st.warning("⚠️ Leaky defence")
    
    st.markdown("---")
    st.markdown("### 📊 Head-to-Head")
    
    home_form_letters, home_form_icons, home_form_points = get_team_form(home_team)
    away_form_letters, away_form_icons, away_form_points = get_team_form(away_team)
    
    comparison_df = pd.DataFrame({
        "Metric": [
            "xG For (L5)",
            "xG Against (L5)",
            "Goal Diff (L5)",
            "Points (L5)",
            "Finishing Over/Under",
            "Defence Over/Under"
        ],
        home_team: [
            f"{home_xg_for:.2f}",
            f"{home_xg_ag:.2f}",
            f"{home_gd:+.0f}",
            f"{home_form_points}",
            f"{home_fin:+.2f}",
            f"{home_def:+.2f}"
        ],
        away_team: [
            f"{away_xg_for:.2f}",
            f"{away_xg_ag:.2f}",
            f"{away_gd:+.0f}",
            f"{away_form_points}",
            f"{away_fin:+.2f}",
            f"{away_def:+.2f}"
        ]
    })
    
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    st.caption(
        "💡 **Finishing > 0:** Scoring above xG | **Defence > 0:** Conceding below xGA"
    )

# Model Agreement
with st.expander("🔍 Model Agreement Analysis", expanded=False):
    st.markdown("### Model Consensus")
    
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
    
    st.markdown(f"• **Logistic:** {outcome_names[log_favorite]} ({pct(log_probs.get(log_favorite, 0))})")
    st.markdown(f"• **Poisson:** {outcome_names[poisson_favorite]}")
    st.markdown(f"• **Dixon-Coles:** {outcome_names[dc_favorite]}")
    
    all_agree = (log_favorite == poisson_favorite == dc_favorite)
    
    if all_agree:
        st.success(
            f'✅ **Strong Consensus:** All models agree on **{outcome_names[log_favorite]}**'
        )
    else:
        st.warning(
            '⚠️ **Model Disagreement:** Consider prediction uncertainty'
        )
    
    st.markdown("---")
    st.markdown("### Probability Comparison")
    
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

# Context Impact
if abs(context_adj) > 0.01:
    with st.expander("⚙️ Context Adjustment Impact", expanded=False):
        st.markdown("### How Adjustments Changed Predictions")
        
        st.markdown(f"**Adjustment Factor:** {context_adj:+.3f}")
        
        st.markdown("**Expected Goals:**")
        st.markdown(f"• Home xG: {lam_h_base:.2f} → {lam_h_adj:.2f} ({lam_h_adj - lam_h_base:+.2f})")
        st.markdown(f"• Away xG: {lam_a_base:.2f} → {lam_a_adj:.2f} ({lam_a_adj - lam_a_base:+.2f})")
        
        st.markdown("---")
        st.markdown("**Match Outcome (Dixon-Coles):**")
        
        delta_h = (dc_adj["P_home"] - dc_base["P_home"]) * 100
        delta_d = (dc_adj["P_draw"] - dc_base["P_draw"]) * 100
        delta_a = (dc_adj["P_away"] - dc_base["P_away"]) * 100
        
        st.markdown(f"• Home Win: {pct(dc_base['P_home'])} → {pct(dc_adj['P_home'])} ({delta_h:+.1f}pp)")
        st.markdown(f"• Draw: {pct(dc_base['P_draw'])} → {pct(dc_adj['P_draw'])} ({delta_d:+.1f}pp)")
        st.markdown(f"• Away Win: {pct(dc_base['P_away'])} → {pct(dc_adj['P_away'])} ({delta_a:+.1f}pp)")
        
        st.markdown("---")
        st.markdown("**Goals Markets:**")
        
        delta_btts = (dc_adj["P_BTTS"] - dc_base["P_BTTS"]) * 100
        delta_o25 = (dc_adj["P_over_2_5"] - dc_base["P_over_2_5"]) * 100
        
        st.markdown(f"• BTTS: {pct(dc_base['P_BTTS'])} → {pct(dc_adj['P_BTTS'])} ({delta_btts:+.1f}pp)")
        st.markdown(f"• Over 2.5: {pct(dc_base['P_over_2_5'])} → {pct(dc_adj['P_over_2_5'])} ({delta_o25:+.1f}pp)")

# Footer
st.markdown('''
<div class="footer-section">
    <div class="footer-text">
        <strong>⚠️ Disclaimer</strong><br>
        Predictions are statistical models based on historical data. Football is inherently unpredictable.<br>
        Many factors (injuries, weather, motivation, referee decisions) cannot be fully modeled.<br><br>
        <strong>📊 Models:</strong> Logistic (outcome-based) • Poisson (goal-based) • Dixon-Coles (correlation-adjusted)<br>
        <strong>🔄 Updates:</strong> Models retrained weekly with latest match data
    </div>
</div>
''', unsafe_allow_html=True)