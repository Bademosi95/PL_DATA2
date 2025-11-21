import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy.stats import poisson
from pathlib import Path
from datetime import datetime
import json

# ═══════════════════════════════════════════════════════════════════
# 1. PAGE CONFIG & MODERN CSS
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="PL Match Predictor",
    page_icon="⚽",
    layout="centered", # Centered looks more "App-like" on desktop
    initial_sidebar_state="collapsed" # Cleaner start
)

# Apple/Google Style Design System
MODERN_CSS = """
<style>
    /* Global Reset & Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        background-color: #F5F5F7; /* Apple Light Gray */
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Card Container Style */
    .css-card {
        background-color: #FFFFFF;
        border-radius: 18px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.04);
        margin-bottom: 20px;
        border: 1px solid #FFFFFF;
        transition: transform 0.2s ease;
    }
    .css-card:hover {
        border-color: #E5E5EA;
    }

    /* Typography */
    h1, h2, h3 {
        color: #1D1D1F;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .sub-label {
        color: #86868B;
        font-size: 13px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .big-stat {
        color: #1D1D1F;
        font-size: 28px;
        font-weight: 700;
    }
    .delta-pos { color: #34C759; font-size: 14px; font-weight: 600; }
    .delta-neg { color: #FF3B30; font-size: 14px; font-weight: 600; }
    .delta-neutral { color: #86868B; font-size: 14px; font-weight: 600; }

    /* Custom Progress Bar */
    .prog-bg {
        background-color: #F2F2F7;
        border-radius: 8px;
        height: 8px;
        width: 100%;
        margin-top: 8px;
        overflow: hidden;
    }
    .prog-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.5s ease;
    }
    
    /* Team Header */
    .vs-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 10px;
    }
    .team-name {
        font-size: 24px;
        font-weight: 700;
        color: #1D1D1F;
    }
    .vs-badge {
        background: #E5E5EA;
        color: #86868B;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }

    /* Form Badges */
    .form-dot {
        height: 10px;
        width: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 4px;
    }
    .win { background-color: #34C759; }
    .draw { background-color: #FFCC00; }
    .loss { background-color: #FF3B30; }

    /* Streamlit Overrides */
    div[data-testid="stExpander"] {
        border: none;
        background: white;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
</style>
"""
st.markdown(MODERN_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# 2. LOGIC & DATA (UNCHANGED)
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    # [Logic kept exactly as provided]
    required_files = ["pipe_result_final.pkl", "poisson_model.pkl", "long_df.pkl", "stats.pkl", "rho_hat.pkl"]
    for file in required_files:
        if not Path(file).exists():
            st.error(f"Missing file: {file}. Please run weekly_update.py"); st.stop()
    
    with open("pipe_result_final.pkl", "rb") as f: pipe_result = pickle.load(f)
    with open("poisson_model.pkl", "rb") as f: poisson_model = pickle.load(f)
    with open("long_df.pkl", "rb") as f: long_df = pickle.load(f)
    with open("stats.pkl", "rb") as f: stats = pickle.load(f)
    with open("rho_hat.pkl", "rb") as f: rho_hat = pickle.load(f)
    
    if Path("feature_cols.pkl").exists():
        with open("feature_cols.pkl", "rb") as f: feature_cols = pickle.load(f)
    else:
        feature_cols = ["strength_diff", "defense_diff", "rolling_points_diff", "rolling_xG_diff", 
                        "rolling_xGA_diff", "rolling_GD_diff", "finishing_overperf_diff", "def_overperf_diff"]

    stats["Squad"] = stats["Squad"].astype(str).str.strip()
    long_df["team"] = long_df["team"].astype(str).str.strip()
    
    return pipe_result, poisson_model, long_df, stats, rho_hat, feature_cols

pipe_result_final, poisson_model, long_df, stats, rho_hat, feature_cols = load_models()

def pct(x, decimals=1): return f"{x * 100:.{decimals}f}%"

def get_latest_team_stat(team, column, default=0.0):
    try:
        team_data = long_df[long_df["team"] == team].sort_values("Date")
        if team_data.empty: return default
        return float(team_data.iloc[-1].get(column, default))
    except: return default

def get_team_strength_metrics(team):
    row = stats[stats["Squad"] == team]
    if row.empty: return None
    row = row.iloc[0]
    # Handle potential zero division safely
    def safe_div(a, b): return a/b if b!=0 else 0
    return {
        "att_home": safe_div(row["Home_xG"], row["Home_MP"]),
        "def_home": safe_div(row["Home_xGA"], row["Home_MP"]),
        "att_away": safe_div(row["Away_xG"], row["Away_MP"]),
        "def_away": safe_div(row["Away_xGA"], row["Away_MP"])
    }

def build_feature_vector(home_team, away_team):
    h_s = get_team_strength_metrics(home_team)
    a_s = get_team_strength_metrics(away_team)
    if not h_s or not a_s: return None
    
    # Helper to get rolling
    def gr(t, c): return get_latest_team_stat(t, c)
    
    features = {
        "strength_diff": h_s["att_home"] - a_s["att_away"],
        "defense_diff": a_s["def_away"] - h_s["def_home"],
        "rolling_points_diff": gr(home_team, "rolling_points") - gr(away_team, "rolling_points"),
        "rolling_xG_diff": gr(home_team, "rolling_xg_for") - gr(away_team, "rolling_xg_for"),
        "rolling_xGA_diff": gr(home_team, "rolling_xg_against") - gr(away_team, "rolling_xg_against"),
        "rolling_GD_diff": gr(home_team, "rolling_GD") - gr(away_team, "rolling_GD"),
        "finishing_overperf_diff": gr(home_team, "rolling_finishing_overperf") - gr(away_team, "rolling_finishing_overperf"),
        "def_overperf_diff": gr(home_team, "rolling_def_overperf") - gr(away_team, "rolling_def_overperf")
    }
    return pd.DataFrame([features])[feature_cols]

def predict_logistic(home_team, away_team):
    X = build_feature_vector(home_team, away_team)
    if X is None: return None
    probs = pipe_result_final.predict_proba(X)[0]
    return dict(zip(pipe_result_final.classes_, probs))

def get_poisson_lambdas(home_team, away_team, context_adj=0.0):
    pred_df = pd.DataFrame({"team": [home_team, away_team], "opponent": [away_team, home_team], "is_home": [1, 0]})
    preds = poisson_model.predict(pred_df)
    lam_h_base, lam_a_base = float(preds.iloc[0]), float(preds.iloc[1])
    return max(lam_h_base * (1 + context_adj), 0.01), max(lam_a_base * (1 - context_adj), 0.01), lam_h_base, lam_a_base

def dixon_coles_tau(hg, ag, lam_h, lam_a, rho):
    if hg == 0 and ag == 0: return 1 - (lam_h * lam_a * rho)
    if hg == 0 and ag == 1: return 1 + lam_a * rho
    if hg == 1 and ag == 0: return 1 + lam_h * rho
    if hg == 1 and ag == 1: return 1 - rho
    return 1.0

def compute_scoreline_probabilities(lam_home, lam_away, use_dc=False, rho=0.0):
    hg, ag = np.arange(0, 7), np.arange(0, 7)
    P = np.outer(poisson.pmf(hg, lam_home), poisson.pmf(ag, lam_away))
    if use_dc:
        for i, h in enumerate(hg):
            for j, a in enumerate(ag):
                P[i, j] *= dixon_coles_tau(h, a, lam_home, lam_away, rho)
    P /= P.sum()
    
    total_goals = hg[:, None] + ag[None, :]
    return {
        "P_home": float(np.tril(P, -1).sum()),
        "P_draw": float(np.trace(P)),
        "P_away": float(np.triu(P, 1).sum()),
        "P_BTTS": float(P[(hg[:, None] > 0) & (ag[None, :] > 0)].sum()),
        "P_over_2_5": float(P[total_goals > 2].sum()),
        "P_CS_Home": float(P[:, 0].sum()),
        "P_CS_Away": float(P[0, :].sum())
    }

def get_form_html(team):
    df = long_df[long_df["team"] == team].sort_values("Date").tail(5)
    html = ""
    for _, r in df.iterrows():
        cls = "win" if r["goals_for"] > r["goals_against"] else ("draw" if r["goals_for"] == r["goals_against"] else "loss")
        html += f'<span class="form-dot {cls}"></span>'
    return html

# ═══════════════════════════════════════════════════════════════════
# 3. UI COMPONENTS (NEW VISUAL LAYER)
# ═══════════════════════════════════════════════════════════════════

def draw_progress_bar(label, value, color="#007AFF"):
    """Renders a minimal HTML progress bar"""
    st.markdown(f"""
    <div style="margin-bottom: 12px;">
        <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
            <span style="font-weight:600; font-size:14px; color:#1D1D1F;">{label}</span>
            <span style="font-weight:600; font-size:14px; color:#1D1D1F;">{value*100:.1f}%</span>
        </div>
        <div class="prog-bg">
            <div class="prog-fill" style="width:{value*100}%; background-color:{color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def card_stat(label, value, subtext=None, highlight=False):
    color = "#007AFF" if highlight else "#1D1D1F"
    html = f"""
    <div style="text-align:center;">
        <div class="sub-label">{label}</div>
        <div class="big-stat" style="color:{color}">{value}</div>
        {f'<div style="font-size:12px; color:#86868B;">{subtext}</div>' if subtext else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# 4. APP LAYOUT
# ═══════════════════════════════════════════════════════════════════

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### Match Settings")
    teams = sorted(stats["Squad"].unique())
    home_team = st.selectbox("Home Team", teams, index=0)
    away_team = st.selectbox("Away Team", teams, index=1)
    
    st.markdown("### Context")
    with st.expander("Adjust Game State", expanded=True):
        adj_val = st.slider("Tilt Advantage", -3, 3, 0, help="Left = Away Adv, Right = Home Adv")
        context_adj = adj_val / 10.0
        if adj_val != 0:
            st.caption(f"Applying {context_adj:+.1%} adjustment to xG.")
            
    st.markdown("---")
    st.info("Model refreshes weekly.\nRho: " + f"{rho_hat:.3f}")

if home_team == away_team:
    st.warning("Select different teams."); st.stop()

# --- CALCULATE ---
log_probs = predict_logistic(home_team, away_team)
lam_h, lam_a, base_h, base_a = get_poisson_lambdas(home_team, away_team, context_adj)
dc_res = compute_scoreline_probabilities(lam_h, lam_a, use_dc=True, rho=rho_hat)
pois_res = compute_scoreline_probabilities(lam_h, lam_a, use_dc=False)

# --- HEADER SECTION ---
st.markdown(f"""
<div class="css-card">
    <div class="vs-container">
        <div style="text-align:left; width:40%;">
            <div class="team-name">{home_team}</div>
            <div style="margin-top:5px;">{get_form_html(home_team)}</div>
        </div>
        <div class="vs-badge">VS</div>
        <div style="text-align:right; width:40%;">
            <div class="team-name">{away_team}</div>
            <div style="margin-top:5px;">{get_form_html(away_team)}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- MAIN OUTCOME ---
st.markdown("### Prediction")
col_main, col_details = st.columns([1.5, 1])

with col_main:
    with st.container():
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        draw_progress_bar(f"{home_team} Win", dc_res['P_home'], "#007AFF") # Blue
        draw_progress_bar("Draw", dc_res['P_draw'], "#8E8E93")   # Gray
        draw_progress_bar(f"{away_team} Win", dc_res['P_away'], "#FF3B30") # Red
        
        st.markdown("<hr style='margin: 15px 0; opacity: 0.2;'>", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        with c1: card_stat("Most Likely", f"{pct(max(dc_res['P_home'], dc_res['P_draw'], dc_res['P_away']))}", "Confidence")
        with c2: card_stat("Exp. Goals", f"{lam_h+lam_a:.2f}", "Combined xG")
        with c3: card_stat("BTTS", f"{pct(dc_res['P_BTTS'])}", "Both Score")
        st.markdown('</div>', unsafe_allow_html=True)

with col_details:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown('<div class="sub-label">Projected Score</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="big-stat">{lam_h:.2f} - {lam_a:.2f}</div>', unsafe_allow_html=True)
    st.caption("Based on Poisson xG")
    
    st.markdown('<div style="height:15px"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sub-label">Clean Sheets</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; font-size:14px; margin-top:5px;">
        <span>{home_team}</span> <span>{pct(dc_res['P_CS_Home'])}</span>
    </div>
    <div style="display:flex; justify-content:space-between; font-size:14px; margin-top:5px;">
        <span>{away_team}</span> <span>{pct(dc_res['P_CS_Away'])}</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- TABS FOR ANALYSIS ---
tab1, tab2 = st.tabs(["📊 Deep Dive", "📈 Model Consensus"])

with tab1:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown("#### Rolling Form (Last 5 Games)")
    
    h_row = get_latest_team_stat(home_team, "rolling_xg_for", 0)
    a_row = get_latest_team_stat(away_team, "rolling_xg_for", 0)
    h_fin = get_latest_team_stat(home_team, "rolling_finishing_overperf", 0)
    a_fin = get_latest_team_stat(away_team, "rolling_finishing_overperf", 0)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**{home_team}**")
        st.progress(min(h_row/15.0, 1.0)) # Normalize roughly
        st.caption(f"Attack Power: {h_row:.2f} xG")
        
        fin_color = "green" if h_fin > 0 else "red"
        st.markdown(f"Finishing: <span style='color:{fin_color}'>{h_fin:+.2f}</span>", unsafe_allow_html=True)

    with col_b:
        st.markdown(f"**{away_team}**")
        st.progress(min(a_row/15.0, 1.0))
        st.caption(f"Attack Power: {a_row:.2f} xG")
        
        fin_color = "green" if a_fin > 0 else "red"
        st.markdown(f"Finishing: <span style='color:{fin_color}'>{a_fin:+.2f}</span>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    # Comparison Table styled
    st.markdown("#### Model Probabilities")
    
    comp_df = pd.DataFrame({
        "Model": ["Logistic (Baseline)", "Poisson (xG)", "Dixon-Coles (Adjusted)"],
        "Home Win": [pct(log_probs.get("H",0)), pct(pois_res['P_home']), pct(dc_res['P_home'])],
        "Draw": [pct(log_probs.get("D",0)), pct(pois_res['P_draw']), pct(dc_res['P_draw'])],
        "Away Win": [pct(log_probs.get("A",0)), pct(pois_res['P_away']), pct(dc_res['P_away'])]
    })
    st.dataframe(comp_df, hide_index=True, use_container_width=True)
    
    if abs(context_adj) > 0:
        st.caption(f"Note: Poisson and Dixon-Coles include your {context_adj:+.0%} adjustment. Logistic does not.")
    st.markdown('</div>', unsafe_allow_html=True)