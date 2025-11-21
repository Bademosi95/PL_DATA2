import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy.stats import poisson
from pathlib import Path
from datetime import datetime
import json

# ═══════════════════════════════════════════════════════════════════
# 1. CONFIGURATION & MODERN DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="PL Match Predictor",
    page_icon="⚽",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Modern CSS (Apple/Google Aesthetic)
MODERN_CSS = """
<style>
    /* Global Font & Reset */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        background-color: #F5F5F7;
        font-family: 'Inter', sans-serif;
        color: #1D1D1F;
    }
    
    /* Hide Default Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Card Component */
    .css-card {
        background-color: #FFFFFF;
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border: 1px solid #FFFFFF;
    }
    
    /* Headings */
    h1, h2, h3 { font-weight: 700; letter-spacing: -0.5px; color: #1D1D1F; }
    h4 { font-weight: 600; font-size: 16px; color: #1D1D1F; margin-top: 0; }
    
    /* Custom Metrics */
    .sub-label {
        color: #86868B;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .big-stat {
        color: #1D1D1F;
        font-size: 26px;
        font-weight: 700;
    }
    
    /* Progress Bar Custom */
    .prog-container { margin-bottom: 16px; }
    .prog-header { display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 14px; font-weight: 600; }
    .prog-bg { background: #F2F2F7; border-radius: 8px; height: 10px; width: 100%; overflow: hidden; }
    .prog-fill { height: 100%; border-radius: 8px; transition: width 0.6s ease; }
    
    /* VS Header */
    .vs-container { display: flex; align-items: center; justify-content: space-between; padding: 10px 0; }
    .team-name { font-size: 22px; font-weight: 700; color: #1D1D1F; }
    .vs-badge { background: #E5E5EA; color: #86868B; padding: 6px 14px; border-radius: 20px; font-size: 12px; font-weight: 700; }
    
    /* Form Dots */
    .form-dot { height: 10px; width: 10px; border-radius: 50%; display: inline-block; margin-right: 5px; }
    .win { background-color: #34C759; }   /* Green */
    .draw { background-color: #FFCC00; }  /* Yellow */
    .loss { background-color: #FF3B30; }  /* Red */
    
    /* Alerts */
    .consensus-box { background: #E8F5E9; border: 1px solid #C8E6C9; border-radius: 12px; padding: 12px; color: #2E7D32; font-size: 14px; font-weight: 500; }
    .conflict-box { background: #FFF3E0; border: 1px solid #FFE0B2; border-radius: 12px; padding: 12px; color: #EF6C00; font-size: 14px; font-weight: 500; }

    /* Streamlit Tweaks */
    div[data-testid="stExpander"] { border: none; background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }
</style>
"""
st.markdown(MODERN_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# 2. ROBUST DATA LOADING & LOGIC (FROM GW11_SS.py)
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    """Robust loading with error handling"""
    required = ["pipe_result_final.pkl", "poisson_model.pkl", "long_df.pkl", "stats.pkl", "rho_hat.pkl"]
    for f in required:
        if not Path(f).exists():
            st.error(f"❌ Missing {f}. Run weekly_update.py."); st.stop()
    
    with open("pipe_result_final.pkl", "rb") as f: pipe = pickle.load(f)
    with open("poisson_model.pkl", "rb") as f: pois = pickle.load(f)
    with open("long_df.pkl", "rb") as f: ldf = pickle.load(f)
    with open("stats.pkl", "rb") as f: stats = pickle.load(f)
    with open("rho_hat.pkl", "rb") as f: rho = pickle.load(f)
    
    try:
        with open("feature_cols.pkl", "rb") as f: fcols = pickle.load(f)
    except:
        fcols = ["strength_diff", "defense_diff", "rolling_points_diff", "rolling_xG_diff", 
                 "rolling_xGA_diff", "rolling_GD_diff", "finishing_overperf_diff", "def_overperf_diff"]

    stats["Squad"] = stats["Squad"].astype(str).str.strip()
    ldf["team"] = ldf["team"].astype(str).str.strip()
    
    # Metadata
    meta = {}
    if Path("metadata.json").exists():
        with open("metadata.json", "r") as f: meta = json.load(f)

    return pipe, pois, ldf, stats, rho, fcols, meta

pipe_result_final, poisson_model, long_df, stats, rho_hat, feature_cols, metadata = load_models()

# --- Helper Maths ---
def pct(x, decimals=1): return f"{x * 100:.{decimals}f}%"
def safe_div(n, d): return n / d if d != 0 else 0.0

def get_latest_stat(team, col, default=0.0):
    df = long_df[long_df["team"] == team].sort_values("Date")
    return float(df.iloc[-1][col]) if not df.empty else default

def get_strength(team):
    r = stats[stats["Squad"] == team]
    if r.empty: return None
    r = r.iloc[0]
    return {
        "att_h": safe_div(r["Home_xG"], r["Home_MP"]), "def_h": safe_div(r["Home_xGA"], r["Home_MP"]),
        "att_a": safe_div(r["Away_xG"], r["Away_MP"]), "def_a": safe_div(r["Away_xGA"], r["Away_MP"])
    }

def build_features(h, a):
    sh, sa = get_strength(h), get_strength(a)
    if not sh or not sa: return None
    
    def gl(t, c): return get_latest_stat(t, c)
    
    data = {
        "strength_diff": sh["att_h"] - sa["att_a"],
        "defense_diff": sa["def_a"] - sh["def_h"],
        "rolling_points_diff": gl(h, "rolling_points") - gl(a, "rolling_points"),
        "rolling_xG_diff": gl(h, "rolling_xg_for") - gl(a, "rolling_xg_for"),
        "rolling_xGA_diff": gl(h, "rolling_xg_against") - gl(a, "rolling_xg_against"),
        "rolling_GD_diff": gl(h, "rolling_GD") - gl(a, "rolling_GD"),
        "finishing_overperf_diff": gl(h, "rolling_finishing_overperf") - gl(a, "rolling_finishing_overperf"),
        "def_overperf_diff": gl(h, "rolling_def_overperf") - gl(a, "rolling_def_overperf")
    }
    # Handle missing columns gracefully by reindexing
    df = pd.DataFrame([data])
    for c in feature_cols:
        if c not in df.columns: df[c] = 0.0
    return df[feature_cols]

def predict_logistic(h, a):
    X = build_features(h, a)
    if X is None: return None
    probs = pipe_result_final.predict_proba(X)[0]
    return dict(zip(pipe_result_final.classes_, probs))

def get_poisson_lambdas(h, a, adj=0.0):
    p_df = pd.DataFrame({"team":[h, a], "opponent":[a, h], "is_home":[1, 0]})
    preds = poisson_model.predict(p_df)
    lh_b, la_b = float(preds.iloc[0]), float(preds.iloc[1])
    return max(lh_b*(1+adj), 0.01), max(la_b*(1-adj), 0.01), lh_b, la_b

def dixon_coles_tau(h, a, lh, la, rho):
    if h==0 and a==0: return 1 - (lh*la*rho)
    if h==0 and a==1: return 1 + la*rho
    if h==1 and a==0: return 1 + lh*rho
    if h==1 and a==1: return 1 - rho
    return 1.0

def compute_probs(lh, la, use_dc=False, rho=0.0):
    hg, ag = np.arange(0, 7), np.arange(0, 7)
    P = np.outer(poisson.pmf(hg, lh), poisson.pmf(ag, la))
    if use_dc:
        for i in hg:
            for j in ag:
                P[i, j] *= dixon_coles_tau(i, j, lh, la, rho)
    P /= P.sum()
    
    total = hg[:, None] + ag[None, :]
    return {
        "H": float(np.tril(P, -1).sum()),
        "D": float(np.trace(P)),
        "A": float(np.triu(P, 1).sum()),
        "BTTS": float(P[(hg[:, None]>0) & (ag[None, :]>0)].sum()),
        "O25": float(P[total>2.5].sum()),
        "CS_H": float(P[:, 0].sum()),
        "CS_A": float(P[0, :].sum())
    }

def get_form_html(team):
    matches = long_df[long_df["team"]==team].sort_values("Date").tail(5)
    html = ""
    for _, m in matches.iterrows():
        cls = "win" if m["goals_for"] > m["goals_against"] else ("draw" if m["goals_for"] == m["goals_against"] else "loss")
        html += f'<span class="form-dot {cls}"></span>'
    return html

# ═══════════════════════════════════════════════════════════════════
# 3. UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════

def draw_bar(label, value, color, delta=None):
    delta_html = ""
    if delta is not None:
        col = "#34C759" if delta > 0 else "#FF3B30"
        delta_html = f"<span style='color:{col}; font-size:12px; margin-left:5px;'>({delta:+.1f}%)</span>"
        
    st.markdown(f"""
    <div class="prog-container">
        <div class="prog-header">
            <span>{label} {delta_html}</span>
            <span>{pct(value)}</span>
        </div>
        <div class="prog-bg">
            <div class="prog-fill" style="width:{value*100}%; background-color:{color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def stat_box(label, value, subtext=""):
    st.markdown(f"""
    <div style="text-align:center;">
        <div class="sub-label">{label}</div>
        <div class="big-stat">{value}</div>
        <div style="color:#86868B; font-size:12px;">{subtext}</div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# 4. APPLICATION LAYOUT
# ═══════════════════════════════════════════════════════════════════

# --- SIDEBAR (Functionality Restored) ---
with st.sidebar:
    st.markdown("### ⚽ Match Setup")
    teams = sorted(stats["Squad"].unique())
    home_team = st.selectbox("Home Team", teams, index=0)
    away_team = st.selectbox("Away Team", teams, index=1)
    
    st.markdown("---")
    st.markdown("### 🧠 Context Engine")
    
    adv_mode = st.checkbox("Advanced Mode", value=False)
    
    if not adv_mode:
        raw = st.slider("Advantage Tilt", -3.0, 3.0, 0.0, 0.1, help="Left=Away, Right=Home")
        context_adj = raw / 10.0
    else:
        st.caption("Fine-tune specific factors:")
        att = st.slider("Attack Power", -3.0, 3.0, 0.0, 0.1)
        defe = st.slider("Defense Solidity", -3.0, 3.0, 0.0, 0.1)
        mor = st.slider("Morale / Form", -3.0, 3.0, 0.0, 0.1)
        context_adj = (att + defe + (0.5 * mor)) / 20.0
        
    if context_adj != 0:
        st.info(f"Adjustment Active: {context_adj:+.1%}")
    
    st.caption(f"Dixon-Coles ρ: {rho_hat:.3f}")

if home_team == away_team: st.warning("Select different teams."); st.stop()

# --- RUN CALCULATIONS ---
log_res = predict_logistic(home_team, away_team)
lam_h, lam_a, base_h, base_a = get_poisson_lambdas(home_team, away_team, context_adj)
dc_res = compute_probs(lam_h, lam_a, use_dc=True, rho=rho_hat)
dc_base = compute_probs(base_h, base_a, use_dc=True, rho=rho_hat) # For deltas
pois_res = compute_probs(lam_h, lam_a, use_dc=False)

# --- SECTION 1: MATCH HEADER ---
st.markdown(f"""
<div class="css-card">
    <div class="vs-container">
        <div style="width:40%;">
            <div class="team-name">{home_team}</div>
            <div style="margin-top:8px;">{get_form_html(home_team)}</div>
        </div>
        <div class="vs-badge">VS</div>
        <div style="width:40%; text-align:right;">
            <div class="team-name">{away_team}</div>
            <div style="margin-top:8px;">{get_form_html(away_team)}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- SECTION 2: PREDICTION DISPLAY ---
col_main, col_side = st.columns([1.6, 1])

with col_main:
    with st.container():
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown("<h4>🏆 Match Outcome (Dixon-Coles)</h4>", unsafe_allow_html=True)
        
        # Calculate deltas from context adjustment
        d_h = (dc_res["H"] - dc_base["H"])*100
        d_a = (dc_res["A"] - dc_base["A"])*100
        
        draw_bar(f"{home_team}", dc_res["H"], "#007AFF", delta=d_h if abs(d_h)>0.1 else None)
        draw_bar("Draw", dc_res["D"], "#8E8E93")
        draw_bar(f"{away_team}", dc_res["A"], "#FF3B30", delta=d_a if abs(d_a)>0.1 else None)
        st.markdown('</div>', unsafe_allow_html=True)

with col_side:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    stat_box("Expected Goals", f"{lam_h:.2f} - {lam_a:.2f}", "Context Adjusted")
    st.markdown("<hr style='opacity:0.1; margin:15px 0;'>", unsafe_allow_html=True)
    stat_box("BTTS Prob", pct(dc_res["BTTS"]), "Both Score")
    st.markdown("<hr style='opacity:0.1; margin:15px 0;'>", unsafe_allow_html=True)
    stat_box("Over 2.5 Goals", pct(dc_res["O25"]), "High Scoring")
    st.markdown('</div>', unsafe_allow_html=True)

# --- SECTION 3: DETAILED ANALYSIS TABS ---
tab_models, tab_deep, tab_agree = st.tabs(["📊 Model Comparison", "📈 Deep Stats", "🔍 Consensus"])

# TAB 1: MODEL COMPARISON (Functionality Restored)
with tab_models:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown("<h4>Model Probability Matrix</h4>", unsafe_allow_html=True)
    
    comp_df = pd.DataFrame({
        "Outcome": ["Home Win", "Draw", "Away Win"],
        "Logistic (Form)": [pct(log_res.get("H",0)), pct(log_res.get("D",0)), pct(log_res.get("A",0))],
        "Poisson (xG)": [pct(pois_res["H"]), pct(pois_res["D"]), pct(pois_res["A"])],
        "Dixon-Coles (Final)": [pct(dc_res["H"]), pct(dc_res["D"]), pct(dc_res["A"])]
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    
    st.caption("""
    • **Logistic:** Ignores goals, looks at W/D/L form & points.
    • **Poisson:** Pure Expected Goals calculation.
    • **Dixon-Coles:** Corrects Poisson for low-scoring draw biases (rho).
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 2: DEEP STATS (Visualized)
with tab_deep:
    c1, c2 = st.columns(2)
    
    h_xg = get_latest_stat(home_team, "rolling_xg_for")
    a_xg = get_latest_stat(away_team, "rolling_xg_for")
    h_fin = get_latest_stat(home_team, "rolling_finishing_overperf")
    a_fin = get_latest_stat(away_team, "rolling_finishing_overperf")
    
    with c1:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown(f"#### {home_team}")
        st.write(f"**xG (5 games):** {h_xg:.2f}")
        st.write(f"**Finishing:** {h_fin:+.2f}")
        st.progress(min(max(h_xg/12, 0), 1))
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c2:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown(f"#### {away_team}")
        st.write(f"**xG (5 games):** {a_xg:.2f}")
        st.write(f"**Finishing:** {a_fin:+.2f}")
        st.progress(min(max(a_xg/12, 0), 1))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Comparison Table
    st.markdown("#### Head-to-Head Metrics")
    h2h_df = pd.DataFrame({
        "Metric": ["xG For", "xG Against", "Goal Diff", "Points (5)"],
        home_team: [f"{h_xg:.2f}", f"{get_latest_stat(home_team, 'rolling_xg_against'):.2f}", 
                   f"{get_latest_stat(home_team, 'rolling_GD'):.0f}", f"{get_latest_stat(home_team, 'rolling_points'):.0f}"],
        away_team: [f"{a_xg:.2f}", f"{get_latest_stat(away_team, 'rolling_xg_against'):.2f}", 
                   f"{get_latest_stat(away_team, 'rolling_GD'):.0f}", f"{get_latest_stat(away_team, 'rolling_points'):.0f}"]
    })
    st.table(h2h_df)

# TAB 3: CONSENSUS & IMPACT (Logic Restored)
with tab_agree:
    # Calculate favorites
    log_fav = max(log_res, key=log_res.get)
    dc_fav_key = max(["H","D","A"], key=lambda k: dc_res[k])
    
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown("<h4>Consensus Check</h4>", unsafe_allow_html=True)
    
    if log_fav == dc_fav_key:
        st.markdown(f'<div class="consensus-box">✅ Strong Consensus: Both Form (Logistic) and xG (Dixon-Coles) favor {home_team if log_fav=="H" else (away_team if log_fav=="A" else "Draw")}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="conflict-box">⚠️ Divergence: Form model favors {log_fav}, but xG model favors {dc_fav_key}. Caution advised.</div>', unsafe_allow_html=True)
    
    if abs(context_adj) > 0.01:
        st.markdown("#### Context Impact")
        st.write(f"Your adjustments moved the Home Win probability by **{(dc_res['H'] - dc_base['H'])*100:+.1f}%**.")
        
    st.markdown('</div>', unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("---")
st.caption("Data updated weekly. Predictions are estimates.")