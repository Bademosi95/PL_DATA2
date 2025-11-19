"""Streamlit dashboard for a weekly-updated Premier League match predictor.

The app uses a pre-trained pipeline (``pipe_result_final.pkl``) alongside a
poisson goal model and Dixon-Coles adjustment to estimate match outcome and
popular betting markets. It intentionally keeps all logic within a single file
to simplify deployment, but still benefits from a few guard rails and
defensive checks to make weekly refreshes smoother.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson

# ==============================
# 1. PAGE CONFIG & CSS
# ==============================
st.set_page_config(
    page_title="PL Match Predictor",
    page_icon="⚽",
    layout="centered"  # 'centered' is often easier to read than 'wide' for data
)

# CSS for clean "Card" look and colored form guides
st.markdown("""
<style>
    /* Main Titles */
    .match-title {
        font-size: 32px;
        font-weight: 800;
        text-align: center;
        color: #ffffff;
        margin-bottom: 5px;
    }
    .vs-text {
        font-size: 18px;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 25px;
    }
    
    /* KPI Cards */
    div[data-testid="stMetric"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    div[data-testid="stMetricLabel"] {
        display: flex;
        justify-content: center;
    }

    /* Form Badges */
    .form-box {
        display: inline-block;
        width: 20px;
        height: 20px;
        margin-right: 3px;
        border-radius: 3px;
        text-align: center;
        font-size: 12px;
        font-weight: bold;
        color: white;
        line-height: 20px;
    }
    .w { background-color: #22c55e; } /* Green */
    .d { background-color: #eab308; } /* Yellow */
    .l { background-color: #ef4444; } /* Red */
    
    /* Section Headers */
    .section-header {
        font-size: 20px;
        font-weight: 600;
        margin-top: 30px;
        margin-bottom: 10px;
        border-bottom: 1px solid #334155;
        padding-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# 2. DATA LOADING
# ==============================
@st.cache_resource
def load_data() -> tuple[Optional[object], Optional[object], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[float], list[str]]:
    """Load all artifacts required for the dashboard.

    Using ``Path`` keeps paths robust to future refactors, and limiting the
    ``except`` to ``FileNotFoundError`` avoids masking other issues (e.g.
    corrupt pickle files).
    """

    def _load_pickle(path: Path):
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("rb") as fh:
            return pickle.load(fh)

    try:
        pipe = _load_pickle(Path("pipe_result_final.pkl"))
        pois = _load_pickle(Path("poisson_model.pkl"))
        ldf = _load_pickle(Path("long_df.pkl"))
        sts = _load_pickle(Path("stats.pkl"))
        rho = _load_pickle(Path("rho_hat.pkl"))
    except FileNotFoundError as exc:
        st.error(f"⚠️ Missing data file: {exc.name if hasattr(exc, 'name') else exc}.")
        st.stop()

    fcols_path = Path("feature_cols.pkl")
    if fcols_path.exists():
        feature_cols = _load_pickle(fcols_path)
    else:
        feature_cols = [
            "strength_diff",
            "defense_diff",
            "rolling_points_diff",
            "rolling_xG_diff",
            "rolling_GD_diff",
            "finishing_overperf_diff",
        ]

    return pipe, pois, ldf, sts, rho, feature_cols


pipe_result_final, poisson_model, long_df, stats, rho_hat, feature_cols = load_data()

# Cleanup Strings
if "Squad" in stats.columns: stats["Squad"] = stats["Squad"].astype(str).str.strip()
long_df["team"] = long_df["team"].astype(str).str.strip()

# ==============================
# 3. CALCULATION LOGIC
# ==============================
def get_form_html(team: str) -> str:
    """Return HTML string of colored boxes for last five games for ``team``."""

    last_5 = long_df[long_df["team"] == team].sort_values("Date").tail(5)
    html = ""
    for _, r in last_5.iterrows():
        if r["goals_for"] > r["goals_against"]:
            html += '<span class="form-box w">W</span>'
        elif r["goals_for"] == r["goals_against"]:
            html += '<span class="form-box d">D</span>'
        else:
            html += '<span class="form-box l">L</span>'
    return html

def get_latest_row(team: str) -> pd.Series:
    """Return the most recent record for ``team``.

    Raises a clear error when weekly refresh fails to include a club, which is
    easier to debug than an ``IndexError`` later in the pipeline.
    """

    team_rows = long_df[long_df["team"] == team]
    if team_rows.empty:
        raise ValueError(f"Team '{team}' not found in `long_df`. Did the weekly update fail?")
    return team_rows.iloc[-1]

def run_predictions(home: str, away: str, adj: float) -> Dict[str, float]:
    """Generate outcome probabilities and common markets for the match."""

    # 1. Poisson Lambdas (xG)
    lam_h_base = float(
        poisson_model.predict(pd.DataFrame({"team": [home], "opponent": [away], "is_home": [1]})).iloc[0]
    )
    lam_a_base = float(
        poisson_model.predict(pd.DataFrame({"team": [away], "opponent": [home], "is_home": [0]})).iloc[0]
    )
    
    lam_h = max(lam_h_base * (1 + adj), 0.01)
    lam_a = max(lam_a_base * (1 - adj), 0.01)

    # 2. Score Matrix & Dixon-Coles
    max_g = 6
    hg = np.arange(0, max_g + 1)
    ag = np.arange(0, max_g + 1)
    P = poisson.pmf(hg[:, None], lam_h) * poisson.pmf(ag[None, :], lam_a)
    
    # Apply Rho Correction
    P[0,0] *= (1 - lam_h * lam_a * rho_hat)
    P[0,1] *= (1 + lam_a * rho_hat)
    P[1,0] *= (1 + lam_h * rho_hat)
    P[1,1] *= (1 - rho_hat)
    P /= P.sum()

    # 3. Extract Markets
    home_prob = np.tril(P, -1).sum()
    draw_prob = np.trace(P)
    away_prob = np.triu(P, 1).sum()
    
    # Totals
    total_goals = hg[:, None] + ag[None, :]
    over15 = P[total_goals > 1.5].sum()
    over25 = P[total_goals > 2.5].sum()
    btts   = P[(hg[:, None] > 0) & (ag[None, :] > 0)].sum()
    
    # Most Likely Score
    idx = np.unravel_index(np.argmax(P, axis=None), P.shape)
    top_score = f"{hg[idx[0]]}-{ag[idx[1]]}"
    
    return {
        "H": home_prob, "D": draw_prob, "A": away_prob,
        "xG_H": lam_h, "xG_A": lam_a,
        "O15": over15, "O25": over25, "BTTS": btts,
        "Score": top_score
    }

# ==============================
# 4. SIDEBAR (INPUTS)
# ==============================
with st.sidebar:
    st.header("Match Setup")
    teams = sorted(long_df["team"].unique())
    home_team = st.selectbox("Home Team", teams, index=0)
    away_team = st.selectbox("Away Team", teams, index=1)
    
    st.markdown("---")
    st.markdown("**Context Adjustment**")
    st.caption("Slide right if Home team has news (e.g., new manager boost). Slide left if Away team has advantage.")
    adj_val = st.slider("Home/Away Tilt", -3, 3, 0)
    context_adj = adj_val / 20.0

if home_team == away_team:
    st.warning("Select different teams.")
    st.stop()

# ==============================
# 5. MAIN DASHBOARD
# ==============================

# --- Header ---
st.markdown(f'<div class="match-title">{home_team} vs {away_team}</div>', unsafe_allow_html=True)
st.markdown('<div class="vs-text">Premier League Forecast</div>', unsafe_allow_html=True)

# --- Run Logic ---
try:
    preds = run_predictions(home_team, away_team, context_adj)
    h_row = get_latest_row(home_team)
    a_row = get_latest_row(away_team)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

# --- SECTION 1: THE HEADLINES (Big & Bold) ---
st.markdown('<div class="section-header">🏆 Match Outcome</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("HOME WIN", f"{preds['H']*100:.1f}%")
with col2:
    st.metric("DRAW", f"{preds['D']*100:.1f}%")
with col3:
    st.metric("AWAY WIN", f"{preds['A']*100:.1f}%")
with col4:
    st.metric("Top Score", preds['Score'], help="The single most probable scoreline")

# --- SECTION 2: MARKETS (Readable Grid) ---
st.markdown('<div class="section-header">📊 Key Markets</div>', unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
with m1:
    st.info(f"**Over 1.5 Goals**\n# {preds['O15']*100:.1f}%")
with m2:
    st.info(f"**Over 2.5 Goals**\n# {preds['O25']*100:.1f}%")
with m3:
    st.info(f"**BTTS (Both Score)**\n# {preds['BTTS']*100:.1f}%")

# --- SECTION 3: TABS FOR DETAILS (Keep the noise hidden) ---
tab_stats, tab_explain = st.tabs(["📈 Team Stats & Form", "📚 Explanation"])

with tab_stats:
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader(home_team)
        st.markdown(f"Form: {get_form_html(home_team)}", unsafe_allow_html=True)
        st.markdown(f"**xG (Last 5):** {h_row['rolling_xg_for']:.2f}")
        st.markdown(f"**Goals (Last 5):** {h_row['rolling_goals_for']:.0f}")
        
    with c2:
        st.subheader(away_team)
        st.markdown(f"Form: {get_form_html(away_team)}", unsafe_allow_html=True)
        st.markdown(f"**xG (Last 5):** {a_row['rolling_xg_for']:.2f}")
        st.markdown(f"**Goals (Last 5):** {a_row['rolling_goals_for']:.0f}")
    
    st.markdown("---")
    st.caption("Stats are based on the rolling sum of the last 5 Matchweeks.")

with tab_explain:
    st.markdown("### What am I looking at?")
    
    st.markdown("""
    **1. The Probabilities (Dixon-Coles Model)** We don't just look at average goals. We use a statistical method called **Dixon-Coles**.
    * It starts by calculating Expected Goals (xG) for this specific match.
    * It then corrects for the fact that 0-0 and 1-1 draws happen more often in real life than basic math predicts.
    
    **2. What is BTTS?** **Both Teams To Score.** A high percentage here (e.g., >60%) means our model expects an open game where both defenses are likely to concede.
    
    **3. What is the Slider?** The slider in the sidebar adjusts the "Context". 
    * **Moving Right (+):** Boosts Home xG and lowers Away xG (use this if the Away team has key injuries or Home has a new manager).
    * **Moving Left (-):** The opposite.
    """)

# --- FOOTER ---
st.markdown("---")
st.caption("Model Updated Weekly | Probabilities are estimates, not guarantees.")


