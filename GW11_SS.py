import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy.stats import poisson

st.set_page_config(
    page_title="Premier League Match Predictor",
    layout="wide"
)

# ==============================
#       BASIC STYLING
# ==============================
CUSTOM_CSS = """
<style>
    .main-title {
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 2px;
    }
    .sub-title {
        font-size: 14px;
        color: #9CA3AF;
        margin-bottom: 18px;
    }
    .model-card {
        border-radius: 12px;
        padding: 16px 18px;
        background-color: #0B1120;
        color: #F9FAFB;
        border: 1px solid #111827;
        margin-bottom: 12px;
    }
    .model-card h4 {
        font-size: 16px;
        margin-bottom: 6px;
    }
    .model-metric-row {
        display: flex;
        justify-content: space-between;
        font-size: 14px;
        margin: 2px 0;
    }
    .model-metric-label {
        color: #9CA3AF;
    }
    .model-metric-value {
        font-weight: 600;
    }
    .footer-note {
        margin-top: 20px;
        font-size: 11px;
        color: #6B7280;
        text-align: left;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==============================
#       LOAD OBJECTS
# ==============================
@st.cache_resource
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

pipe_result_final = load_pickle("pipe_result_final.pkl")
poisson_model     = load_pickle("poisson_model.pkl")
long_df           = load_pickle("long_df.pkl")
stats             = load_pickle("stats.pkl")
rho_hat           = load_pickle("rho_hat.pkl")

# normalise team names and position
stats["Squad"] = stats["Squad"].astype(str).str.strip()
stats["Position"] = stats["Position"].astype(str).str.strip()

# ==============================
#     HELPER FUNCTIONS
# ==============================
def pct(x, decimals=1):
    return f"{x * 100:.{decimals}f}%"


def get_expected_goals_poisson(home_team, away_team, context_adj: float = 0.0):
    """
    Expected goals (λ values) from Poisson model, adjusted by context slider.
    """
    pred_df = pd.DataFrame({
        "team":     [home_team, away_team],
        "opponent": [away_team, home_team],
        "is_home":  [1, 0],
    })
    lam = poisson_model.predict(pred_df)
    lam_home = float(lam.iloc[0])
    lam_away = float(lam.iloc[1])

    # apply context only to Poisson/DC
    lam_home = lam_home * (1 + context_adj / 10)
    lam_away = lam_away * (1 - context_adj / 10)

    lam_home = max(lam_home, 0.01)
    lam_away = max(lam_away, 0.01)

    return lam_home, lam_away


def poisson_score_matrix(home_team, away_team, max_goals=6, context_adj=0.0):
    lam_home, lam_away = get_expected_goals_poisson(home_team, away_team, context_adj)

    home_goals = np.arange(0, max_goals + 1)
    away_goals = np.arange(0, max_goals + 1)
    P = np.outer(poisson.pmf(home_goals, lam_home),
                 poisson.pmf(away_goals, lam_away))

    return home_goals, away_goals, P, lam_home, lam_away


def poisson_match_markets(home_team, away_team, max_goals=6, context_adj=0.0):
    hg, ag, P, lam_h, lam_a = poisson_score_matrix(home_team, away_team, max_goals, context_adj)
    total = hg[:, None] + ag[None, :]

    return {
        "lambda_home": lam_h,
        "lambda_away": lam_a,
        "P_home": float(np.tril(P, -1).sum()),
        "P_draw": float(np.trace(P)),
        "P_away": float(np.triu(P, 1).sum()),
        "P_over_1_5": float(P[total >= 2].sum()),
        "P_over_2_5": float(P[total >= 3].sum()),
        "P_over_3_5": float(P[total >= 4].sum()),
        "P_BTTS": float(P[(hg[:, None] > 0) & (ag[None, :] > 0)].sum()),
    }


def dixon_coles_tau(hg, ag, lam_h, lam_a, rho):
    if hg == 0 and ag == 0:
        return 1 - (lam_h * lam_a * rho)
    if hg == 0 and ag == 1:
        return 1 + lam_a * rho
    if hg == 1 and ag == 0:
        return 1 + lam_h * rho
    if hg == 1 and ag == 1:
        return 1 - rho
    return 1.0


def dixon_coles_match_markets(home_team, away_team, rho, max_goals=6, context_adj=0.0):
    hg, ag, P, lam_h, lam_a = poisson_score_matrix(home_team, away_team, max_goals, context_adj)

    for i, h in enumerate(hg):
        for j, a in enumerate(ag):
            P[i, j] *= dixon_coles_tau(h, a, lam_h, lam_a, rho)

    P /= P.sum()
    total = hg[:, None] + ag[None, :]

    return {
        "lambda_home": lam_h,
        "lambda_away": lam_a,
        "P_home": float(np.tril(P, -1).sum()),
        "P_draw": float(np.trace(P)),
        "P_away": float(np.triu(P, 1).sum()),
        "P_over_1_5": float(P[total >= 2].sum()),
        "P_over_2_5": float(P[total >= 3].sum()),
        "P_over_3_5": float(P[total >= 4].sum()),
        "P_BTTS": float(P[(hg[:, None] > 0) & (ag[None, :] > 0)].sum()),
    }


def build_feature_row_for_match(home_team, away_team):
    """
    Build the logistic model feature row.
    Logistic model remains UNADJUSTED by slider.
    """
    home_row = stats[stats["Squad"] == home_team].iloc[0]
    away_row = stats[stats["Squad"] == away_team].iloc[0]

    home_Att = home_row["Home_xG"] / home_row["Home_MP"]
    home_Def = home_row["Home_xGA"] / home_row["Home_MP"]
    away_Att = away_row["Away_xG"] / away_row["Away_MP"]
    away_Def = away_row["Away_xGA"] / away_row["Away_MP"]

    home_lf = long_df[long_df["team"] == home_team].sort_values("Date").tail(1)
    away_lf = long_df[long_df["team"] == away_team].sort_values("Date").tail(1)

    feats = {
        "home_Att": home_Att,
        "home_Def": home_Def,
        "away_Att": away_Att,
        "away_Def": away_Def,
        "strength_diff": home_Att - away_Att,
        "defense_diff": away_Def - home_Def,
        "balance_index": (home_Att - away_Att) - (away_Def - home_Def),
        "attack_defense_ratio": home_Att / (away_Def + 1e-6),
        "home_rolling_points": float(home_lf["rolling_points"]),
        "home_rolling_xG_for": float(home_lf["rolling_xG_for"]),
        "home_rolling_xG_against": float(home_lf["rolling_xG_against"]),
        "home_rolling_goals_for": float(home_lf["rolling_goals_for"]),
        "home_rolling_goals_against": float(home_lf["rolling_goals_against"]),
        "home_rolling_GD": float(home_lf["rolling_GD"]),
        "away_rolling_points": float(away_lf["rolling_points"]),
        "away_rolling_xG_for": float(away_lf["rolling_xG_for"]),
        "away_rolling_xG_against": float(away_lf["rolling_xG_against"]),
        "away_rolling_goals_for": float(away_lf["rolling_goals_for"]),
        "away_rolling_goals_against": float(away_lf["rolling_goals_against"]),
        "away_rolling_GD": float(away_lf["rolling_GD"]),
    }

    # ----------------------
# Finishing Overperformance Metrics
# ----------------------

# Attack finishing
feats["home_finishing_overperf"] = (
    feats["home_rolling_goals_for"] - feats["home_rolling_xG_for"]
)
feats["away_finishing_overperf"] = (
    feats["away_rolling_goals_for"] - feats["away_rolling_xG_for"]
)

# Defensive overperformance
feats["home_def_overperf"] = (
    feats["home_rolling_xG_against"] - feats["home_rolling_goals_against"]
)
feats["away_def_overperf"] = (
    feats["away_rolling_xG_against"] - feats["away_rolling_goals_against"]
)

# Net finishing overperformance difference
feats["finishing_overperf_diff"] = (
    feats["home_finishing_overperf"] - feats["away_finishing_overperf"]
)

feats["rolling_points_diff"] = feats["home_rolling_points"] - feats["away_rolling_points"]
feats["rolling_xG_diff"]     = feats["home_rolling_xG_for"] - feats["away_rolling_xG_for"]
feats["rolling_xGA_diff"]    = feats["home_rolling_xG_against"] - feats["away_rolling_xG_against"]
feats["rolling_GD_diff"]     = feats["home_rolling_GD"] - feats["away_rolling_GD"]

return pd.DataFrame([feats])


def get_team_position(team):
    row = stats[stats["Squad"] == team]
    return row["Position"].iloc[0] if not row.empty else "–"


def last_5_results(team):
    """Return form oldest → latest."""
    df_team = long_df[long_df["team"] == team].sort_values("Date").tail(5)
    letters, icons = [], []
    for _, r in df_team.iterrows():
        gf, ga = r["goals_for"], r["goals_against"]
        if gf > ga:
            letters.append("W"); icons.append("🟩")
        elif gf == ga:
            letters.append("D"); icons.append("🟨")
        else:
            letters.append("L"); icons.append("🟥")
    return "".join(letters), "".join(icons)

# ==============================
#           SIDEBAR
# ==============================
with st.sidebar:
    st.header("Match Setup")
    teams = sorted(stats["Squad"].unique())

    home_team = st.selectbox("Home team", teams)
    away_team = st.selectbox("Away team", teams)

    st.markdown("---")

    advanced_mode = st.checkbox(
        "Advanced adjustment mode",
        value=False,
        help="Enable more granular match context controls (injuries, morale, tactics)."
    )

    if not advanced_mode:
        context_adj = st.slider(
            "Context Adjustment (−3 = strong AWAY boost, +3 = strong HOME boost)",
            -3.0, 3.0, 0.0, 0.1,
            help="Adjusts expected goals and Dixon–Coles probabilities only."
        )
    else:
        st.subheader("Advanced Context Controls")

        att_adj = st.slider(
            "Attack Impact (Home attacking boost / Away penalty)",
            -3.0, 3.0, 0.0, 0.1
        )
        def_adj = st.slider(
            "Defensive Impact (Home defensive boost / Away vulnerability)",
            -3.0, 3.0, 0.0, 0.1
        )
        morale_adj = st.slider(
            "Morale / Momentum Shift (Psychological advantage)",
            -3.0, 3.0, 0.0, 0.1
        )

        context_adj = (att_adj + def_adj + 0.5 * morale_adj) / 2.0

    st.markdown(f"**Current adjustment:** `{context_adj:+.2f}`")
    st.caption("Note: adjustments affect Poisson & Dixon–Coles models only.")

# ==============================
#       MAIN AREA
# ==============================
if home_team == away_team:
    st.warning("Home and Away teams must be different.")
    st.stop()

st.markdown('<div class="main-title">Premier League Match Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Poisson expected goals · Dixon–Coles correction · Logistic baseline</div>',
    unsafe_allow_html=True
)
st.markdown(f"### {home_team} vs {away_team}")
st.markdown("---")

# -----------------------
# Logistic Model (baseline only)
# -----------------------
X_row = build_feature_row_for_match(home_team, away_team)
log_probs = pipe_result_final.predict_proba(X_row)[0]
classes = pipe_result_final.named_steps["clf"].classes_
p_log = dict(zip(classes, log_probs))

# -----------------------
# Poisson & Dixon Coles (adjusted)
# -----------------------
pm     = poisson_match_markets(home_team, away_team, context_adj=context_adj)
pm_base = poisson_match_markets(home_team, away_team)

dc     = dixon_coles_match_markets(home_team, away_team, rho_hat, context_adj=context_adj)
dc_base = dixon_coles_match_markets(home_team, away_team, rho_hat)

home_adjustment_delta = (dc["P_home"] - dc_base["P_home"]) * 100

# ==============================
#  HEADLINE (DIXON–COLES)
# ==============================
st.subheader("Headline (Dixon–Coles adjusted)")

colA, colB, colC, colD = st.columns(4)
colA.metric("Home Win", pct(dc["P_home"]))
colB.metric("Draw", pct(dc["P_draw"]))
colC.metric("Away Win", pct(dc["P_away"]))
colD.metric("BTTS", pct(dc["P_BTTS"]))

st.caption("Dixon–Coles adjusts Poisson scorelines for realistic low-scoring match dependencies.")

# ==============================
#     MODEL CARDS
# ==============================
c1, c2, c3 = st.columns(3)

# Logistic model card (baseline)
with c1:
    st.markdown('<div class="model-card">', unsafe_allow_html=True)
    st.markdown("<h4>Logistic Model (Baseline)</h4>", unsafe_allow_html=True)
    st.markdown(f'<div class="model-metric-row"><span class="model-metric-label">Home Win</span><span class="model-metric-value">{pct(p_log["H"])}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="model-metric-row"><span class="model-metric-label">Draw</span><span class="model-metric-value">{pct(p_log["D"])}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="model-metric-row"><span class="model-metric-label">Away Win</span><span class="model-metric-value">{pct(p_log["A"])}</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title" style="margin-top:8px;">This model is unaffected by context sliders.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Poisson model card
with c2:
    st.markdown('<div class="model-card">', unsafe_allow_html=True)
    st.markdown("<h4>Poisson Model (Adjusted)</h4>", unsafe_allow_html=True)

    st.markdown(f'<div class="model-metric-row"><span class="model-metric-label">λ Home</span><span class="model-metric-value">{pm["lambda_home"]:.2f}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="model-metric-row"><span class="model-metric-label">λ Away</span><span class="model-metric-value">{pm["lambda_away"]:.2f}</span></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="model-metric-row"><span class="model-metric-label">Over 1.5</span><span class="model-metric-value">{pct(pm["P_over_1_5"])}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="model-metric-row"><span class="model-metric-label">Over 2.5</span><span class="model-metric-value">{pct(pm["P_over_2_5"])}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="model-metric-row"><span class="model-metric-label">Over 3.5</span><span class="model-metric-value">{pct(pm["P_over_3_5"])}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="model-metric-row"><span class="model-metric-label">BTTS</span><span class="model-metric-value">{pct(pm["P_BTTS"])}</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="sub-title" style="margin-top:8px;">Sliders adjust expected goals, influencing scoreline probabilities.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# DC model card
with c3:
    st.markdown('<div class="model-card">', unsafe_allow_html=True)
    st.markdown("<h4>Dixon–Coles Model (Adjusted)</h4>", unsafe_allow_html=True)

    st.markdown(f'<div class="model-metric-row"><span class="model-metric-label">Home Win</span><span class="model-metric-value">{pct(dc["P_home"])}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="model-metric-row"><span class="model-metric-label">Draw</span><span class="model-metric-value">{pct(dc["P_draw"])}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="model-metric-row"><span class="model-metric-label">Away Win</span><span class="model-metric-value">{pct(dc["P_away"])}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="model-metric-row"><span class="model-metric-label">BTTS</span><span class="model-metric-value">{pct(dc["P_BTTS"])}</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="sub-title" style="margin-top:8px;">Combines adjusted Poisson λs with dependency correction.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================
#  ADJUSTMENT IMPACT
# ==============================
impact_color = "green" if home_adjustment_delta > 0 else ("red" if home_adjustment_delta < 0 else "#9CA3AF")

st.markdown(
    f"""
    <div class="model-card" style="margin-top:4px;">
        <strong>Adjustment Impact</strong><br/>
        Your context adjustment changed the 
        <strong>Home Win probability (Dixon–Coles)</strong> by 
        <span style="color:{impact_color};">{home_adjustment_delta:+.2f}%</span>.<br>
        <span style="font-size:12px; color:#9CA3AF;">
            Note: Logistic model is unchanged. Adjustments modify Poisson λ-values and DC probabilities only.
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================
#  TEAM FORM & LEAGUE POSITION
# ==============================
with st.expander("Team Form & League Position (last 5 matches)"):
    col_h, col_a = st.columns(2)

    h_letters, h_icons = last_5_results(home_team)
    a_letters, a_icons = last_5_results(away_team)

    with col_h:
        st.markdown(f"**{home_team}**")
        st.markdown(f"Position: **{get_team_position(home_team)}**")
        st.markdown(f"Form (old → new): {h_icons}  ({h_letters})")

    with col_a:
        st.markdown(f"**{away_team}**")
        st.markdown(f"Position: **{get_team_position(away_team)}**")
        st.markdown(f"Form (old → new): {a_icons}  ({a_letters})")

st.markdown(
    '<div class="footer-note">'
    'Probabilities are model-based estimates only. Models update weekly as new matches are added.</div>',
    unsafe_allow_html=True
)

