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
#       STYLING / THEME
# ==============================
CUSTOM_CSS = """
<style>
    .main-title {
        font-size: 32px;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 16px;
        color: #888888;
        margin-bottom: 20px;
    }
    .model-card {
        border-radius: 12px;
        padding: 16px 18px;
        background-color: #111827;
        color: #F9FAFB;
        border: 1px solid #1F2937;
    }
    .model-card h3 {
        margin-top: 0;
        margin-bottom: 6px;
    }
    .model-meta {
        font-size: 13px;
        color: #9CA3AF;
        margin-bottom: 10px;
    }
    .percent-table th, .percent-table td {
        padding: 4px 8px;
    }
    .footer-note {
        font-size: 11px;
        color: #6B7280;
        margin-top: 20px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================
#        LOAD OBJECTS
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


# ==============================
#     HELPER FUNCTIONS
# ==============================
def pct(x, decimals=1):
    """Convert probability 0–1 to percentage string."""
    return f"{x * 100:.{decimals}f}%"

def get_expected_goals_poisson(home_team, away_team):
    pred_df = pd.DataFrame({
        "team":     [home_team, away_team],
        "opponent": [away_team, home_team],
        "is_home":  [1, 0],
    })
    lam = poisson_model.predict(pred_df)
    return float(lam.iloc[0]), float(lam.iloc[1])


def poisson_score_matrix(home_team, away_team, max_goals=6):
    lam_home, lam_away = get_expected_goals_poisson(home_team, away_team)
    home_goals = np.arange(0, max_goals + 1)
    away_goals = np.arange(0, max_goals + 1)
    P = np.outer(poisson.pmf(home_goals, lam_home),
                 poisson.pmf(away_goals, lam_away))
    return home_goals, away_goals, P, lam_home, lam_away


def poisson_match_markets(home_team, away_team, max_goals=6):
    hg, ag, P, lam_h, lam_a = poisson_score_matrix(home_team, away_team, max_goals=max_goals)
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


def dixon_coles_match_markets(home_team, away_team, rho, max_goals=6):
    hg, ag, P, lam_h, lam_a = poisson_score_matrix(home_team, away_team, max_goals=max_goals)

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
    home_row = stats[stats["Squad"] == home_team].iloc[0]
    away_row = stats[stats["Squad"] == away_team].iloc[0]

    home_Att = home_row["Home_xG"] / home_row["Home_MP"]
    home_Def = home_row["Home_xGA"] / home_row["Home_MP"]
    away_Att = away_row["Away_xG"] / away_row["Away_MP"]
    away_Def = away_row["Away_xGA"] / away_row["Away_MP"]

    home_lf = long_df[long_df["team"] == home_team].tail(1)
    away_lf = long_df[long_df["team"] == away_team].tail(1)

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

    feats["rolling_points_diff"] = feats["home_rolling_points"] - feats["away_rolling_points"]
    feats["rolling_xG_diff"] = feats["home_rolling_xG_for"] - feats["away_rolling_xG_for"]
    feats["rolling_xGA_diff"] = feats["home_rolling_xG_against"] - feats["away_rolling_xG_against"]
    feats["rolling_GD_diff"] = feats["home_rolling_GD"] - feats["away_rolling_GD"]

    return pd.DataFrame([feats])


# ==============================
#             UI
# ==============================

st.markdown('<div class="main-title">⚽ Premier League Match Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Three-model engine combining form, xG, and scoreline modelling.</div>',
    unsafe_allow_html=True
)

teams = sorted(stats["Squad"].unique())
col_home, col_away = st.columns(2)
home_team = col_home.selectbox("Home team", teams, index=teams.index(teams[0]))
away_team = col_away.selectbox("Away team", teams, index=min(1, len(teams)-1))

if home_team == away_team:
    st.warning("Home and away teams must be different.")
    st.stop()

st.markdown(f"### {home_team} vs {away_team}")
st.markdown("---")

# ----------------- Core calculations -----------------
# Logistic
X_row = build_feature_row_for_match(home_team, away_team)
log_probs = pipe_result_final.predict_proba(X_row)[0]
# classes_: should be ['A','D','H']
classes = pipe_result_final.named_steps["clf"].classes_
p_log = {
    cls: float(prob) for cls, prob in zip(classes, log_probs)
}

# Poisson
pm = poisson_match_markets(home_team, away_team)
# DC
dc = dixon_coles_match_markets(home_team, away_team, rho_hat)

# ==============================
#      TOPLINE SUMMARY ROW
# ==============================
st.subheader("Headline View (Dixon–Coles model)")

top_col1, top_col2, top_col3, top_col4 = st.columns(4)
top_col1.metric("Home Win", pct(dc["P_home"]))
top_col2.metric("Draw", pct(dc["P_draw"]))
top_col3.metric("Away Win", pct(dc["P_away"]))
top_col4.metric("BTTS", pct(dc["P_BTTS"]))

st.caption("Headline probabilities based on the Dixon–Coles adjusted Poisson model (better calibrated for low-scoring matches).")

st.markdown("---")

# ==============================
#      LOGISTIC MODEL CARD
# ==============================
st.markdown('<div class="model-card">', unsafe_allow_html=True)
st.markdown("<h3>📊 Logistic Model</h3>", unsafe_allow_html=True)
st.markdown(
    '<div class="model-meta">'
    'Uses team strength, home/away splits and 5-game rolling form features to directly predict the match result (Home / Draw / Away).'
    '</div>',
    unsafe_allow_html=True
)

log_col1, log_col2, log_col3 = st.columns(3)
log_col1.metric("Home Win", pct(p_log.get("H", np.nan)))
log_col2.metric("Draw", pct(p_log.get("D", np.nan)))
log_col3.metric("Away Win", pct(p_log.get("A", np.nan)))

st.markdown("</div>", unsafe_allow_html=True)  # end card
st.markdown("")

# ==============================
#       POISSON MODEL CARD
# ==============================
st.markdown('<div class="model-card">', unsafe_allow_html=True)
st.markdown("<h3>📈 Poisson Expected Goals Model</h3>", unsafe_allow_html=True)
st.markdown(
    '<div class="model-meta">'
    'Predicts expected goals for each team, then derives probabilities for results and goal-based markets assuming independent Poisson scoring.'
    '</div>',
    unsafe_allow_html=True
)

p_col1, p_col2, p_col3, p_col4 = st.columns(4)
p_col1.metric("xG Home", f"{pm['lambda_home']:.2f}")
p_col2.metric("xG Away", f"{pm['lambda_away']:.2f}")
p_col3.metric("Over 2.5", pct(pm["P_over_2_5"]))
p_col4.metric("BTTS", pct(pm["P_BTTS"]))

# Result probs table (Poisson)
poisson_df = pd.DataFrame({
    "Outcome": ["Home win", "Draw", "Away win"],
    "Poisson Probability": [
        pct(pm["P_home"]),
        pct(pm["P_draw"]),
        pct(pm["P_away"])
    ]
})
st.table(poisson_df.style.hide(axis="index"))

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("")

# ==============================
#     DIXON–COLES MODEL CARD
# ==============================
st.markdown('<div class="model-card">', unsafe_allow_html=True)
st.markdown("<h3>⚡ Dixon–Coles Adjusted Poisson</h3>", unsafe_allow_html=True)
st.markdown(
    '<div class="model-meta">'
    "Builds on the Poisson model, but corrects for dependence in low-scoring results (0–0, 1–0, 1–1 etc.), "
    "which improves calibration for tight games and draw probabilities."
    '</div>',
    unsafe_allow_html=True
)

dc_c1, dc_c2, dc_c3, dc_c4 = st.columns(4)
dc_c1.metric("Home Win", pct(dc["P_home"]))
dc_c2.metric("Draw", pct(dc["P_draw"]))
dc_c3.metric("Away Win", pct(dc["P_away"]))
dc_c4.metric("Over 2.5", pct(dc["P_over_2_5"]))

dc_c5, dc_c6, dc_c7, dc_c8 = st.columns(4)
dc_c5.metric("Over 1.5", pct(dc["P_over_1_5"]))
dc_c6.metric("Over 3.5", pct(dc["P_over_3_5"]))
dc_c7.metric("BTTS", pct(dc["P_BTTS"]))
dc_c8.metric("xG Home / Away", f"{dc['lambda_home']:.2f} / {dc['lambda_away']:.2f}")

st.markdown("</div>", unsafe_allow_html=True)

# ==============================
#      FOOTER / CONTEXT
# ==============================
st.markdown(
    '<div class="footer-note">'
    'All probabilities are model-based estimates and not guarantees. '
    'Models are built on current-season Premier League data and update as new gameweeks are added.'
    '</div>',
    unsafe_allow_html=True
)

