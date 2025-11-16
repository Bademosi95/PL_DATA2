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
        margin-bottom: 10px;
    }
    .model-card h3 {
        margin-top: 0;
        margin-bottom: 4px;
        font-size: 18px;
    }
    .model-meta {
        font-size: 12px;
        color: #9CA3AF;
        margin-bottom: 12px;
    }
    .footer-note {
        font-size: 11px;
        color: #6B7280;
        margin-top: 12px;
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
if "Squad" in stats.columns:
    stats["Squad"] = stats["Squad"].astype(str).str.strip()
if "Position" in stats.columns:
    stats["Position"] = stats["Position"].astype(str).str.strip()


# ==============================
#     HELPER FUNCTIONS
# ==============================
def pct(x, decimals=1):
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

    feats["rolling_points_diff"] = feats["home_rolling_points"] - feats["away_rolling_points"]
    feats["rolling_xG_diff"] = feats["home_rolling_xG_for"] - feats["away_rolling_xG_for"]
    feats["rolling_xGA_diff"] = feats["home_rolling_xG_against"] - feats["away_rolling_xG_against"]
    feats["rolling_GD_diff"] = feats["home_rolling_GD"] - feats["away_rolling_GD"]

    return pd.DataFrame([feats])


def get_team_position(team_name: str) -> str:
    row = stats[stats["Squad"] == team_name]
    if "Position" in stats.columns and not row.empty:
        return str(row["Position"].iloc[0])
    return "–"


def last_5_results(team_name: str):
    df_team = long_df[long_df["team"] == team_name].sort_values("Date", ascending=False).head(5)
    res_letters = []
    res_icons = []
    for _, r in df_team.iterrows():
        gf = r["goals_for"]
        ga = r["goals_against"]
        if gf > ga:
            res_letters.append("W")
            res_icons.append("🟩")
        elif gf == ga:
            res_letters.append("D")
            res_icons.append("🟨")
        else:
            res_letters.append("L")
            res_icons.append("🟥")
    return "".join(res_letters[::-1]), "".join(res_icons[::-1])  # oldest→latest left→right


# ==============================
#           SIDEBAR
# ==============================
with st.sidebar:
    st.header("Match Setup")
    teams = sorted(stats["Squad"].unique())
    home_team = st.selectbox("Home team", teams, key="home_team")
    away_team = st.selectbox("Away team", teams, key="away_team")

    st.markdown("---")
    st.caption("Tip: update the underlying CSVs weekly with new GW data, then retrain & re-export the models/pickles.")


if home_team == away_team:
    st.warning("Home and away teams must be different.")
    st.stop()

# ==============================
#       MAIN AREA
# ==============================
st.markdown('<div class="main-title">Premier League Match Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Logistic model · Poisson expected goals · Dixon–Coles adjustment</div>',
    unsafe_allow_html=True
)

st.markdown(f"### {home_team} vs {away_team}")
st.markdown("---")

# ---- core calculations
X_row = build_feature_row_for_match(home_team, away_team)
log_probs = pipe_result_final.predict_proba(X_row)[0]
classes = pipe_result_final.named_steps["clf"].classes_
p_log = {cls: float(prob) for cls, prob in zip(classes, log_probs)}

pm = poisson_match_markets(home_team, away_team)
dc = dixon_coles_match_markets(home_team, away_team, rho_hat)

# ==============================
#  HEADLINE (DIXON–COLES)
# ==============================
st.subheader("Headline view (Dixon–Coles model)")

hc1, hc2, hc3, hc4 = st.columns(4)
hc1.metric("Home Win", pct(dc["P_home"]))
hc2.metric("Draw", pct(dc["P_draw"]))
hc3.metric("Away Win", pct(dc["P_away"]))
hc4.metric("BTTS", pct(dc["P_BTTS"]))

st.caption("Dixon–Coles: Poisson-based, calibrated for low-scoring match dependence (0–0, 1–0, 1–1 etc.).")
st.markdown("---")

# ==============================
#  MODEL CARDS
# ==============================

# Logistic card
st.markdown('<div class="model-card">', unsafe_allow_html=True)
st.markdown("<h3>📊 Logistic Model</h3>", unsafe_allow_html=True)
st.markdown(
    '<div class="model-meta">'
    'Directly predicts Home / Draw / Away using team strength, home/away splits, and 5-game rolling form.'
    '</div>',
    unsafe_allow_html=True
)

lc1, lc2, lc3 = st.columns(3)
lc1.metric("Home Win", pct(p_log.get("H", np.nan)))
lc2.metric("Draw", pct(p_log.get("D", np.nan)))
lc3.metric("Away Win", pct(p_log.get("A", np.nan)))
st.markdown('</div>', unsafe_allow_html=True)

# Poisson card
st.markdown('<div class="model-card">', unsafe_allow_html=True)
st.markdown("<h3>📈 Poisson Expected Goals Model</h3>", unsafe_allow_html=True)
st.markdown(
    '<div class="model-meta">'
    'Estimates expected goals for each team, then derives result and goal-market probabilities assuming independent scoring.'
    '</div>',
    unsafe_allow_html=True
)

pc1, pc2, pc3, pc4 = st.columns(4)
pc1.metric("xG Home", f"{pm['lambda_home']:.2f}")
pc2.metric("xG Away", f"{pm['lambda_away']:.2f}")
pc3.metric("Over 2.5", pct(pm["P_over_2_5"]))
pc4.metric("BTTS", pct(pm["P_BTTS"]))

poisson_df = pd.DataFrame({
    "Outcome": ["Home win", "Draw", "Away win"],
    "Poisson (Prob)": [pct(pm["P_home"]), pct(pm["P_draw"]), pct(pm["P_away"])]
})
st.table(poisson_df.style.hide(axis="index"))
st.markdown('</div>', unsafe_allow_html=True)

# Dixon–Coles card
st.markdown('<div class="model-card">', unsafe_allow_html=True)
st.markdown("<h3>⚡ Dixon–Coles Adjusted Poisson</h3>", unsafe_allow_html=True)
st.markdown(
    '<div class="model-meta">'
    'Refines the Poisson model by adjusting for correlation in low-scoring matches, improving calibration for tight games and draws.'
    '</div>',
    unsafe_allow_html=True
)

dc1, dc2, dc3, dc4 = st.columns(4)
dc1.metric("Home Win", pct(dc["P_home"]))
dc2.metric("Draw", pct(dc["P_draw"]))
dc3.metric("Away Win", pct(dc["P_away"]))
dc4.metric("Over 2.5", pct(dc["P_over_2_5"]))

dc5, dc6, dc7, dc8 = st.columns(4)
dc5.metric("Over 1.5", pct(dc["P_over_1_5"]))
dc6.metric("Over 3.5", pct(dc["P_over_3_5"]))
dc7.metric("BTTS", pct(dc["P_BTTS"]))
dc8.metric("xG H / A", f"{dc['lambda_home']:.2f} / {dc['lambda_away']:.2f}")
st.markdown('</div>', unsafe_allow_html=True)

# ==============================
#  TEAM FORM & LEAGUE POSITION
# ==============================
with st.expander("Team form & league position (last 5 matches)"):
    home_form_letters, home_form_icons = last_5_results(home_team)
    away_form_letters, away_form_icons = last_5_results(away_team)

    table_df = pd.DataFrame([
        {
            "Team": home_team,
            "Position": get_team_position(home_team),
            "Last 5 (W/D/L)": " ".join(list(home_form_letters)),
            "Form (old → recent)": " ".join(list(home_form_icons)),
        },
        {
            "Team": away_team,
            "Position": get_team_position(away_team),
            "Last 5 (W/D/L)": " ".join(list(away_form_letters)),
            "Form (old → recent)": " ".join(list(away_form_icons)),
        }
    ])

    st.table(table_df)

st.markdown(
    '<div class="footer-note">'
    'Probabilities are model-based estimates only. Underlying data and models update as new GWs are added.'
    '</div>',
    unsafe_allow_html=True
)


