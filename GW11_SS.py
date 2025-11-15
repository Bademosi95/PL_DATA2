import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy.stats import poisson

st.set_page_config(page_title="Premier League Predictor", layout="wide")


# =========================================================
#             LOAD MODELS & DATA (WITH CACHING)
# =========================================================
@st.cache_resource
def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


pipe_result_final = load_pickle("pipe_result_final.pkl")
poisson_model = load_pickle("poisson_model.pkl")
long_df = load_pickle("long_df.pkl")
stats = load_pickle("stats.pkl")
rho_hat = load_pickle("rho_hat.pkl")


# =========================================================
#                  HELPER FUNCTIONS
# =========================================================

def get_expected_goals_poisson(home_team, away_team):
    pred_df = pd.DataFrame({
        "team": [home_team, away_team],
        "opponent": [away_team, home_team],
        "is_home": [1, 0]
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
    hg, ag, P, lam_h, lam_a = poisson_score_matrix(home_team, away_team)
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
    elif hg == 0 and ag == 1:
        return 1 + lam_a * rho
    elif hg == 1 and ag == 0:
        return 1 + lam_h * rho
    elif hg == 1 and ag == 1:
        return 1 - rho
    else:
        return 1.0


def dixon_coles_match_markets(home_team, away_team, rho, max_goals=6):
    hg, ag, P, lam_h, lam_a = poisson_score_matrix(home_team, away_team)

    # apply tau adjustments
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

    # season xG attack/defence
    home_Att = home_row["Home_xG"] / home_row["Home_MP"]
    home_Def = home_row["Home_xGA"] / home_row["Home_MP"]
    away_Att = away_row["Away_xG"] / away_row["Away_MP"]
    away_Def = away_row["Away_xGA"] / away_row["Away_MP"]

    home_lf = long_df[long_df["team"] == home_team].tail(1)
    away_lf = long_df[long_df["team"] == away_team].tail(1)

    features = {
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

    features["rolling_points_diff"] = (
        features["home_rolling_points"] - features["away_rolling_points"]
    )
    features["rolling_xG_diff"] = (
        features["home_rolling_xG_for"] - features["away_rolling_xG_for"]
    )
    features["rolling_xGA_diff"] = (
        features["home_rolling_xG_against"] - features["away_rolling_xG_against"]
    )
    features["rolling_GD_diff"] = (
        features["home_rolling_GD"] - features["away_rolling_GD"]
    )

    return pd.DataFrame([features])


# =========================================================
#                    STREAMLIT UI
# =========================================================

st.title("⚽ Premier League Match Predictor")
st.write("Three-model prediction engine: Logistic • Poisson • Dixon–Coles")
st.markdown("---")

teams = sorted(stats["Squad"].unique())
col1, col2 = st.columns(2)

home_team = col1.selectbox("Home Team", teams)
away_team = col2.selectbox("Away Team", teams)

if home_team == away_team:
    st.warning("Home and away teams must be different.")
    st.stop()

st.header(f"### {home_team} vs {away_team}")
st.markdown("---")

# =========================================================
# LOGISTIC MODEL
# =========================================================
st.subheader("📊 Logistic Model Prediction")
X_row = build_feature_row_for_match(home_team, away_team)
log_probs = pipe_result_final.predict_proba(X_row)[0]

st.write({
    "Home Win": float(log_probs[2]),
    "Draw": float(log_probs[1]),
    "Away Win": float(log_probs[0])
})

# =========================================================
# POISSON MODEL
# =========================================================
st.subheader("📈 Poisson Model (Expected Goals)")

pm = poisson_match_markets(home_team, away_team)

st.write({
    "Home Win": pm["P_home"],
    "Draw": pm["P_draw"],
    "Away Win": pm["P_away"],
    "xG Home": pm["lambda_home"],
    "xG Away": pm["lambda_away"],
    "Over 1.5": pm["P_over_1_5"],
    "Over 2.5": pm["P_over_2_5"],
    "Over 3.5": pm["P_over_3_5"],
    "BTTS": pm["P_BTTS"],
})

# =========================================================
# DIXON–COLES
# =========================================================
st.subheader("⚡ Dixon–Coles Adjusted Poisson")

dc = dixon_coles_match_markets(home_team, away_team, rho_hat)

st.write({
    "Home Win": dc["P_home"],
    "Draw": dc["P_draw"],
    "Away Win": dc["P_away"],
    "Over 1.5": dc["P_over_1_5"],
    "Over 2.5": dc["P_over_2_5"],
    "Over 3.5": dc["P_over_3_5"],
    "BTTS": dc["P_BTTS"],
})
