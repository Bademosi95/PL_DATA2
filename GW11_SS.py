import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy.stats import poisson

st.set_page_config(
    page_title="KB Premier League Match Predictor",
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
if "Squad" in stats.columns:
    stats["Squad"] = stats["Squad"].astype(str).str.strip()
if "Position" in stats.columns:
    stats["Position"] = stats["Position"].astype(str).str.strip()

# ==============================
#     HELPER FUNCTIONS
# ==============================
def pct(x, decimals=1):
    return f"{x * 100:.{decimals}f}%"


def get_expected_goals_poisson(home_team, away_team, context_adj: float = 0.0):
    """
    Return expected goals (lambda_home, lambda_away) from the Poisson model,
    optionally adjusted by a user-defined context factor.
    Positive context_adj favours the home side; negative favours the away side.
    """
    pred_df = pd.DataFrame({
        "team":     [home_team, away_team],
        "opponent": [away_team, home_team],
        "is_home":  [1, 0],
    })
    lam = poisson_model.predict(pred_df)
    lam_home = float(lam.iloc[0])
    lam_away = float(lam.iloc[1])

    if context_adj != 0.0:
        lam_home = lam_home * (1.0 + context_adj / 10.0)
        lam_away = lam_away * (1.0 - context_adj / 10.0)
        # keep lambdas positive
        lam_home = max(lam_home, 0.01)
        lam_away = max(lam_away, 0.01)

    return lam_home, lam_away


def poisson_score_matrix(home_team, away_team, max_goals: int = 6, context_adj: float = 0.0):
    lam_home, lam_away = get_expected_goals_poisson(home_team, away_team, context_adj=context_adj)
    home_goals = np.arange(0, max_goals + 1)
    away_goals = np.arange(0, max_goals + 1)
    P = np.outer(poisson.pmf(home_goals, lam_home),
                 poisson.pmf(away_goals, lam_away))
    return home_goals, away_goals, P, lam_home, lam_away


def poisson_match_markets(home_team, away_team, max_goals: int = 6, context_adj: float = 0.0):
    hg, ag, P, lam_h, lam_a = poisson_score_matrix(
        home_team, away_team,
        max_goals=max_goals,
        context_adj=context_adj
    )
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


def dixon_coles_match_markets(home_team, away_team, rho, max_goals: int = 6, context_adj: float = 0.0):
    hg, ag, P, lam_h, lam_a = poisson_score_matrix(
        home_team, away_team,
        max_goals=max_goals,
        context_adj=context_adj
    )

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
    Rebuilds the logistic feature row for a single fixture using stats + long_df.
    This mirrors the feature engineering used in training.
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

    feats["rolling_points_diff"] = feats["home_rolling_points"] - feats["away_rolling_points"]
    feats["rolling_xG_diff"]     = feats["home_rolling_xG_for"] - feats["away_rolling_xG_for"]
    feats["rolling_xGA_diff"]    = feats["home_rolling_xG_against"] - feats["away_rolling_xG_against"]
    feats["rolling_GD_diff"]     = feats["home_rolling_GD"] - feats["away_rolling_GD"]

    return pd.DataFrame([feats])


def get_team_position(team):
    row = stats[stats["Squad"] == team]
    if row.empty or "Position" not in row.columns:
        return "–"
    return str(row["Position"].iloc[0])


def last_5_results(team):
    df_team = long_df[long_df["team"] == team].sort_values("Date").tail(5)
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

    # --- Match context adjustment controls ---
    advanced_mode = st.checkbox(
        "Advanced adjustment mode",
        value=False,
        help="Enable more granular match context controls (injuries, suspensions, European mid week game, morale, etc.)."
    )

    if not advanced_mode:
        context_adj = st.slider(
            "Match context adjustment",
            min_value=-3.0,
            max_value=3.0,
            value=0.0,
            step=0.1,
            help="Negative values favour the away side; positive values favour the home side."
        )
    else:
        st.subheader("Context controls")
        att_adj = st.slider(
            "Attack impact (home boost / away penalty)",
            min_value=-3.0,
            max_value=3.0,
            value=0.0,
            step=0.1,
        )
        def_adj = st.slider(
            "Defensive impact (home resilience / away vulnerability)",
            min_value=-3.0,
            max_value=3.0,
            value=0.0,
            step=0.1,
        )
        morale_adj = st.slider(
            "Morale / momentum shift",
            min_value=-3.0,
            max_value=3.0,
            value=0.0,
            step=0.1,
        )
        # Combine into a single context factor
        context_adj = (att_adj + def_adj + 0.5 * morale_adj) / 2.0

    st.markdown(f"**Current context adjustment:** `{context_adj:+.2f}`")

    st.markdown("---")
    st.caption("Tip: Keep an eye on real life events that may affect outcome")

if home_team == away_team:
    st.warning("Home and away teams must be different.")
    st.stop()

# ==============================
#       MAIN AREA
# ==============================
st.markdown('<div class="main-title">Premier League Match Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Logistic result model · Poisson expected goals · Dixon–Coles correction · Context-aware slider</div>',
    unsafe_allow_html=True
)
st.markdown(f"### {home_team} vs {away_team}")
st.markdown("---")

# ---- core calculations
X_row = build_feature_row_for_match(home_team, away_team)

# Logistic model: baseline vs context-adjusted
log_probs_base = pipe_result_final.predict_proba(X_row)[0]
classes = pipe_result_final.named_steps["clf"].classes_
p_log_base = {cls: float(prob) for cls, prob in zip(classes, log_probs_base)}

X_row_adj = X_row.copy()
if "strength_diff" in X_row_adj.columns:
    X_row_adj["strength_diff"] = X_row_adj["strength_diff"] + context_adj
if "balance_index" in X_row_adj.columns:
    X_row_adj["balance_index"] = X_row_adj["balance_index"] + context_adj
if "attack_defense_ratio" in X_row_adj.columns:
    X_row_adj["attack_defense_ratio"] = X_row_adj["attack_defense_ratio"] * (1.0 + context_adj / 10.0)
if "rolling_points_diff" in X_row_adj.columns:
    X_row_adj["rolling_points_diff"] = X_row_adj["rolling_points_diff"] + 0.2 * context_adj
if "rolling_GD_diff" in X_row_adj.columns:
    X_row_adj["rolling_GD_diff"] = X_row_adj["rolling_GD_diff"] + 0.1 * context_adj

log_probs = pipe_result_final.predict_proba(X_row_adj)[0]
p_log = {cls: float(prob) for cls, prob in zip(classes, log_probs)}

# Poisson / Dixon–Coles: baseline vs context-adjusted
pm_base = poisson_match_markets(home_team, away_team)
dc_base = dixon_coles_match_markets(home_team, away_team, rho_hat)

pm = poisson_match_markets(home_team, away_team, context_adj=context_adj)
dc = dixon_coles_match_markets(home_team, away_team, rho_hat, context_adj=context_adj)

# Adjustment impact for headline metric (DC home win prob)
H_base = dc_base["P_home"]
H_adj = dc["P_home"]
home_adjustment_delta = (H_adj - H_base) * 100.0

# ==============================
#  HEADLINE (DIXON–COLES)
# ==============================
st.subheader("Headline view (Dixon–Coles model)")

hc1, hc2, hc3, hc4 = st.columns(4)
hc1.metric("Home Win", pct(dc["P_home"]))
hc2.metric("Draw", pct(dc["P_draw"]))
hc3.metric("Away Win", pct(dc["P_away"]))
hc4.metric("BTTS", pct(dc["P_BTTS"]))

st.caption("Dixon–Coles: Poisson-based probabilities with correction for low-scoring outcomes (0–0, 1–0, 1–1, etc.).")

# ==============================
#  MODEL CARDS
# ==============================
c1, c2, c3 = st.columns(3)

# ----- Logistic model card -----
with c1:
    st.markdown('<div class="model-card">', unsafe_allow_html=True)
    st.markdown("<h4>Logistic model (result)</h4>", unsafe_allow_html=True)
    st.markdown('<div class="model-metric-row"><span class="model-metric-label">Home Win</span>'
                f'<span class="model-metric-value">{pct(p_log.get("H", 0.0))}</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="model-metric-row"><span class="model-metric-label">Draw</span>'
                f'<span class="model-metric-value">{pct(p_log.get("D", 0.0))}</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="model-metric-row"><span class="model-metric-label">Away Win</span>'
                f'<span class="model-metric-value">{pct(p_log.get("A", 0.0))}</span></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title" style="margin-top:8px;">Form & strength based classification using engineered features (rolling form, xG, attack/defence balance).</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ----- Poisson model card -----
with c2:
    st.markdown('<div class="model-card">', unsafe_allow_html=True)
    st.markdown("<h4>Poisson model (goals)</h4>", unsafe_allow_html=True)
    st.markdown(
        '<div class="model-metric-row"><span class="model-metric-label">λ Home</span>'
        f'<span class="model-metric-value">{pm["lambda_home"]:.2f}</span></div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="model-metric-row"><span class="model-metric-label">λ Away</span>'
        f'<span class="model-metric-value">{pm["lambda_away"]:.2f}</span></div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="model-metric-row"><span class="model-metric-label">Over 2.5</span>'
        f'<span class="model-metric-value">{pct(pm["P_over_2_5"])}</span></div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="model-metric-row"><span class="model-metric-label">Over 3.5</span>'
        f'<span class="model-metric-value">{pct(pm["P_over_3_5"])}</span></div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="model-metric-row"><span class="model-metric-label">BTTS</span>'
        f'<span class="model-metric-value">{pct(pm["P_BTTS"])}</span></div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-title" style="margin-top:8px;">Classic expected-goals framework modelling scorelines as independent Poisson processes.</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ----- Dixon–Coles model card -----
with c3:
    st.markdown('<div class="model-card">', unsafe_allow_html=True)
    st.markdown("<h4>Dixon–Coles model</h4>", unsafe_allow_html=True)
    st.markdown(
        '<div class="model-metric-row"><span class="model-metric-label">Home Win</span>'
        f'<span class="model-metric-value">{pct(dc["P_home"])}</span></div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="model-metric-row"><span class="model-metric-label">Draw</span>'
        f'<span class="model-metric-value">{pct(dc["P_draw"])}</span></div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="model-metric-row"><span class="model-metric-label">Away Win</span>'
        f'<span class="model-metric-value">{pct(dc["P_away"])}</span></div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="model-metric-row"><span class="model-metric-label">BTTS</span>'
        f'<span class="model-metric-value">{pct(dc["P_BTTS"])}</span></div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-title" style="margin-top:8px;">Poisson scorelines adjusted for low-score dependence (improved probabilities around 0–0, 1–0, 1–1, etc.).</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================
#  ADJUSTMENT IMPACT
# ==============================
impact_color = "green" if home_adjustment_delta > 0 else ("red" if home_adjustment_delta < 0 else "#9CA3AF")

st.markdown(
    f"""
    <div class="model-card" style="margin-top:4px;">
        <strong>Adjustment impact</strong><br/>
        Your match context adjustment changed the <strong>home win probability (Dixon–Coles)</strong> by
        <span style="color:{impact_color};">{home_adjustment_delta:+.2f}%</span>.
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================
#  TEAM FORM & LEAGUE POSITION
# ==============================
with st.expander("Team form & league position (last 5 matches)"):
    home_form_letters, home_form_icons = last_5_results(home_team)
    away_form_letters, away_form_icons = last_5_results(away_team)

    col_h, col_a = st.columns(2)

    with col_h:
        st.markdown(f"**{home_team}**")
        st.markdown(f"League position: **{get_team_position(home_team)}**")
        st.markdown(f"Last 5: {home_form_icons}  ({home_form_letters})")

    with col_a:
        st.markdown(f"**{away_team}**")
        st.markdown(f"League position: **{get_team_position(away_team)}**")
        st.markdown(f"Last 5: {away_form_icons}  ({away_form_letters})")

st.markdown(
    '<div class="footer-note">'
    'Please remember probabilities are model-based estimates only. Underlying data and models update as new GWs are added.'
    '</div>',
    unsafe_allow_html=True
)
