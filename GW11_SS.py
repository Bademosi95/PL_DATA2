"""
Premier League Match Predictor - Streamlit App (Glassmorphism Edition)
----------------------------------------------------------------------

- iOS widget feel
- Glassmorphism (blur, translucency, depth)
- Inter font
- Custom VS header with team form dots (R/Y/G)
- Fully aligned with weekly_update.py outputs
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy.stats import poisson
from pathlib import Path
from datetime import datetime
import json

def get_model_update_version():
    """Return update timestamp from metadata.json to bust Streamlit cache."""
    try:
        if Path("metadata.json").exists():
            with open("metadata.json", "r") as f:
                metadata = json.load(f)
                return metadata.get("update_time", "0")
    except:
        pass
    return str(datetime.utcnow())


# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Premier League Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════
# CSS: GLASSMORPHISM + INTER FONT + VS HEADER
# ═══════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
    /* Font system: Inter + system fallbacks */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }

    /* Background: deep gradient for glassmorphism */
    .stApp {
        background: radial-gradient(circle at 0% 0%, #4f46e5 0, #111827 40%, #020617 100%) !important;
        color: #e5e7eb;
    }

    .main .block-container {
        padding: 1.6rem 2.2rem;
        max-width: 1320px;
    }

    /* Base glass panel */
    .glass-panel {
        background: rgba(15, 23, 42, 0.55);
        border-radius: 24px;
        padding: 2rem 2rem 1.75rem 2rem;
        margin-bottom: 1.8rem;
        border: 1px solid rgba(148, 163, 184, 0.5);
        box-shadow:
            0 18px 45px rgba(15, 23, 42, 0.65),
            0 0 0 1px rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(22px);
        -webkit-backdrop-filter: blur(22px);
    }

    /* HERO */
    .hero-section {
        /* no composes; just use glass-panel-like styling */
        background: rgba(15, 23, 42, 0.55);
        border-radius: 24px;
        padding: 2rem 2rem 1.75rem 2rem;
        margin-bottom: 1.8rem;
        border: 1px solid rgba(148, 163, 184, 0.5);
        box-shadow:
            0 18px 45px rgba(15, 23, 42, 0.65),
            0 0 0 1px rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(22px);
        -webkit-backdrop-filter: blur(22px);
    }

    .hero-title {
        font-size: 30px;
        font-weight: 800;
        color: #f9fafb;
        letter-spacing: -0.04em;
        margin: 0;
    }

    .hero-subtitle {
        font-size: 13px;
        color: #cbd5f5;
        font-weight: 500;
        margin-top: 0.3rem;
    }

    /* VS header with form dots */
    .vs-header {
        margin-top: 1.5rem;
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        align-items: center;
        gap: 1.4rem;
    }

    .vs-team-block {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.25rem;
    }

    .vs-team-name {
        font-size: 20px;
        font-weight: 700;
        color: #e5e7eb;
        text-align: center;
    }

    .vs-form-row {
        display: flex;
        align-items: center;
        gap: 0.25rem;
        margin-top: 0.1rem;
    }

    .form-dot {
        width: 7px;
        height: 7px;
        border-radius: 999px;
        display: inline-block;
        box-shadow: 0 0 0 1px rgba(15, 23, 42, 0.9);
    }

    .form-dot-win { background: #22c55e; }
    .form-dot-draw { background: #eab308; }
    .form-dot-loss { background: #ef4444; }

    .vs-center {
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .vs-chip {
        width: 56px;
        height: 56px;
        border-radius: 999px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: radial-gradient(circle at 30% 0%, #a5b4fc 0, #4f46e5 40%, #1d2a49 100%);
        color: white;
        font-weight: 800;
        letter-spacing: 0.12em;
        font-size: 13px;
        text-transform: uppercase;
        box-shadow:
            0 0 0 1px rgba(191, 219, 254, 0.65),
            0 16px 35px rgba(15, 23, 42, 0.8);
    }

    .vs-subtext {
        margin-top: 0.4rem;
        font-size: 11px;
        color: #9ca3af;
        text-align: center;
    }

    /* Glass cards */
    .glass-card {
        background: rgba(15, 23, 42, 0.58);
        border-radius: 20px;
        padding: 1.4rem 1.5rem;
        border: 1px solid rgba(148, 163, 184, 0.6);
        box-shadow:
            0 14px 35px rgba(15, 23, 42, 0.9),
            0 0 0 1px rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        transition: all 0.16s ease-out;
        height: 100%;
    }

    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow:
            0 18px 40px rgba(15, 23, 42, 1.0),
            0 0 0 1px rgba(15, 23, 42, 0.9);
    }

    .card-title {
        font-size: 15px;
        font-weight: 700;
        color: #e5e7eb;
        margin: 0 0 0.9rem 0;
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }

    .card-icon { font-size: 18px; }

    /* Headline widget metrics */
    .headline-metric {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.8), rgba(15, 23, 42, 0.4));
        border-radius: 18px;
        padding: 1rem 1.1rem;
        text-align: left;
        box-shadow:
            0 12px 30px rgba(15, 23, 42, 0.9),
            0 0 0 1px rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.9);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        transition: all 0.15s ease-out;
    }

    .headline-metric:hover {
        transform: translateY(-1px);
        box-shadow:
            0 16px 35px rgba(15, 23, 42, 1.0),
            0 0 0 1px rgba(15, 23, 42, 0.9);
    }

    .headline-label {
        font-size: 11px;
        font-weight: 600;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.11em;
        margin-bottom: 0.25rem;
    }

    .headline-value {
        font-size: 26px;
        font-weight: 800;
        color: #e5e7eb;
    }

    /* Metric rows */
    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.52rem 0;
        border-bottom: 1px solid rgba(148, 163, 184, 0.35);
    }

    .metric-row:last-child { border-bottom: none; }

    .metric-label {
        font-size: 13px;
        font-weight: 500;
        color: #cbd5f5;
    }

    .metric-value {
        font-size: 13px;
        font-weight: 600;
        color: #e5e7eb;
        padding: 0.24rem 0.7rem;
        background: rgba(15, 23, 42, 0.7);
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.75);
    }

    .metric-value-highlight {
        font-size: 13px;
        font-weight: 700;
        background: linear-gradient(135deg, #4f46e5, #a855f7);
        color: #ffffff;
        padding: 0.26rem 0.85rem;
        border-radius: 999px;
        box-shadow: 0 0 0 1px rgba(191, 219, 254, 0.5);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.85);
        border-right: 1px solid rgba(148, 163, 184, 0.7);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        color: #e5e7eb;
    }

    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #e5e7eb !important;
        font-weight: 600 !important;
    }

    /* Section headers */
    .section-header {
        font-size: 18px;
        font-weight: 700;
        color: #e5e7eb;
        margin: 1.4rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .section-divider {
        height: 1px;
        background: linear-gradient(to right, rgba(148, 163, 184, 0.0), rgba(148, 163, 184, 0.75), rgba(148, 163, 184, 0.0));
        margin: 1.6rem 0 1.6rem 0;
        border: none;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    .badge-success {
        background: linear-gradient(135deg, #22c55e, #16a34a);
        color: #ffffff;
        box-shadow: 0 8px 20px rgba(22, 163, 74, 0.7);
    }

    .badge-warning {
        background: linear-gradient(135deg, #f97316, #ea580c);
        color: #ffffff;
        box-shadow: 0 8px 20px rgba(234, 88, 12, 0.7);
    }

    .badge-neutral {
        background: linear-gradient(135deg, #64748b, #4b5563);
        color: #ffffff;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.7);
    }

    /* Footer */
    .footer-section {
        background: rgba(15, 23, 42, 0.9);
        border-radius: 18px;
        padding: 1.4rem;
        margin-top: 2rem;
        text-align: center;
        border: 1px solid rgba(148, 163, 184, 0.9);
        box-shadow:
            0 16px 40px rgba(15, 23, 42, 1.0),
            0 0 0 1px rgba(15, 23, 42, 0.9);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
    }

    .footer-text {
        font-size: 12px;
        color: #cbd5f5;
        line-height: 1.7;
    }

    /* Hide default streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Fade-in animation */
    @keyframes fadeInSoft {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .hero-section, .glass-card, .headline-metric {
        animation: fadeInSoft 0.35s ease-out;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1.1rem !important;
        }

        .hero-section {
            padding: 1.5rem 1.3rem 1.2rem 1.3rem;
        }

        .hero-title {
            font-size: 24px;
        }

        .vs-header {
            grid-template-columns: 1fr;
            gap: 0.75rem;
        }

        .vs-center {
            order: -1;
        }

        .headline-value {
            font-size: 22px;
        }

        .glass-card {
            padding: 1.2rem 1.1rem;
        }
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# LOAD MODELS
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource
def load_models(_version):
    required_files = {
        "pipe_result_final.pkl": "Logistic regression model",
        "poisson_model.pkl": "Poisson GLM model",
        "long_df.pkl": "Match history data",
        "stats.pkl": "Team statistics",
        "rho_hat.pkl": "Dixon-Coles rho parameter",
        "feature_cols.pkl": "Feature column order",
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
                "strength_diff",
                "defense_diff",
                "rolling_points_diff",
                "rolling_xG_diff",
                "rolling_xGA_diff",
                "rolling_GD_diff",
                "finishing_overperf_diff",
                "def_overperf_diff",
            ]

        metadata = {}
        if Path("metadata.json").exists():
            with open("metadata.json", "r") as f:
                metadata = json.load(f)

        stats["Squad"] = stats["Squad"].astype(str).str.strip()
        long_df["team"] = long_df["team"].astype(str).str.strip()

        return {
            "pipe_result": pipe_result,
            "poisson_model": poisson_model,
            "long_df": long_df,
            "stats": stats,
            "rho_hat": float(rho_hat) if not isinstance(rho_hat, float) else rho_hat,
            "feature_cols": feature_cols,
            "metadata": metadata,
        }
    except Exception as e:
        st.error("### ❌ Error Loading Models")
        st.error(f"```\n{str(e)}\n```")
        st.stop()


models = load_models(get_model_update_version())
pipe_result_final = models["pipe_result"]
poisson_model = models["poisson_model"]
long_df = models["long_df"]
stats = models["stats"]
rho_hat = models["rho_hat"]
feature_cols = models["feature_cols"]
metadata = models["metadata"]

# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def pct(x: float, decimals: int = 1) -> str:
    return f"{x * 100:.{decimals}f}%"


def safe_division(numerator, denominator, default: float = 0.0) -> float:
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except Exception:
        return default


def get_latest_team_stat(team: str, column: str, default: float = 0.0) -> float:
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
    except Exception:
        return default


def get_team_strength_metrics(team: str):
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
            "def_away": safe_division(row["Away_xGA"], row["Away_MP"], 0.0),
        }
    except Exception as e:
        st.error(f"Error getting strength for {team}: {e}")
        return None


def get_team_form(team: str, n: int = 5):
    """
    Returns:
        letters: e.g. 'WDLDW'
        points : total points over last n
    """
    try:
        team_matches = long_df[long_df["team"] == team].sort_values("Date").tail(n)
        letters = []
        total_points = 0
        for _, match in team_matches.iterrows():
            gf = match["goals_for"]
            ga = match["goals_against"]
            if gf > ga:
                letters.append("W")
                total_points += 3
            elif gf == ga:
                letters.append("D")
                total_points += 1
            else:
                letters.append("L")
        return "".join(letters), total_points
    except Exception:
        return "", 0


def render_form_dots_html(team: str, n: int = 5) -> str:
    """Small R/Y/G dots summarising last n games."""
    letters, _ = get_team_form(team, n=n)
    if not letters:
        return ""
    mapping = {"W": "form-dot-win", "D": "form-dot-draw", "L": "form-dot-loss"}
    dots = []
    for ch in letters:
        cls = mapping.get(ch, "form-dot-draw")
        dots.append(f'<span class="form-dot {cls}"></span>')
    return f'<div class="vs-form-row">{"".join(dots)}</div>'


def get_team_position(team: str) -> str:
    try:
        if "Position" in stats.columns:
            row = stats[stats["Squad"] == team]
            if not row.empty:
                pos = row.iloc[0]["Position"]
                if pd.notna(pos):
                    return str(pos)
        return "—"
    except Exception:
        return "—"


def build_feature_vector(home_team: str, away_team: str):
    home_strength = get_team_strength_metrics(home_team)
    away_strength = get_team_strength_metrics(away_team)
    if home_strength is None or away_strength is None:
        return None

    home_rolling = {
        "points": get_latest_team_stat(home_team, "rolling_points"),
        "xg_for": get_latest_team_stat(home_team, "rolling_xg_for"),
        "xg_against": get_latest_team_stat(home_team, "rolling_xg_against"),
        "GD": get_latest_team_stat(home_team, "rolling_GD"),
        "finishing_overperf": get_latest_team_stat(
            home_team, "rolling_finishing_overperf"
        ),
        "def_overperf": get_latest_team_stat(home_team, "rolling_def_overperf"),
    }

    away_rolling = {
        "points": get_latest_team_stat(away_team, "rolling_points"),
        "xg_for": get_latest_team_stat(away_team, "rolling_xg_for"),
        "xg_against": get_latest_team_stat(away_team, "rolling_xg_against"),
        "GD": get_latest_team_stat(away_team, "rolling_GD"),
        "finishing_overperf": get_latest_team_stat(
            away_team, "rolling_finishing_overperf"
        ),
        "def_overperf": get_latest_team_stat(away_team, "rolling_def_overperf"),
    }

    features = {
        "strength_diff": home_strength["att_home"] - away_strength["att_away"],
        "defense_diff": away_strength["def_away"] - home_strength["def_home"],
        "rolling_points_diff": home_rolling["points"] - away_rolling["points"],
        "rolling_xG_diff": home_rolling["xg_for"] - away_rolling["xg_for"],
        "rolling_xGA_diff": home_rolling["xg_against"] - away_rolling["xg_against"],
        "rolling_GD_diff": home_rolling["GD"] - away_rolling["GD"],
        "finishing_overperf_diff": home_rolling["finishing_overperf"]
        - away_rolling["finishing_overperf"],
        "def_overperf_diff": home_rolling["def_overperf"]
        - away_rolling["def_overperf"],
    }

    return pd.DataFrame([features])[feature_cols]


def predict_logistic(home_team: str, away_team: str):
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


def get_poisson_lambdas(home_team: str, away_team: str, context_adj: float = 0.0):
    try:
        pred_df = pd.DataFrame(
            {
                "team": [home_team, away_team],
                "opponent": [away_team, home_team],
                "is_home": [1, 0],
            }
        )
        predictions = poisson_model.predict(pred_df)
        lam_home_base = float(predictions.iloc[0])
        lam_away_base = float(predictions.iloc[1])

        lam_home_adj = max(lam_home_base * (1 + context_adj), 0.01)
        lam_away_adj = max(lam_away_base * (1 - context_adj), 0.01)
        return lam_home_adj, lam_away_adj, lam_home_base, lam_away_base
    except Exception as e:
        st.error(f"Poisson prediction error: {e}")
        return None


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


def compute_scoreline_probabilities(
    lam_home: float, lam_away: float, max_goals: int = 6, use_dc: bool = False, rho: float = 0.0
):
    hg = np.arange(0, max_goals + 1)
    ag = np.arange(0, max_goals + 1)

    P = np.outer(poisson.pmf(hg, lam_home), poisson.pmf(ag, lam_away))

    if use_dc and abs(rho) > 1e-6:
        for i, h in enumerate(hg):
            for j, a in enumerate(ag):
                P[i, j] *= dixon_coles_tau(h, a, lam_home, lam_away, rho)

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
        "P_clean_sheet_away": float(P[0, :].sum()),
    }

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚽ Match Setup")

    teams = sorted(stats["Squad"].unique())
    if len(teams) == 0:
        st.error("No teams found in stats file!")
        st.stop()

    home_team = st.selectbox("🏠 Home Team", teams, key="home")
    away_team = st.selectbox("✈️ Away Team", teams, key="away")

    st.markdown("---")
    st.markdown("## 🎯 Match Context")
    st.caption("Adjust for injuries, form, tactics and schedule effects.")

    advanced_mode = st.checkbox("Advanced Controls", value=False)
    if not advanced_mode:
        context_raw = st.slider(
            "Context Adjustment",
            min_value=-3.0,
            max_value=3.0,
            value=0.0,
            step=0.1,
            help="Negative = Away advantage | Positive = Home advantage",
        )
        context_adj = context_raw / 10.0
    else:
        att_adj = st.slider("⚔️ Attack", -3.0, 3.0, 0.0, 0.1)
        def_adj = st.slider("🛡️ Defense", -3.0, 3.0, 0.0, 0.1)
        morale_adj = st.slider("💪 Morale", -3.0, 3.0, 0.0, 0.1)
        context_adj = (att_adj + def_adj + 0.5 * morale_adj) / 20.0

    if abs(context_adj) < 0.01:
        st.markdown(
            '<span class="status-badge badge-neutral">⚖️ Neutral Context</span>',
            unsafe_allow_html=True,
        )
    elif context_adj > 0:
        st.markdown(
            f'<span class="status-badge badge-success">🏠 Home +{context_adj:.2f}</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<span class="status-badge badge-warning">✈️ Away {context_adj:.2f}</span>',
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════════════
# MAIN – HERO
# ═══════════════════════════════════════════════════════════════════

if home_team == away_team:
    st.error("### ⚠️ Please select different teams")
    st.stop()

home_form_dots_html = render_form_dots_html(home_team)
away_form_dots_html = render_form_dots_html(away_team)

hero_html = f"""
<div class="hero-section">
    <div class="hero-title">Premier League Predictor</div>
    <div class="hero-subtitle">AI-powered match probabilities from your weekly modelling pipeline.</div>
    <div class="vs-header">
        <div class="vs-team-block">
            <div class="vs-team-name">{home_team}</div>
            {home_form_dots_html}
        </div>
        <div class="vs-center">
            <div class="vs-chip">VS</div>
        </div>
        <div class="vs-team-block">
            <div class="vs-team-name">{away_team}</div>
            {away_form_dots_html}
        </div>
    </div>
    <div class="vs-subtext">
        Form dots: green = win, yellow = draw, red = loss (last 5 matches)
    </div>
</div>
"""

st.markdown(hero_html, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# PREDICTIONS
# ═══════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════
# HEADLINE WIDGETS
# ═══════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">🎯 Headline Prediction</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="headline-metric">
            <div class="headline-label">🏠 Home Win</div>
            <div class="headline-value">{pct(dc_adj["P_home"])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="headline-metric">
            <div class="headline-label">🤝 Draw</div>
            <div class="headline-value">{pct(dc_adj["P_draw"])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="headline-metric">
            <div class="headline-label">✈️ Away Win</div>
            <div class="headline-value">{pct(dc_adj["P_away"])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="headline-metric">
            <div class="headline-label">⚽ BTTS</div>
            <div class="headline-value">{pct(dc_adj["P_BTTS"])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if abs(context_adj) > 0.01:
    delta = (dc_adj["P_home"] - dc_base["P_home"]) * 100
    st.markdown(
        f"<p style='text-align:center; color:#e5e7eb; margin-top:0.9rem;'>"
        f"💡 Context adjustment changed home win by <strong>{delta:+.1f}pp</strong>"
        f"</p>",
        unsafe_allow_html=True,
    )

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# MODEL CARDS
# ═══════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">📊 Model Comparison</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

# Logistic
with c1:
    st.markdown(
        f"""
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
            <p style="margin-top:1rem; font-size:12px; color:#e5e7eb; line-height:1.6;">
                Historical result-based prediction using team strength and 5-game form.
                Context-independent baseline.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Poisson
with c2:
    st.markdown(
        f"""
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
            <hr style="margin:0.75rem 0; border:none; border-top:1px solid rgba(148, 163, 184, 0.35);">
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
            <p style="margin-top:1rem; font-size:12px; color:#e5e7eb; line-height:1.6;">
                Goal expectation model. Context adjustments affect xG values.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Dixon–Coles
with c3:
    st.markdown(
        f"""
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
            <hr style="margin:0.75rem 0; border:none; border-top:1px solid rgba(148, 163, 184, 0.35);">
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
            <p style="margin-top:1rem; font-size:12px; color:#e5e7eb; line-height:1.6;">
                Enhanced Poisson with correlation adjustment (rho={rho_hat:.3f}).
                Corrects low-scoring dependencies.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# BETTING MARKETS
# ═══════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">🎰 Betting Markets</div>', unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown(
        f"""
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
        """,
        unsafe_allow_html=True,
    )

with col_b:
    st.markdown(
        f"""
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
        """,
        unsafe_allow_html=True,
    )

with col_c:
    st.markdown(
        f"""
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
        """,
        unsafe_allow_html=True,
    )

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════

st.markdown(
    """
<div class="footer-section">
    <div class="footer-text">
        <strong>⚠️ Disclaimer</strong><br>
        Predictions are statistical models based on historical data. Football is inherently unpredictable.<br>
        Many factors (injuries, weather, motivation, referee decisions) cannot be fully modelled.<br><br>
        <strong>📊 Models:</strong> Logistic (outcome-based) • Poisson (goal-based) • Dixon-Coles (correlation-adjusted)<br>
        <strong>🔄 Updates:</strong> Models retrained weekly with latest match data
    </div>
</div>
""",
    unsafe_allow_html=True,
)

