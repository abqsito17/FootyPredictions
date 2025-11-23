import sys, os, base64
import streamlit as st
import pandas as pd
from PIL import Image
from src.data_loader import load_team_data
from src.predictor import load_model, predict_match, generate_insights

#Page Configuration
st.set_page_config(page_title="Premier League Predictor ⚽", page_icon="⚽", layout="wide")

#Background Setup
def set_background(image_file: str):
    abs_path = os.path.abspath(image_file)
    if not os.path.exists(abs_path):
        st.warning(f"⚠️ Background image not found: {abs_path}")
        return
    with open(abs_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(0, 0, 0, 0.45);
        z-index: 0;
    }}
    .stApp * {{
        position: relative;
        z-index: 1;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background("assets/background/bg-premier.jpg")

#Loading Custom CSS
import time
css_version = int(time.time())
css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(
            f"<style id='custom-style-{css_version}'>{f.read()}</style>",
            unsafe_allow_html=True
        )

# Helper Functions
def file_to_data_uri(path: str) -> str | None:
    if not os.path.exists(path): return None
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(path)[1].lower().strip(".") or "png"
    return f"data:image/{ext};base64,{b64}"

def team_logo_uri(team: str):
    path = os.path.join("assets", "logos", f"{team}.png")
    return file_to_data_uri(path)

def form_to_letters(form_str: str) -> str | None:
    """Convert '[1, 0, -1, 1]' → '🟢 ⚪ 🔴 🟢'."""
    if pd.isna(form_str):
        return None
    s = str(form_str).strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    if not s:
        return None

    # 🔥 Emoji mapping
    mapping = {
        1: "🟢",   # Win
        0: "⚪",   # Draw
        -1: "🔴"   # Loss
    }

    symbols = []
    for part in s.split(","):
        part = part.strip()
        try:
            val = int(part)
        except ValueError:
            continue
        symbols.append(mapping.get(val, "❓"))

    return " ".join(symbols) if symbols else None


def get_team_overview(team_name: str,
                      overview_df: pd.DataFrame,
                      stats_df: pd.DataFrame | None = None):
    row = overview_df[overview_df["team"] == team_name]
    if row.empty:
        return {}

    data = row.iloc[0]
    info = {
        "Position": int(data["position"]),
        "Played": int(data["played"]),
        "Wins": int(data["wins"]),
        "Draws": int(data["draws"]),
        "Losses": int(data["losses"]),
        "Goals For": int(data["goals_for"]),
        "Goals Against": int(data["goals_against"]),
        "Points": int(data["points"]),
    }

    # Add recent form from pl_team_stats.csv if available
    if stats_df is not None and "form_last_5" in stats_df.columns:
        srow = stats_df[stats_df["team"] == team_name]
        if not srow.empty:
            form_letters = form_to_letters(srow.iloc[0]["form_last_5"])
            if form_letters:
                info["Form"] = form_letters

    return info

# Load Data + Model
df = load_team_data()
team_overview = pd.read_csv("data/team_overview.csv")
team_stats = pd.read_csv("data/pl_team_stats.csv")
model, le, class_order = load_model()
teams = sorted(df["team"].unique())

# Header Section
pl_logo_path = os.path.join("assets", "logos", "premier-league.png")
pl_logo_uri = file_to_data_uri(pl_logo_path)
if pl_logo_uri:
    st.markdown(f"""
        <div class='header'>
            <img class='pl-logo' src='{pl_logo_uri}' style='width:160px; margin-bottom:4px;'>
            <h3>Score Predictor</h3>
        </div>
    """, unsafe_allow_html=True)

# Team Selection Section
col1, col_mid, col2 = st.columns([2, 1, 2])

# Home team
with col1:
    st.markdown("<div class='column-align'>", unsafe_allow_html=True)

    home_team = st.selectbox(
        "Select home team",
        teams,
        key="home_team_select"
    )

    box_html = f"""
    <div class="team-box">
        <div class="diamond-container">
            <div class="diamond"></div>
            <img src="{team_logo_uri(home_team)}" class="team-logo" alt="{home_team} logo">
        </div>
        <h4>{home_team}</h4>
    """

    stats = get_team_overview(home_team, team_overview, team_stats)
    for k, v in stats.items():
        box_html += f"<p><strong>{k}:</strong> {v}</p>"

    box_html += "</div>"

    st.markdown(box_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Away team
with col2:
    st.markdown("<div class='column-align'>", unsafe_allow_html=True)

    away_team = st.selectbox(
        "Select away team",
        teams,
        key="away_team_select"
    )

    box_html = f"""
    <div class="team-box">
        <div class="diamond-container">
            <div class="diamond"></div>
            <img src="{team_logo_uri(away_team)}" class="team-logo" alt="{away_team} logo">
        </div>
        <h4>{away_team}</h4>
    """

    stats = get_team_overview(away_team, team_overview, team_stats)
    for k, v in stats.items():
        box_html += f"<p><strong>{k}:</strong> {v}</p>"

    box_html += "</div>"

    st.markdown(box_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

with col_mid:
    st.markdown("<div class='center-align'>", unsafe_allow_html=True)
    st.markdown("<p class='vs-text'>🤜 VS 🤛</p>", unsafe_allow_html=True)
    st.markdown("<h5>Match Probabilities</h5>", unsafe_allow_html=True)
    if st.button("Predict", key="predict_button", use_container_width=True):
        probs, label, home, away = predict_match(model, le, class_order, df, home_team, away_team)
        labels = ["Home Win", "Draw", "Away Win"]
        for lbl, prob in zip(labels, probs):
            st.markdown(f"""
            <div class="prediction-result">
                <p class="prediction-label">{lbl}</p>
                <p class="prediction-value">{prob*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<h5>Insights</h5>", unsafe_allow_html=True)
        insights = generate_insights(home, away, home_team, away_team)
        for i in insights:
            st.markdown(f"<p>{i}</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='prediction-placeholder'>(Press \"Predict\" to view results)</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<hr style="margin-top: 60px; border: 1px solid rgba(255,255,255,0.1);">
<p style="text-align: center; color: #ccc; font-size: 14px;">
    Built by <strong>Ahmed Qasim</strong> 🧠 | Premier League Score Predictor v1.0
</p>
""", unsafe_allow_html=True)
