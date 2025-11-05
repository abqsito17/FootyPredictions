import sys, os, base64
from PIL import Image
import streamlit as st
import numpy as np

st.set_page_config(
    page_title="Premier League Predictor ⚽",
    page_icon="⚽",
    layout="wide",  # gives you more space for the two columns
)


# local imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.data_loader import load_team_data
from src.predictor import load_model, predict_match, generate_insights


def set_background(image_file: str):
    """Encodes and sets a background image as base64 CSS applied to Streamlit container."""
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
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }}
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(0, 0, 0, 0.45);  /* dark overlay */
        z-index: 0;
    }}
    .stApp * {{
        position: relative;
        z-index: 1;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# Set background here (before any UI)
set_background("assets/background/bg-premier.jpg")

# --- Hide Streamlit default chrome (menu, footer, etc.) ---
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Helpers
def file_to_data_uri(path: str) -> str | None:
    """Return a base64 data URI for an image file or None if missing."""
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(path)[1].lower().strip(".") or "png"
    return f"data:image/{ext};base64,{b64}"

def team_logo_uri(team: str) -> str | None:
    path = os.path.join("assets", "logos", f"{team}.png")
    return file_to_data_uri(path)


# -----------------------------
# CSS
# -----------------------------
with open("style.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# -----------------------------
# Data / Model
# -----------------------------
df = load_team_data()
model, le, class_order = load_model()
teams = sorted(df["team"].unique())


# -----------------------------
# Header
# -----------------------------
pl_logo_path = os.path.join("assets", "logos", "premier-league.png")
pl_logo_uri = file_to_data_uri(pl_logo_path)

st.markdown('<div class="header">', unsafe_allow_html=True)
if pl_logo_uri:
    st.markdown(f'<img class="pl-logo" src="{pl_logo_uri}" alt="Premier League Logo">', unsafe_allow_html=True)
st.markdown('<h3>Score Predictor</h3></div>', unsafe_allow_html=True)


# -----------------------------
# Team selection row
# -----------------------------
col1, col_mid, col2 = st.columns([2, 0.6, 2])

with col1:
    home_team = st.selectbox("Select home team:", teams, key="home")
with col_mid:
    st.markdown('<p class="vs-text">VS</p>', unsafe_allow_html=True)
with col2:
    away_team = st.selectbox("Select away team:", teams, key="away")


# -----------------------------
# Diamonds + logos
# -----------------------------
col1, _, col2 = st.columns([2, 0.6, 2])

with col1:
    h_uri = team_logo_uri(home_team)
    st.markdown('<div class="diamond-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="diamond-container"><div class="diamond"></div>', unsafe_allow_html=True)
    if h_uri:
        st.markdown(f'<img class="team-logo" src="{h_uri}" alt="{home_team} logo">', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)
    st.markdown('<p class="team-label">Pick home team</p>', unsafe_allow_html=True)

with col2:
    a_uri = team_logo_uri(away_team)
    st.markdown('<div class="diamond-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="diamond-container"><div class="diamond"></div>', unsafe_allow_html=True)
    if a_uri:
        st.markdown(f'<img class="team-logo" src="{a_uri}" alt="{away_team} logo">', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)
    st.markdown('<p class="team-label">Pick away team</p>', unsafe_allow_html=True)


# -----------------------------
# Predict button + Results
# -----------------------------
st.markdown('<div class="predict-section">', unsafe_allow_html=True)
center = st.columns([1, 1, 1])
with center[1]:
    disabled = (home_team == away_team)
    if disabled:
        st.warning("Please select two different teams.")
    if st.button("Predict", use_container_width=True, disabled=disabled):
        probs, label, home, away = predict_match(model, le, class_order, df, home_team, away_team)

        # Results
        st.markdown(f"<div class='result-label'>{label}</div>", unsafe_allow_html=True)
        st.markdown("<div class='probability-section'>", unsafe_allow_html=True)
        labels = ["Home win", "Draw", "Away win"]
        for lbl, prob in zip(labels, probs):
            st.markdown(
                f"<div class='prob'><p class='value'>{prob*100:.0f}%</p><p class='label'>{lbl}</p></div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # Insights
        insights = generate_insights(home, away, home_team, away_team)
        st.markdown("<div class='insights'><h3>Insights</h3><ul>", unsafe_allow_html=True)
        for i in insights:
            st.markdown(f"<li>{i}</li>", unsafe_allow_html=True)
        st.markdown("</ul></div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<hr style="margin-top: 60px; border: 1px solid rgba(255,255,255,0.1);">
<p style="text-align: center; color: #ccc; font-size: 14px;">
    Built by <strong>Ahmed Qasim</strong> 🧠 | Premier League Score Predictor v1.0
</p>
""", unsafe_allow_html=True)
