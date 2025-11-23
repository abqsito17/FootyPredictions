# src/predictor.py

import joblib
import numpy as np
import pandas as pd
import json
from src.constants import MODEL_PATH, LABEL_ENCODER_PATH


def load_model():
    """Load trained model, label encoder, and class order."""
    model = joblib.load(MODEL_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)

    # ✅ Load class order (saved during training)
    try:
        with open("model_classes.json", "r") as f:
            class_order = json.load(f)
    except FileNotFoundError:
        print("⚠️ model_classes.json not found — using default class order.")
        class_order = list(le.classes_)

    return model, le, class_order


def predict_match(model, le, class_order, df, home_team, away_team):
    """Predict match outcome and return probabilities + label."""
    home = df[df["team"] == home_team].iloc[0]
    away = df[df["team"] == away_team].iloc[0]

    # ✅ Make sure feature order matches the training
    feature_data = {
        "home_points": [home["points"]],
        "away_points": [away["points"]],
        "home_goal_diff": [home["goal_diff"]],
        "away_goal_diff": [away["goal_diff"]],
        "home_form": [home["form_total"]],
        "away_form": [away["form_total"]],
        "home_weighted_form": [home["strength_weighted_form"]],
        "away_weighted_form": [away["strength_weighted_form"]],
    }

    X_new = pd.DataFrame(feature_data)

    # Predict probabilities
    probs = model.predict_proba(X_new)[0]

    # Map probabilities to correct class order
    prob_map = dict(zip(class_order, probs))

    # ✅ Reorder to football-friendly logic
    ordered_labels = ["HomeWin", "Draw", "AwayWin"]
    ordered_probs = [prob_map.get(lbl, 0) for lbl in ordered_labels]

    # Get final label based on highest probability
    label = ordered_labels[np.argmax(ordered_probs)]

    return ordered_probs, label, home, away


def generate_insights(home, away, home_team, away_team):
    """Generate brief match insights based on stats."""
    insights = []

    if home["points"] > away["points"]:
        insights.append(f"{home_team} have higher points ({home['points']} vs {away['points']}).")
    elif away["points"] > home["points"]:
        insights.append(f"{away_team} have higher points ({away['points']} vs {home['points']}).")

    if home["goal_diff"] > away["goal_diff"]:
        insights.append(f"{home_team} have a stronger goal difference.")
    elif away["goal_diff"] > home["goal_diff"]:
        insights.append(f"{away_team} have a stronger goal difference.")

    if home["form_total"] > away["form_total"]:
        insights.append(f"{home_team} have better recent form.")
    elif away["form_total"] > home["form_total"]:
        insights.append(f"{away_team} have better recent form.")

    return insights
