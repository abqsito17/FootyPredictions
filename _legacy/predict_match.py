import joblib
import numpy as np
import pandas as pd
import json
from src.constants import MODEL_PATH, LABEL_ENCODER_PATH

def load_model():
    model = joblib.load(MODEL_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    with open("model_classes.json", "r") as f:
        class_order = json.load(f)
    return model, le, class_order

def predict_match(model, le, class_order, df, home_team, away_team):
    home = df[df["team"] == home_team].iloc[0]
    away = df[df["team"] == away_team].iloc[0]

    # ✅ Ensure feature order matches training exactly
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
    prob_map = dict(zip(class_order, probs))

    # ✅ Reorder to football logic
    ordered_labels = ["HomeWin", "Draw", "AwayWin"]
    ordered_probs = [prob_map.get(lbl, 0) for lbl in ordered_labels]
    label = ordered_labels[np.argmax(ordered_probs)]

    return ordered_probs, label, home, away
