# scripts/train_team_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import joblib
import json
import os

# Load the team stats
df = pd.read_csv("data/pl_team_stats.csv")

# Ensure required columns are present
required_cols = {
    "team", "points", "goal_diff", "form_total", "strength_weighted_form"
}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"pl_team_stats.csv is missing columns: {missing}")

print("✅ Loaded pl_team_stats.csv with columns:", list(df.columns))

# -----------------------------
# Build synthetic matchups table
# -----------------------------
matchups = []
teams = df["team"].tolist()

for home in teams:
    for away in teams:
        if home == away:
            continue
        home_row = df.loc[df["team"] == home].iloc[0]
        away_row = df.loc[df["team"] == away].iloc[0]

        matchups.append({
            "home_team": home,
            "away_team": away,
            "home_points": home_row["points"],
            "away_points": away_row["points"],
            "home_goal_diff": home_row["goal_diff"],
            "away_goal_diff": away_row["goal_diff"],
            "home_form": home_row["form_total"],
            "away_form": away_row["form_total"],
            "home_weighted_form": home_row["strength_weighted_form"],
            "away_weighted_form": away_row["strength_weighted_form"],
        })

matchups_df = pd.DataFrame(matchups)
print("✅ Built matchups_df with columns:", list(matchups_df.columns))

# -----------------------------
# Create proxy target
# -----------------------------
matchups_df["result"] = matchups_df.apply(
    lambda r: "HomeWin" if r["home_points"] > r["away_points"]
    else ("AwayWin" if r["away_points"] > r["home_points"] else "Draw"),
    axis=1
)

# -----------------------------
# Features and label
# -----------------------------
train_features = [
    "home_points", "away_points",
    "home_goal_diff", "away_goal_diff",
    "home_form", "away_form",
    "home_weighted_form", "away_weighted_form",
]

X = matchups_df[train_features]

le = LabelEncoder()
y = le.fit_transform(matchups_df["result"])

# -----------------------------
# Split and train
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=600,
    random_state=42,
    class_weight="balanced"
)

print("\n🚀 Training model...")
model.fit(X_train, y_train)

print("\n✅ Model Performance:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# -----------------------------
# Feature importance
# -----------------------------
importances = model.feature_importances_
plt.bar(train_features, importances)
plt.title("Feature Importance")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# Save model, encoder, and class order
# -----------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/team_model.joblib")
joblib.dump(le, "models/label_encoder.joblib")

with open("model_classes.json", "w") as f:
    json.dump(list(le.classes_), f)

print("\n✅ Saved team_model.joblib, label_encoder.joblib, and model_classes.json")
