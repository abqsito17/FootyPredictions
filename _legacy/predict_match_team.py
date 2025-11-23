# predict_match_team.py

import pandas as pd
import joblib

df = pd.read_csv("pl_team_stats.csv")
model = joblib.load("team_model.joblib")
le = joblib.load("label_encoder.joblib")

# 👇 Short code aliases
TEAM_ALIAS = {
    "ARS": "Arsenal FC",
    "AVL": "Aston Villa FC",
    "BOU": "AFC Bournemouth",
    "BRE": "Brentford FC",
    "BRI": "Brighton & Hove Albion FC",
    "BUR": "Burnley FC",
    "CHE": "Chelsea FC",
    "CRY": "Crystal Palace FC",
    "EVE": "Everton FC",
    "FUL": "Fulham FC",
    "IPS": "Ipswich Town FC",
    "LEI": "Leicester City FC",
    "LIV": "Liverpool FC",
    "MCI": "Manchester City FC",
    "MUN": "Manchester United FC",
    "NEW": "Newcastle United FC",
    "NFO": "Nottingham Forest FC",
    "SHU": "Sheffield United FC",
    "TOT": "Tottenham Hotspur FC",
    "WHU": "West Ham United FC",
    "WOL": "Wolverhampton Wanderers FC"
}

def normalize_team_name(name: str):
    name = name.strip().lower()

    # 1️⃣ Short alias
    code = name.upper()
    if code in TEAM_ALIAS:
        return TEAM_ALIAS[code]

    # 2️⃣ Exact match
    for team in df["team"].values:
        if name == team.lower():
            return team

    # 3️⃣ Partial match
    for team in df["team"].values:
        if name in team.lower():
            return team

    return None


def predict_match(home_team, away_team):
    home = df[df["team"] == home_team].iloc[0]
    away = df[df["team"] == away_team].iloc[0]

    X_new = [[
        home["points"], away["points"],
        home["goal_diff"], away["goal_diff"],
        home["form_total"], away["form_total"],
        home["strength_weighted_form"], away["strength_weighted_form"]
    ]]


    probs = model.predict_proba(X_new)[0]
    pred_label = le.inverse_transform([probs.argmax()])[0]

    print(f"\n⚽ Prediction: {home_team} vs {away_team}")
    print(f"📊 Predicted result: {pred_label}")
    print("\n📈 Probabilities:")
    for label, prob in zip(le.classes_, probs):
        print(f"  {label}: {prob*100:.1f}%")

    print("\n🧠 Insights:")
    if home["points"] > away["points"]:
        print(f" - {home_team} have more points ({home['points']} vs {away['points']})")
    elif home["points"] < away["points"]:
        print(f" - {away_team} have more points ({away['points']} vs {home['points']})")

    if home["goal_diff"] > away["goal_diff"]:
        print(f" - {home_team} stronger goal difference ({home['goal_diff']} vs {away['goal_diff']})")
    elif home["goal_diff"] < away["goal_diff"]:
        print(f" - {away_team} stronger goal difference ({away['goal_diff']} vs {home['goal_diff']})")

    if home["form_total"] > away["form_total"]:
        print(f" - Better recent form from {home_team}")
    elif home["form_total"] < away["form_total"]:
        print(f" - Better recent form from {away_team}")
        
    if home["strength_weighted_form"] > away["strength_weighted_form"]:
        print(f" - {home_team} have faced stronger opponents recently and performed well")
    elif home["strength_weighted_form"] < away["strength_weighted_form"]:
        print(f" - {away_team} have faced stronger opponents recently and performed well")
    else:
        print(" - Both teams have similar strength-weighted form")



# 🔽 Interactive prompt
if __name__ == "__main__":
    print("⚽ Team Strength Match Predictor")
    home_input = input("Enter home team: ")
    away_input = input("Enter away team: ")

    home_team = normalize_team_name(home_input)
    away_team = normalize_team_name(away_input)

    if not home_team or not away_team:
        print("❌ Could not match one of the team names.")
    else:
        predict_match(home_team, away_team)
