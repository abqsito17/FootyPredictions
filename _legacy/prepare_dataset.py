import pandas as pd

# Load both datasets
matches = pd.read_csv("pl_matches_basic.csv", parse_dates=["utc_date"])
features = pd.read_csv("team_features.csv", parse_dates=["date"])

# Merge home team stats
merged = matches.merge(
    features,
    left_on=["home_team", "utc_date"],
    right_on=["team", "date"],
    how="left",
    suffixes=("", "_home")
)

# Merge away team stats
merged = merged.merge(
    features,
    left_on=["away_team", "utc_date"],
    right_on=["team", "date"],
    how="left",
    suffixes=("_home", "_away")
)

# Drop redundant columns
merged = merged.drop(columns=["team_home", "team_away", "date_home", "date_away"])

# Add additional calculated features
merged["goal_diff"] = merged["home_score"] - merged["away_score"]
merged["home_advantage"] = 1  # Home = 1, Away = 0
merged["form_diff"] = merged["recent_form_home"] - merged["recent_form_away"]
merged["rest_diff"] = merged["rest_days_home"].fillna(0) - merged["rest_days_away"].fillna(0)

# Replace any missing values
merged = merged.fillna(0)

# Encode target variable for model
merged["result"] = merged.apply(
    lambda row: "HomeWin" if row["home_score"] > row["away_score"]
    else ("AwayWin" if row["away_score"] > row["home_score"] else "Draw"),
    axis=1
)

def compute_team_records(df_matches):
    records = []
    teams = sorted(set(df_matches["home_team"]) | set(df_matches["away_team"]))

    for team in teams:
        team_data = df_matches[
            (df_matches["home_team"] == team) | (df_matches["away_team"] == team)
        ].sort_values("utc_date")

        home_records = team_data[team_data["home_team"] == team]
        away_records = team_data[team_data["away_team"] == team]

        records.append({
            "team": team,
            "home_wins": sum(home_records["home_score"] > home_records["away_score"]),
            "home_draws": sum(home_records["home_score"] == home_records["away_score"]),
            "home_losses": sum(home_records["home_score"] < home_records["away_score"]),
            "away_wins": sum(away_records["away_score"] > away_records["home_score"]),
            "away_draws": sum(away_records["away_score"] == away_records["home_score"]),
            "away_losses": sum(away_records["away_score"] < away_records["home_score"]),
        })

    return pd.DataFrame(records)

team_records = compute_team_records(matches)
merged = merged.merge(team_records, left_on="home_team", right_on="team", how="left")
merged = merged.merge(team_records, left_on="away_team", right_on="team", suffixes=("_home_record", "_away_record"), how="left")
merged = merged.drop(columns=["team_home_record", "team_away_record"])


print(merged[[
    "home_team", "away_team", "utc_date", 
    "recent_form_home", "recent_form_away", 
    "rest_days_home", "rest_days_away",
    "goal_diff", "form_diff", "rest_diff", "result",    "home_wins_home_record",
    "away_losses_away_record",
    "home_draws_home_record",
    "away_draws_away_record",
]].head())

merged.to_csv("pl_dataset_ready.csv", index=False)
print("\n✅ Final dataset saved as pl_dataset_ready.csv")
