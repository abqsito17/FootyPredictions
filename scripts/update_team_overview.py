import requests
import pandas as pd
import os

from dotenv import load_dotenv
load_dotenv()

def update_team_overview():
    API_KEY = os.getenv("FOOTBALL_API_KEY")  # secure environment variable
    if not API_KEY:
        raise ValueError("Missing FOOTBALL_API_KEY environment variable.")

    url = "https://api.football-data.org/v4/competitions/PL/standings"
    headers = {"X-Auth-Token": API_KEY}

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.status_code}")

    data = response.json()
    standings = data["standings"][0]["table"]

    teams_data = []
    for team in standings:
        teams_data.append({
            "team": team["team"]["name"],
            "position": team["position"],
            "played": team["playedGames"],
            "wins": team["won"],
            "draws": team["draw"],
            "losses": team["lost"],
            "goals_for": team["goalsFor"],
            "goals_against": team["goalsAgainst"],
            "points": team["points"]
        })

    df = pd.DataFrame(teams_data)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/team_overview.csv", index=False)
    print("✅ Team overview updated successfully!")

if __name__ == "__main__":
    update_team_overview()
