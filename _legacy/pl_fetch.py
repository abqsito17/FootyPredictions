import requests
import pandas as pd

def get_premier_league_matches(api_key):
    """
    Fetch Premier League matches (finished and upcoming)
    using football-data.org's official API.
    """
    url = "https://api.football-data.org/v4/competitions/PL/matches"
    headers = {"X-Auth-Token": "1cf4a50fc47d43c193258b0e5a013a2e"}

    response = requests.get(url, headers=headers)
    print("Response status:", response.status_code)

    if response.status_code != 200:
        print("Error: Could not fetch data.")
        print(response.text[:300])
        return None

    data = response.json()

    matches = []
    for match in data["matches"]:
        matches.append({
            "utc_date": match["utcDate"],
            "status": match["status"],
            "home_team": match["homeTeam"]["name"],
            "away_team": match["awayTeam"]["name"],
            "home_score": match["score"]["fullTime"]["home"],
            "away_score": match["score"]["fullTime"]["away"],
        })

    df = pd.DataFrame(matches)
    df["utc_date"] = pd.to_datetime(df["utc_date"])
    df = df[df["status"] == "FINISHED"]
    return df


if __name__ == "__main__":
    API_KEY = "YOUR_API_KEY_HERE"  # <-- paste your key here
    df_matches = get_premier_league_matches(API_KEY)
    if df_matches is not None:
        print(df_matches.head())
        df_matches.to_csv("pl_matches_basic.csv", index=False)
        print("\n✅ Data saved to pl_matches_basic.csv")
