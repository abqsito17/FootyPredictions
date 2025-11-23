# src/utils.py
import pandas as pd
from src.constants import TEAM_STATS_PATH

# Load the dataset for team name normalization
df = pd.read_csv(TEAM_STATS_PATH)

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
    """Normalize user input to match dataset team names."""
    name = name.strip().lower()

    # Check short alias (e.g. ARS → Arsenal FC)
    code = name.upper()
    if code in TEAM_ALIAS:
        return TEAM_ALIAS[code]

    # Exact match
    for team in df["team"].values:
        if name == team.lower():
            return team

    # Partial match
    for team in df["team"].values:
        if name in team.lower():
            return team

    return None
