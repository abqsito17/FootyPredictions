import pandas as pd

DATA_PROCESSED = "data/pl_team_stats.csv"

def load_team_data():
    return pd.read_csv(DATA_PROCESSED)
