import pandas as pd
from src.constants import DATA_PROCESSED

def load_team_data():
    return pd.read_csv(DATA_PROCESSED)
