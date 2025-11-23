import requests
import pandas as pd
from datetime import datetime
import os

API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
HEADERS = {"X-Auth-Token": API_KEY}

PL = "PL"
ELC = "ELC"

def fd_get(url, params=None):
    r = requests.get(url, headers=HEADERS, params=params)
    r.raise_for_status()
    return r.json()

def season_start():
    now = datetime.utcnow()
    return now.year if now.month >= 7 else now.year - 1

def pull_standings(comp, season):
    res = fd_get(f"https://api.football-data.org/v4/competitions/{comp}/standings",
                 {"season": season})
    for s in res["standings"]:
        if s["type"] == "TOTAL":
            return pd.DataFrame([{
                "team": e["team"]["name"],
                "played": e["playedGames"],
                "wins": e["won"],
                "draws": e["draw"],
                "losses": e["lost"],
                "goals_for": e["goalsFor"],
                "goals_against": e["goalsAgainst"],
                "goal_diff": e["goalDifference"],
                "points": e["points"]
            } for e in s["table"]])

def pull_matches(comp, season):
    res = fd_get(f"https://api.football-data.org/v4/competitions/{comp}/matches",
                 {"season": season})
    rows = []
    for m in res["matches"]:
        rows.append({
            "utcDate": pd.to_datetime(m["utcDate"]),
            "homeTeam": m["homeTeam"]["name"],
            "awayTeam": m["awayTeam"]["name"],
            "homeScore": m["score"]["fullTime"]["home"],
            "awayScore": m["score"]["fullTime"]["away"]
        })
    return pd.DataFrame(rows)

def result_value(home, away, hs, as_, team):
    if hs is None or as_ is None:
        return None
    if team == home:
        return 1 if hs > as_ else (0 if hs == as_ else -1)
    else:
        return 1 if as_ > hs else (0 if as_ == hs else -1)

def home_away(df, team):
    home = df[(df["homeTeam"] == team) & df["homeScore"].notna()]
    away = df[(df["awayTeam"] == team) & df["awayScore"].notna()]

    hw = sum(home["homeScore"] > home["awayScore"])
    hd = sum(home["homeScore"] == home["awayScore"])
    hl = sum(home["homeScore"] < home["awayScore"])

    aw = sum(away["awayScore"] > away["homeScore"])
    ad = sum(away["awayScore"] == away["homeScore"])
    al = sum(away["awayScore"] < away["homeScore"])

    return hw, hd, hl, aw, ad, al

def last5_weighted(df_matches, team, team_stats):
    recent = df_matches[
        ((df_matches["homeTeam"] == team) | (df_matches["awayTeam"] == team)) &
         df_matches["homeScore"].notna() &
         df_matches["awayScore"].notna()
    ].sort_values("utcDate").tail(5)

    form_vec = []
    total_form = 0
    total_weighted = 0

    for _, m in recent.iterrows():
        r = result_value(m.homeTeam, m.awayTeam, m.homeScore, m.awayScore, team)
        if r is None:
            continue

        opponent = m.awayTeam if m.homeTeam == team else m.homeTeam
        opp_stats = team_stats[team_stats["team"] == opponent].iloc[0]

        opp_rating = 0.5 * opp_stats["points"] + 0.5 * opp_stats["goal_diff"]
        if m["homeTeam"] == opponent:  # opponent was home → tougher
            opp_rating += 5

        form_vec.append(r)
        total_form += r
        total_weighted += r * opp_rating

    return form_vec, total_form, total_weighted


if __name__ == "__main__":
    SEASON = season_start()
    PREV = SEASON - 1

    standings_now = pull_standings(PL, SEASON)
    pl_matches_now = pull_matches(PL, SEASON)

    standings_prev = pull_standings(PL, PREV)
    pl_matches_prev = pull_matches(PL, PREV)
    elc_matches_prev = pull_matches(ELC, PREV)

    teams_now = set(standings_now["team"])
    teams_prev = set(standings_prev["team"])

    rows = []

    print("Building team strength dataset...")

    for team in sorted(teams_now):
        base = standings_now[standings_now["team"] == team].iloc[0].to_dict()

        hw, hd, hl, aw, ad, al = home_away(pl_matches_now, team)
        form_vec, form_total, weighted_form = last5_weighted(pl_matches_now, team, standings_now)

        src = pl_matches_prev if team in teams_prev else elc_matches_prev
        phw, phd, phl, paw, pad, pal = home_away(src, team)

        row = {
            "team": team,
            **base,
            "home_wins": hw, "home_draws": hd, "home_losses": hl,
            "away_wins": aw, "away_draws": ad, "away_losses": al,
            "weighted_home_wins": hw + 0.5*phw,
            "weighted_home_draws": hd + 0.5*phd,
            "weighted_home_losses": hl + 0.5*phl,
            "weighted_away_wins": aw + 0.5*paw,
            "weighted_away_draws": ad + 0.5*pad,
            "weighted_away_losses": al + 0.5*pal,
            "form_last_5": str(form_vec),
            "form_total": form_total,
            "strength_weighted_form": weighted_form
        }

        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv("data/pl_team_stats.csv", index=False)
    print("✅ pl_team_stats.csv generated successfully!")
    print(out.head())
