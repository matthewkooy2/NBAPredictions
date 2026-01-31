"""
Feature engineering functions for NBA player stats prediction.

This module provides functions to compute features from raw game log data:
- Rolling statistics (averages, std dev)
- Rest days and back-to-back game flags
- Home/away game indicators
- Opponent-specific features (defensive rating, pace)
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def compute_rolling_stats(
    df: pd.DataFrame,
    stat_columns: List[str] = ["PTS", "REB", "AST", "MIN"],
    windows: List[int] = [5, 10]
) -> pd.DataFrame:
    """
    Compute rolling averages and standard deviations for player stats.

    CRITICAL: This function shifts stats by 1 game to prevent data leakage.
    Features for game G only use data from games BEFORE G.

    Args:
        df: DataFrame with player game logs (must be sorted by GAME_DATE ascending)
        stat_columns: List of stat columns to compute rolling features for
        windows: List of window sizes for rolling calculations

    Returns:
        DataFrame with added rolling stat columns
    """
    df = df.copy()

    # Verify data is sorted by date
    if not df["GAME_DATE"].is_monotonic_increasing:
        raise ValueError("DataFrame must be sorted by GAME_DATE in ascending order")

    for stat in stat_columns:
        if stat not in df.columns:
            raise ValueError(f"Column '{stat}' not found in DataFrame")

        for window in windows:
            # Shift by 1 to prevent leakage (use only past games)
            # Example: For game 10, last_5 uses games 5-9 (not including game 10)
            shifted = df[stat].shift(1)

            # Rolling mean
            col_name = f"{stat.lower()}_last_{window}"
            df[col_name] = shifted.rolling(window=window, min_periods=1).mean()

            # Rolling std dev (for volatility/consistency metrics)
            if window == 10:  # Only compute std for 10-game window
                std_col_name = f"{stat.lower()}_std_last_{window}"
                df[std_col_name] = shifted.rolling(window=window, min_periods=1).std()

    return df


def compute_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute days of rest since last game and back-to-back flags.

    Args:
        df: DataFrame with GAME_DATE column (must be sorted by GAME_DATE ascending)

    Returns:
        DataFrame with added columns:
        - days_rest: Integer days since last game (NaN for first game)
        - is_b2b: Binary flag (1 if back-to-back, 0 otherwise)
    """
    df = df.copy()

    if not df["GAME_DATE"].is_monotonic_increasing:
        raise ValueError("DataFrame must be sorted by GAME_DATE in ascending order")

    # Calculate days between games
    df["days_rest"] = df["GAME_DATE"].diff().dt.days

    # Back-to-back flag: 1 if days_rest == 1, else 0
    # For first game, days_rest is NaN, so is_b2b will be 0
    df["is_b2b"] = (df["days_rest"] == 1).astype(int)

    # Fill NaN in days_rest with 0 for first game of season
    # (Or use a high value like 7 to indicate "well-rested")
    df["days_rest"] = df["days_rest"].fillna(7)

    return df


def add_home_away(df: pd.DataFrame, matchup_col: str = "MATCHUP") -> pd.DataFrame:
    """
    Parse the MATCHUP column to determine if game is home or away.

    The MATCHUP column format from nba_api is:
    - "LAL vs. BOS" (home game for LAL)
    - "LAL @ BOS" (away game for LAL)

    Args:
        df: DataFrame with MATCHUP column
        matchup_col: Name of the matchup column (default: "MATCHUP")

    Returns:
        DataFrame with added columns:
        - home_away: String ("HOME" or "AWAY")
        - is_home: Binary flag (1 if home, 0 if away)
    """
    df = df.copy()

    if matchup_col not in df.columns:
        raise ValueError(f"Column '{matchup_col}' not found in DataFrame")

    # Extract home/away from matchup string
    # "vs." indicates home game, "@" indicates away game
    df["home_away"] = df[matchup_col].apply(
        lambda x: "HOME" if "vs." in x else "AWAY"
    )

    # Binary encoding: 1 for home, 0 for away
    df["is_home"] = (df["home_away"] == "HOME").astype(int)

    return df


def extract_opponent_id(df: pd.DataFrame, matchup_col: str = "MATCHUP") -> pd.DataFrame:
    """
    Extract opponent team abbreviation from MATCHUP column.

    Args:
        df: DataFrame with MATCHUP column
        matchup_col: Name of the matchup column

    Returns:
        DataFrame with added column:
        - opponent_abbr: Opponent team abbreviation (e.g., "BOS", "LAL")
    """
    df = df.copy()

    if matchup_col not in df.columns:
        raise ValueError(f"Column '{matchup_col}' not found in DataFrame")

    # Extract opponent abbreviation
    # Format: "LAL vs. BOS" or "LAL @ BOS"
    # We want the part after "vs." or "@"
    def get_opponent(matchup: str) -> str:
        if "vs." in matchup:
            return matchup.split("vs.")[1].strip()
        elif "@" in matchup:
            return matchup.split("@")[1].strip()
        else:
            return ""

    df["opponent_abbr"] = df[matchup_col].apply(get_opponent)

    return df


def add_opponent_features(
    df: pd.DataFrame,
    team_stats_df: pd.DataFrame,
    merge_on: str = "opponent_abbr"
) -> pd.DataFrame:
    """
    Add opponent team statistics (defensive rating, pace, etc.) to player game logs.

    Args:
        df: Player game log DataFrame (must have opponent_abbr column)
        team_stats_df: League team stats DataFrame from get_league_team_stats()
        merge_on: Column to merge on (default: "opponent_abbr")

    Returns:
        DataFrame with added opponent features:
        - opp_def_rating: Opponent defensive rating
        - opp_pace: Opponent pace
        - opp_net_rating: Opponent net rating (off - def)
    """
    df = df.copy()
    team_stats_df = team_stats_df.copy()

    if merge_on not in df.columns:
        raise ValueError(f"Column '{merge_on}' not found in player DataFrame. "
                        "Run extract_opponent_id() first.")

    # Map team abbreviation for merging
    # team_stats_df has TEAM_ABBREVIATION, we need to match with opponent_abbr
    if "TEAM_ABBREVIATION" not in team_stats_df.columns:
        raise ValueError("team_stats_df must have 'TEAM_ABBREVIATION' column")

    # Select relevant columns and rename for clarity
    opponent_stats = team_stats_df[[
        "TEAM_ABBREVIATION",
        "DEF_RATING",
        "PACE",
        "NET_RATING"
    ]].copy()

    opponent_stats = opponent_stats.rename(columns={
        "TEAM_ABBREVIATION": merge_on,
        "DEF_RATING": "opp_def_rating",
        "PACE": "opp_pace",
        "NET_RATING": "opp_net_rating"
    })

    # Merge opponent stats
    df = df.merge(opponent_stats, on=merge_on, how="left")

    # Handle missing values (if opponent not found in team stats)
    # Fill with league average or median
    for col in ["opp_def_rating", "opp_pace", "opp_net_rating"]:
        if df[col].isna().any():
            median_val = team_stats_df[col.replace("opp_", "").upper()].median()
            df[col] = df[col].fillna(median_val)

    return df


def build_features(
    player_gamelog: pd.DataFrame,
    team_stats: Optional[pd.DataFrame] = None,
    stat_columns: List[str] = ["PTS", "REB", "AST", "MIN"],
    windows: List[int] = [5, 10]
) -> pd.DataFrame:
    """
    All-in-one function to build all features from raw player game log.

    This is a convenience function that applies all feature engineering steps
    in the correct order.

    Args:
        player_gamelog: Raw player game log from get_player_gamelog()
        team_stats: League team stats from get_league_team_stats() (optional)
        stat_columns: Stats to compute rolling features for
        windows: Window sizes for rolling calculations

    Returns:
        DataFrame with all features added
    """
    # Ensure sorted by date
    df = player_gamelog.sort_values("GAME_DATE").reset_index(drop=True)

    # 1. Add home/away indicators
    df = add_home_away(df)

    # 2. Extract opponent
    df = extract_opponent_id(df)

    # 3. Add opponent features if team stats provided
    if team_stats is not None:
        df = add_opponent_features(df, team_stats)

    # 4. Compute rest days
    df = compute_rest_days(df)

    # 5. Compute rolling stats (MUST BE LAST to prevent leakage)
    df = compute_rolling_stats(df, stat_columns, windows)

    return df


# Test the functions when run directly
if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.nba_client import NBAClient

    print("Testing Feature Engineering Functions...")
    print("-" * 60)

    # Initialize client
    client = NBAClient()

    # Get a player's game log
    print("\n1. Fetching player data (Nikola Jokic, 2024-25)...")
    player_id = client.get_player_id("Nikola Jokic")
    gamelog = client.get_player_gamelog(player_id, "2024-25")
    print(f"   Retrieved {len(gamelog)} games")

    # Get team stats
    print("\n2. Fetching league team stats...")
    team_stats = client.get_league_team_stats("2024-25")
    print(f"   Retrieved stats for {len(team_stats)} teams")

    # Build features
    print("\n3. Building features...")
    featured_df = build_features(gamelog, team_stats)

    print(f"\n4. Feature columns added:")
    new_cols = [col for col in featured_df.columns if col not in gamelog.columns]
    for col in new_cols:
        print(f"   - {col}")

    print(f"\n5. Sample of latest game with features:")
    display_cols = [
        "GAME_DATE", "MATCHUP", "PTS", "REB", "AST",
        "home_away", "days_rest", "is_b2b",
        "pts_last_5", "pts_last_10", "opp_def_rating", "opp_pace"
    ]
    available_cols = [col for col in display_cols if col in featured_df.columns]
    print(featured_df[available_cols].tail(3).to_string(index=False))

    print("\n" + "-" * 60)
    print("Feature engineering test completed!")
