"""
Dataset builder for NBA player stats prediction.

This script fetches game logs for multiple players across multiple seasons,
applies feature engineering, and saves the processed data for model training.

Usage:
    python -m src.data_builder
"""

import logging
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.nba_client import NBAClient
from src.features import build_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import configuration
try:
    from config import SEASONS, TRAIN_SEASONS, TEST_SEASON, TARGETS
except ImportError:
    logger.warning("config.py not found, using default values")
    SEASONS = ["2022-23", "2023-24", "2024-25"]
    TRAIN_SEASONS = ["2022-23", "2023-24"]
    TEST_SEASON = "2024-25"
    TARGETS = ["PTS", "REB", "AST"]

# Output directory
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_top_players(client: NBAClient, season: str = "2024-25", min_games: int = 20) -> List[dict]:
    """
    Get a list of top NBA players based on games played and minutes.

    Args:
        client: NBAClient instance
        season: Season to fetch players from
        min_games: Minimum games played to be included

    Returns:
        List of player dictionaries with id, name, and stats
    """
    logger.info(f"Fetching top players for season {season}...")

    # Get all active players
    all_players = client.get_all_players(season)
    active_players = all_players[all_players["ROSTERSTATUS"] == 1]

    top_players = []

    # Sample a diverse set of players across teams
    # Get players from each team to ensure variety
    for team_name in active_players["TEAM_NAME"].unique():
        if pd.isna(team_name) or team_name == "":
            continue

        team_players = active_players[active_players["TEAM_NAME"] == team_name]

        # Get 2-3 players per team
        for _, player_row in team_players.head(3).iterrows():
            player_id = int(player_row["PERSON_ID"])
            player_name = player_row["DISPLAY_FIRST_LAST"]

            # Check if player has enough games
            try:
                gamelog = client.get_player_gamelog(player_id, season)

                if len(gamelog) >= min_games:
                    # Calculate average minutes to filter out bench players
                    avg_min = gamelog["MIN"].mean() if "MIN" in gamelog.columns else 0

                    if avg_min >= 15:  # Only include players averaging 15+ minutes
                        top_players.append({
                            "player_id": player_id,
                            "player_name": player_name,
                            "team": team_name,
                            "games": len(gamelog),
                            "avg_min": avg_min
                        })

            except Exception as e:
                logger.debug(f"Error fetching data for {player_name}: {e}")
                continue

    logger.info(f"Found {len(top_players)} eligible players")
    return top_players


def fetch_player_data(
    client: NBAClient,
    player_id: int,
    player_name: str,
    seasons: List[str]
) -> pd.DataFrame:
    """
    Fetch and combine game logs for a player across multiple seasons.

    Args:
        client: NBAClient instance
        player_id: Player ID
        player_name: Player name (for logging)
        seasons: List of seasons to fetch

    Returns:
        Combined DataFrame with all seasons
    """
    all_games = []

    for season in seasons:
        try:
            gamelog = client.get_player_gamelog(player_id, season)

            if len(gamelog) > 0:
                gamelog["season"] = season
                gamelog["player_id"] = player_id
                gamelog["player_name"] = player_name
                all_games.append(gamelog)
                logger.debug(f"  {player_name}: {len(gamelog)} games in {season}")

        except Exception as e:
            logger.debug(f"  Could not fetch {season} for {player_name}: {e}")
            continue

    if not all_games:
        return pd.DataFrame()

    combined = pd.concat(all_games, ignore_index=True)
    combined = combined.sort_values("GAME_DATE").reset_index(drop=True)

    return combined


def build_dataset(
    max_players: int = 100,
    seasons: List[str] = SEASONS,
    min_games_per_season: int = 20
) -> pd.DataFrame:
    """
    Build the complete training dataset.

    Args:
        max_players: Maximum number of players to include
        seasons: List of seasons to fetch
        min_games_per_season: Minimum games played to include a player

    Returns:
        DataFrame with all player games and engineered features
    """
    client = NBAClient()

    # Step 1: Get top players from most recent season
    logger.info("=" * 70)
    logger.info("STEP 1: Identifying players to include")
    logger.info("=" * 70)

    top_players = get_top_players(client, season=seasons[-1], min_games=min_games_per_season)

    # Limit to max_players
    if len(top_players) > max_players:
        # Sort by average minutes (get the best players)
        top_players = sorted(top_players, key=lambda x: x["avg_min"], reverse=True)
        top_players = top_players[:max_players]

    logger.info(f"Building dataset for {len(top_players)} players across {len(seasons)} seasons")

    # Step 2: Fetch game logs for all players
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Fetching game logs")
    logger.info("=" * 70)

    all_player_data = []

    for player_info in tqdm(top_players, desc="Fetching player data"):
        player_games = fetch_player_data(
            client,
            player_info["player_id"],
            player_info["player_name"],
            seasons
        )

        if len(player_games) > 0:
            all_player_data.append(player_games)

    if not all_player_data:
        raise ValueError("No player data was fetched!")

    combined_data = pd.concat(all_player_data, ignore_index=True)
    logger.info(f"Total games fetched: {len(combined_data)}")

    # Step 3: Get team stats for all seasons (for opponent features)
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Fetching team stats for opponent features")
    logger.info("=" * 70)

    team_stats_by_season = {}
    for season in seasons:
        logger.info(f"Fetching team stats for {season}...")
        team_stats_by_season[season] = client.get_league_team_stats(season)

    # Step 4: Build features per player (to maintain proper rolling windows)
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Building features")
    logger.info("=" * 70)

    featured_data = []

    for player_id in tqdm(combined_data["player_id"].unique(), desc="Engineering features"):
        player_df = combined_data[combined_data["player_id"] == player_id].copy()

        # Build features for each season separately (to prevent leakage across seasons)
        for season in seasons:
            season_df = player_df[player_df["season"] == season].copy()

            if len(season_df) == 0:
                continue

            # Get team stats for this season
            team_stats = team_stats_by_season[season]

            # Apply feature engineering
            try:
                featured_season = build_features(season_df, team_stats)
                featured_data.append(featured_season)
            except Exception as e:
                logger.error(f"Error building features for player {player_id} in {season}: {e}")
                continue

    if not featured_data:
        raise ValueError("No features were generated!")

    final_df = pd.concat(featured_data, ignore_index=True)

    # Step 5: Clean up and filter
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: Cleaning and filtering dataset")
    logger.info("=" * 70)

    # Remove rows with missing targets
    logger.info(f"Rows before cleaning: {len(final_df)}")

    for target in TARGETS:
        if target in final_df.columns:
            final_df = final_df[final_df[target].notna()]

    # Remove rows with too many missing features (first few games of a player's data)
    # We need at least 10 games of history for the rolling_10 features
    feature_cols = [col for col in final_df.columns if "_last_" in col]
    final_df = final_df[final_df[feature_cols].notna().all(axis=1)]

    logger.info(f"Rows after cleaning: {len(final_df)}")
    logger.info(f"Players in final dataset: {final_df['player_id'].nunique()}")
    logger.info(f"Games per season:")
    for season in seasons:
        count = len(final_df[final_df["season"] == season])
        logger.info(f"  {season}: {count} games")

    return final_df


def save_dataset(df: pd.DataFrame, filename: str = "player_games.parquet"):
    """
    Save the dataset to parquet format.

    Args:
        df: DataFrame to save
        filename: Output filename
    """
    output_path = OUTPUT_DIR / filename
    df.to_parquet(output_path, index=False)
    logger.info(f"\nDataset saved to: {output_path}")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {len(df.columns)}")


if __name__ == "__main__":
    logger.info("Starting dataset builder...")
    logger.info(f"Seasons: {SEASONS}")
    logger.info(f"Target variables: {TARGETS}")

    try:
        # Build dataset
        dataset = build_dataset(
            max_players=100,
            seasons=SEASONS,
            min_games_per_season=20
        )

        # Save to parquet
        save_dataset(dataset)

        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("DATASET SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total rows: {len(dataset)}")
        logger.info(f"Total players: {dataset['player_id'].nunique()}")
        logger.info(f"\nFeature columns:")
        feature_cols = [col for col in dataset.columns if any(
            x in col for x in ["_last_", "opp_", "is_", "days_", "home_"]
        )]
        for col in sorted(feature_cols):
            logger.info(f"  - {col}")

        logger.info(f"\nTarget distributions:")
        for target in TARGETS:
            if target in dataset.columns:
                logger.info(f"  {target}: mean={dataset[target].mean():.2f}, "
                          f"std={dataset[target].std():.2f}")

        logger.info("\n✓ Dataset build complete!")

    except Exception as e:
        logger.error(f"Dataset build failed: {e}", exc_info=True)
        raise
