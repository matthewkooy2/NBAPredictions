"""
NBA player stats prediction CLI.

This script predicts a player's stats (PTS, REB, AST) for their next scheduled game.

Usage:
    python -m src.predict --player "Nikola Jokic"
    python -m src.predict --player "LeBron James"
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from catboost import CatBoostRegressor

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
    from config import TARGETS
except ImportError:
    TARGETS = ["PTS", "REB", "AST"]

# Paths
MODELS_DIR = Path("models")


def load_models() -> dict:
    """
    Load trained models from disk.

    Returns:
        Dictionary mapping target names to loaded models
    """
    models = {}

    for target in TARGETS:
        model_path = MODELS_DIR / f"{target.lower()}_model.cbm"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = CatBoostRegressor()
        model.load_model(str(model_path))
        models[target] = model
        logger.debug(f"Loaded model for {target}")

    logger.info(f"Loaded {len(models)} models")
    return models


def find_next_game(client: NBAClient, player_name: str) -> dict:
    """
    Find a player's next scheduled game.

    Args:
        client: NBAClient instance
        player_name: Player's name

    Returns:
        Dictionary with next game info (opponent, date, home/away)
    """
    logger.info(f"Finding next game for {player_name}...")

    # Get player ID
    player_id = client.get_player_id(player_name)
    if player_id is None:
        raise ValueError(f"Player not found: {player_name}")

    # Get recent games to find player's current team
    recent_games = client.get_player_gamelog(player_id, "2025-26")

    if len(recent_games) == 0:
        raise ValueError(f"No games found for {player_name} in 2025-26 season")

    # Get player's team from most recent game
    latest_game = recent_games.iloc[-1]
    matchup = latest_game["MATCHUP"]

    # Extract player's team abbreviation (first part of matchup)
    player_team = matchup.split()[0]

    logger.info(f"Player team: {player_team}")

    # Get team ID
    team_id = client.get_team_id(player_team)
    if team_id is None:
        raise ValueError(f"Team not found: {player_team}")

    # Search for next game in upcoming days
    today = datetime.now()
    next_game = None

    for days_ahead in range(0, 14):  # Search up to 2 weeks ahead
        search_date = today + timedelta(days=days_ahead)
        date_str = search_date.strftime("%Y-%m-%d")

        try:
            scoreboard = client.get_scoreboard(date_str)

            # Look for games involving this team
            if 'HOME_TEAM_ID' in scoreboard.columns and 'VISITOR_TEAM_ID' in scoreboard.columns:
                team_games = scoreboard[
                    (scoreboard['HOME_TEAM_ID'] == team_id) |
                    (scoreboard['VISITOR_TEAM_ID'] == team_id)
                ]

                if len(team_games) > 0:
                    game = team_games.iloc[0]

                    # Determine if home or away
                    is_home = game['HOME_TEAM_ID'] == team_id

                    if is_home:
                        opponent_id = game['VISITOR_TEAM_ID']
                        home_away = "HOME"
                    else:
                        opponent_id = game['HOME_TEAM_ID']
                        home_away = "AWAY"

                    # Get opponent name
                    all_teams = client.get_all_teams()
                    opponent = next((t for t in all_teams if t['id'] == opponent_id), None)
                    opponent_name = opponent['full_name'] if opponent else "Unknown"
                    opponent_abbr = opponent['abbreviation'] if opponent else "UNK"

                    next_game = {
                        'date': date_str,
                        'opponent': opponent_name,
                        'opponent_abbr': opponent_abbr,
                        'home_away': home_away,
                        'player_team': player_team,
                        'days_ahead': days_ahead
                    }
                    break

        except Exception as e:
            logger.debug(f"No games found on {date_str}: {e}")
            continue

    if next_game is None:
        raise ValueError(f"No upcoming games found for {player_team} in the next 14 days")

    logger.info(f"Next game: {player_team} vs {next_game['opponent']} ({next_game['home_away']}) on {next_game['date']}")

    return next_game


def build_prediction_features(
    client: NBAClient,
    player_id: int,
    next_game: dict,
    season: str = "2025-26"
) -> pd.DataFrame:
    """
    Build features for the next game prediction.

    Args:
        client: NBAClient instance
        player_id: Player ID
        next_game: Dictionary with next game info
        season: Current season

    Returns:
        DataFrame with one row containing features for prediction
    """
    logger.info("Building features for prediction...")

    # Get player's recent games
    gamelog = client.get_player_gamelog(player_id, season)

    if len(gamelog) < 10:
        logger.warning(f"Only {len(gamelog)} games available. Predictions may be less accurate.")

    # Get team stats for opponent features
    team_stats = client.get_league_team_stats(season)

    # Build features from historical data
    featured_df = build_features(gamelog, team_stats)

    # Use the most recent game's features as a template
    latest_features = featured_df.iloc[-1:].copy()

    # Calculate days of rest (days between last game and next game)
    last_game_date = pd.to_datetime(gamelog.iloc[-1]["GAME_DATE"])
    next_game_date = pd.to_datetime(next_game["date"])
    days_rest = (next_game_date - last_game_date).days

    # Update features for the next game
    latest_features["home_away"] = next_game["home_away"]
    latest_features["is_home"] = 1 if next_game["home_away"] == "HOME" else 0
    latest_features["days_rest"] = days_rest
    latest_features["is_b2b"] = 1 if days_rest == 1 else 0
    latest_features["opponent_abbr"] = next_game["opponent_abbr"]

    # Get opponent stats
    opponent_stats = team_stats[team_stats["TEAM_ABBREVIATION"] == next_game["opponent_abbr"]]

    if len(opponent_stats) > 0:
        latest_features["opp_def_rating"] = opponent_stats["DEF_RATING"].iloc[0]
        latest_features["opp_pace"] = opponent_stats["PACE"].iloc[0]
        latest_features["opp_net_rating"] = opponent_stats["NET_RATING"].iloc[0]
    else:
        logger.warning(f"Opponent stats not found for {next_game['opponent_abbr']}, using medians")
        latest_features["opp_def_rating"] = team_stats["DEF_RATING"].median()
        latest_features["opp_pace"] = team_stats["PACE"].median()
        latest_features["opp_net_rating"] = team_stats["NET_RATING"].median()

    # Add player_id
    latest_features["player_id"] = player_id

    logger.info("Features built successfully")

    return latest_features


def get_feature_columns() -> list:
    """
    Get the list of feature columns used during training.

    Returns:
        List of feature column names
    """
    feature_patterns = ["_last_", "opp_", "is_", "days_"]

    feature_cols = [
        "pts_last_5", "pts_last_10", "pts_std_last_10",
        "reb_last_5", "reb_last_10", "reb_std_last_10",
        "ast_last_5", "ast_last_10", "ast_std_last_10",
        "min_last_5", "min_last_10", "min_std_last_10",
        "days_rest", "is_b2b", "is_home",
        "opp_def_rating", "opp_pace", "opp_net_rating",
        "home_away", "player_id"
    ]

    return feature_cols


def make_predictions(models: dict, features: pd.DataFrame) -> dict:
    """
    Make predictions for all targets.

    Args:
        models: Dictionary of loaded models
        features: DataFrame with feature values

    Returns:
        Dictionary mapping target names to predictions
    """
    logger.info("Making predictions...")

    feature_cols = get_feature_columns()

    # Ensure all required columns are present
    missing_cols = [col for col in feature_cols if col not in features.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    X = features[feature_cols]

    predictions = {}

    for target, model in models.items():
        pred = model.predict(X)[0]
        predictions[target] = max(0, pred)  # Ensure non-negative predictions
        logger.debug(f"Predicted {target}: {predictions[target]:.1f}")

    logger.info("Predictions complete")

    return predictions


def print_prediction_summary(player_name: str, next_game: dict, predictions: dict, features: pd.DataFrame):
    """
    Print a nice summary of the prediction.

    Args:
        player_name: Player's name
        next_game: Dictionary with next game info
        predictions: Dictionary of predictions
        features: Feature DataFrame (for showing recent stats)
    """
    print("\n" + "=" * 70)
    print(f"NBA STATS PREDICTION - {player_name.upper()}")
    print("=" * 70)

    # Next game info
    print(f"\nNext Game:")
    print(f"  {next_game['player_team']} vs {next_game['opponent']}")
    print(f"  Location: {next_game['home_away']}")
    print(f"  Date: {next_game['date']} ({next_game['days_ahead']} days away)")
    print(f"  Days Rest: {features['days_rest'].iloc[0]:.0f}")

    # Recent form
    print(f"\nRecent Form (Last 10 Games):")
    print(f"  Points:   {features['pts_last_10'].iloc[0]:.1f} avg")
    print(f"  Rebounds: {features['reb_last_10'].iloc[0]:.1f} avg")
    print(f"  Assists:  {features['ast_last_10'].iloc[0]:.1f} avg")
    print(f"  Minutes:  {features['min_last_10'].iloc[0]:.1f} avg")

    # Opponent context
    print(f"\nOpponent ({next_game['opponent']}):")
    print(f"  Defensive Rating: {features['opp_def_rating'].iloc[0]:.1f}")
    print(f"  Pace: {features['opp_pace'].iloc[0]:.1f}")
    print(f"  Net Rating: {features['opp_net_rating'].iloc[0]:.1f}")

    # Predictions
    print(f"\n{'-' * 70}")
    print(f"PREDICTED STATS")
    print(f"{'-' * 70}")
    print(f"  Points:   {predictions['PTS']:.1f}")
    print(f"  Rebounds: {predictions['REB']:.1f}")
    print(f"  Assists:  {predictions['AST']:.1f}")

    print("\n" + "=" * 70)
    print()


def main():
    """Main prediction pipeline."""
    parser = argparse.ArgumentParser(description="Predict NBA player stats for next game")
    parser.add_argument("--player", type=str, required=True, help="Player name (e.g., 'LeBron James')")
    args = parser.parse_args()

    try:
        # Initialize client
        client = NBAClient()

        # Load models
        models = load_models()

        # Find next game
        next_game = find_next_game(client, args.player)

        # Get player ID
        player_id = client.get_player_id(args.player)

        # Build features
        features = build_prediction_features(client, player_id, next_game)

        # Make predictions
        predictions = make_predictions(models, features)

        # Print results
        print_prediction_summary(args.player, next_game, predictions, features)

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        print(f"\nError: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
