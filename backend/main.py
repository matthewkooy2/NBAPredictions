"""
FastAPI backend for NBA Stats Predictor.

Run with: uvicorn backend.main:app --reload
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import List

from backend.api.models import (
    PlayerSearchResponse,
    PredictionResponse,
    NextGameInfo,
    RecentStats,
    OpponentStats,
    PlayerHistoryResponse,
    GameHistory,
    ErrorResponse
)

from src.nba_client import NBAClient
from src.predict import (
    load_models,
    find_next_game,
    build_prediction_features,
    make_predictions,
    get_feature_columns
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NBA Stats Predictor API",
    description="Predict NBA player stats for their next game using machine learning",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://nba-predictor-frontend.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for caching
_nba_client = None
_models = None


def get_nba_client() -> NBAClient:
    """Get or create NBA client singleton."""
    global _nba_client
    if _nba_client is None:
        _nba_client = NBAClient()
        logger.info("NBA Client initialized")
    return _nba_client


def get_models() -> dict:
    """Get or load models singleton."""
    global _models
    if _models is None:
        _models = load_models()
        logger.info("Models loaded")
    return _models


@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    logger.info("Starting NBA Stats Predictor API...")
    get_nba_client()
    get_models()
    logger.info("API ready!")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "NBA Stats Predictor API",
        "version": "1.0.0"
    }


@app.get("/api/health")
async def health_check():
    """Detailed health check."""
    try:
        client = get_nba_client()
        models = get_models()

        return {
            "status": "healthy",
            "client_ready": client is not None,
            "models_loaded": len(models) == 3,
            "models": list(models.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.get("/api/players/search", response_model=List[PlayerSearchResponse])
async def search_players(
    query: str = Query(..., min_length=2, description="Player name to search for")
):
    """
    Search for players by name.

    Returns a list of matching players with their IDs and teams.
    """
    try:
        client = get_nba_client()

        # Get all players
        all_players = client.get_all_players("2025-26")
        active_players = all_players[all_players["ROSTERSTATUS"] == 1]

        # Normalize search query
        from src.nba_client import normalize_name
        normalized_query = normalize_name(query)

        # Create normalized column
        active_players['normalized_name'] = active_players['DISPLAY_FIRST_LAST'].apply(normalize_name)

        # Search for matches
        mask = active_players['normalized_name'].str.contains(normalized_query, na=False)
        matches = active_players[mask].head(10)  # Limit to 10 results

        results = []
        for _, row in matches.iterrows():
            results.append(PlayerSearchResponse(
                player_id=int(row['PERSON_ID']),
                player_name=row['DISPLAY_FIRST_LAST'],
                team=row.get('TEAM_NAME', None)
            ))

        return results

    except Exception as e:
        logger.error(f"Player search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/api/players/{player_id}/predict", response_model=PredictionResponse)
async def predict_player_stats(player_id: int):
    """
    Predict stats for a player's next game.

    Fetches the player's recent games, finds their next scheduled game,
    and returns predictions for PTS, REB, AST.
    """
    try:
        client = get_nba_client()
        models = get_models()

        # Get player name
        all_players = client.get_all_players("2025-26")
        player_row = all_players[all_players["PERSON_ID"] == player_id]

        if len(player_row) == 0:
            raise HTTPException(status_code=404, detail="Player not found")

        player_name = player_row.iloc[0]["DISPLAY_FIRST_LAST"]

        # Find next game
        try:
            next_game = find_next_game(client, player_name)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

        # Build features
        features = build_prediction_features(client, player_id, next_game)

        # Make predictions
        predictions = make_predictions(models, features)

        # Build response
        response = PredictionResponse(
            player_name=player_name,
            player_id=player_id,
            next_game=NextGameInfo(**next_game),
            recent_stats=RecentStats(
                pts_avg=float(features['pts_last_10'].iloc[0]),
                reb_avg=float(features['reb_last_10'].iloc[0]),
                ast_avg=float(features['ast_last_10'].iloc[0]),
                min_avg=float(features['min_last_10'].iloc[0])
            ),
            opponent_stats=OpponentStats(
                def_rating=float(features['opp_def_rating'].iloc[0]),
                pace=float(features['opp_pace'].iloc[0]),
                net_rating=float(features['opp_net_rating'].iloc[0])
            ),
            predictions={
                "PTS": round(predictions["PTS"], 1),
                "REB": round(predictions["REB"], 1),
                "AST": round(predictions["AST"], 1)
            },
            days_rest=float(features['days_rest'].iloc[0])
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/api/players/{player_id}/history", response_model=PlayerHistoryResponse)
async def get_player_history(
    player_id: int,
    limit: int = Query(10, ge=1, le=50, description="Number of recent games to return")
):
    """
    Get a player's recent game history.

    Returns the last N games for visualization.
    """
    try:
        client = get_nba_client()

        # Get player info
        all_players = client.get_all_players("2025-26")
        player_row = all_players[all_players["PERSON_ID"] == player_id]

        if len(player_row) == 0:
            raise HTTPException(status_code=404, detail="Player not found")

        player_name = player_row.iloc[0]["DISPLAY_FIRST_LAST"]

        # Get game log
        gamelog = client.get_player_gamelog(player_id, "2025-26")

        # Get recent games
        recent_games = gamelog.tail(limit)

        # Build response
        games = []
        for _, game in recent_games.iterrows():
            # Determine home/away
            matchup = game['MATCHUP']
            if 'vs.' in matchup:
                home_away = "HOME"
                opponent = matchup.split('vs.')[1].strip()
            else:
                home_away = "AWAY"
                opponent = matchup.split('@')[1].strip()

            games.append(GameHistory(
                date=game['GAME_DATE'].strftime('%Y-%m-%d'),
                opponent=opponent,
                home_away=home_away,
                pts=int(game['PTS']),
                reb=int(game['REB']),
                ast=int(game['AST']),
                min=float(game['MIN'])
            ))

        return PlayerHistoryResponse(
            player_name=player_name,
            games=games
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"History fetch failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
