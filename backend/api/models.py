"""
Pydantic models for API request/response validation.
"""

from typing import Optional, List
from pydantic import BaseModel


class PlayerSearchResponse(BaseModel):
    """Response model for player search."""
    player_id: int
    player_name: str
    team: Optional[str] = None


class NextGameInfo(BaseModel):
    """Information about the next scheduled game."""
    date: str
    opponent: str
    opponent_abbr: str
    home_away: str
    player_team: str
    days_ahead: int


class RecentStats(BaseModel):
    """Recent performance statistics."""
    pts_avg: float
    reb_avg: float
    ast_avg: float
    min_avg: float


class OpponentStats(BaseModel):
    """Opponent team statistics."""
    def_rating: float
    pace: float
    net_rating: float


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    player_name: str
    player_id: int
    next_game: NextGameInfo
    recent_stats: RecentStats
    opponent_stats: OpponentStats
    predictions: dict  # {"PTS": 24.5, "REB": 8.2, "AST": 7.1}
    days_rest: float


class GameHistory(BaseModel):
    """Single game history entry."""
    date: str
    opponent: str
    home_away: str
    pts: int
    reb: int
    ast: int
    min: float


class PlayerHistoryResponse(BaseModel):
    """Response model for player history."""
    player_name: str
    games: List[GameHistory]


class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str
