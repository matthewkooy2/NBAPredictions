"""
NBA API client with caching and retry logic.

This module provides a reliable interface to the NBA API with:
- SQLite-based response caching
- Exponential backoff for rate limits
- Comprehensive logging
- Error handling
"""

import logging
import time
import unicodedata
from pathlib import Path
from typing import Optional

import pandas as pd
import requests_cache
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from nba_api.stats.endpoints import (
    CommonAllPlayers,
    PlayerGameLog,
    TeamGameLog,
    ScoreboardV2,
    LeagueDashTeamStats,
)
from nba_api.stats.static import teams
from requests.exceptions import RequestException, Timeout

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set up cache directory
# Use /tmp for cloud deployments (writable), fall back to local data/cache
import os
if os.path.exists("/tmp"):
    # Cloud environment (Streamlit Cloud, etc.) - use /tmp
    CACHE_DIR = Path("/tmp/nba_cache")
else:
    # Local environment - use data/cache
    CACHE_DIR = Path("data/cache")

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Install cache for requests (used by nba_api under the hood)
requests_cache.install_cache(
    cache_name=str(CACHE_DIR / "nba_api_cache"),
    backend="sqlite",
    expire_after=3600 * 6,  # Cache expires after 6 hours
    allowable_methods=["GET", "POST"],
)

# Constants
REQUEST_DELAY = 0.6  # Delay between API calls in seconds
TIMEOUT = 30  # Request timeout in seconds


def normalize_name(name: str) -> str:
    """
    Normalize a name by removing accents and converting to lowercase.

    This handles cases like 'Jokić' -> 'jokic' for better matching.

    Args:
        name: The name to normalize

    Returns:
        Normalized name string
    """
    # Normalize to NFD (decompose characters + accents)
    nfd = unicodedata.normalize('NFD', name)
    # Remove accent marks (category 'Mn' = Mark, nonspacing)
    without_accents = ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')
    return without_accents.lower()


class NBAClient:
    """
    A robust NBA API client with caching and automatic retries.
    
    Example usage:
        client = NBAClient()
        player_id = client.get_player_id("LeBron James")
        games = client.get_player_gamelog(player_id, "2025-26")
    """
    
    def __init__(self, delay: float = REQUEST_DELAY):
        """
        Initialize the NBA client.
        
        Args:
            delay: Seconds to wait between API calls (default 0.6)
        """
        self.delay = delay
        self._last_request_time = 0
        self._player_cache: Optional[pd.DataFrame] = None
        
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay:
            sleep_time = self.delay - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self._last_request_time = time.time()
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((RequestException, Timeout, ConnectionError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _make_request(self, endpoint_class, **kwargs):
        """
        Make an API request with retry logic.
        
        Args:
            endpoint_class: The nba_api endpoint class to use
            **kwargs: Arguments to pass to the endpoint
            
        Returns:
            The endpoint instance with data loaded
        """
        self._rate_limit()
        logger.debug(f"Requesting {endpoint_class.__name__} with {kwargs}")
        
        try:
            endpoint = endpoint_class(timeout=TIMEOUT, **kwargs)
            return endpoint
        except Exception as e:
            logger.error(f"Request failed: {endpoint_class.__name__} - {e}")
            raise
    
    def get_all_players(self, season: str = "2025-26") -> pd.DataFrame:
        """
        Get all players for a given season.
        
        Args:
            season: NBA season string (e.g., "2025-26")
            
        Returns:
            DataFrame with player info including PERSON_ID, DISPLAY_FIRST_LAST
        """
        if self._player_cache is not None:
            return self._player_cache
            
        logger.info(f"Fetching all players for season {season}")
        endpoint = self._make_request(
            CommonAllPlayers,
            is_only_current_season=0,
            league_id="00",
            season=season,
        )
        
        df = endpoint.get_data_frames()[0]
        self._player_cache = df
        logger.info(f"Retrieved {len(df)} players")
        return df
    
    def get_player_id(self, player_name: str, season: str = "2025-26") -> Optional[int]:
        """
        Look up a player's ID by name.

        Args:
            player_name: Player's name (e.g., "LeBron James" or "Nikola Jokic")
            season: Season for player lookup

        Returns:
            Player ID as integer, or None if not found
        """
        players_df = self.get_all_players(season)

        # Normalize the search name (removes accents, lowercases)
        normalized_search = normalize_name(player_name)

        # Create normalized column for comparison
        players_df['normalized_name'] = players_df['DISPLAY_FIRST_LAST'].apply(normalize_name)

        # Try exact match first (normalized, case-insensitive)
        mask = players_df['normalized_name'] == normalized_search

        if mask.any():
            player_id = players_df.loc[mask, "PERSON_ID"].iloc[0]
            actual_name = players_df.loc[mask, "DISPLAY_FIRST_LAST"].iloc[0]
            logger.info(f"Found player: '{player_name}' -> {actual_name} (ID {player_id})")
            return int(player_id)

        # Try partial match (normalized)
        mask = players_df['normalized_name'].str.contains(normalized_search, na=False)

        if mask.any():
            matches = players_df.loc[mask, ["PERSON_ID", "DISPLAY_FIRST_LAST"]]
            if len(matches) == 1:
                player_id = matches["PERSON_ID"].iloc[0]
                matched_name = matches["DISPLAY_FIRST_LAST"].iloc[0]
                logger.info(f"Partial match: '{player_name}' -> {matched_name} (ID {player_id})")
                return int(player_id)
            else:
                logger.warning(f"Multiple matches for '{player_name}': {matches['DISPLAY_FIRST_LAST'].tolist()}")
                # Return first match
                return int(matches["PERSON_ID"].iloc[0])

        logger.error(f"Player not found: {player_name}")
        return None
    
    def get_player_gamelog(
        self, 
        player_id: int, 
        season: str = "2025-26",
        season_type: str = "Regular Season"
    ) -> pd.DataFrame:
        """
        Get a player's game log for a season.
        
        Args:
            player_id: NBA player ID
            season: NBA season string (e.g., "2025-26")
            season_type: "Regular Season", "Playoffs", or "All Star"
            
        Returns:
            DataFrame with game-by-game stats
        """
        logger.info(f"Fetching gamelog for player {player_id}, season {season}")
        
        endpoint = self._make_request(
            PlayerGameLog,
            player_id=player_id,
            season=season,
            season_type_all_star=season_type,
        )
        
        df = endpoint.get_data_frames()[0]
        
        # Convert GAME_DATE to datetime
        if "GAME_DATE" in df.columns:
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
            df = df.sort_values("GAME_DATE").reset_index(drop=True)
        
        logger.info(f"Retrieved {len(df)} games for player {player_id}")
        return df
    
    def get_team_gamelog(
        self,
        team_id: int,
        season: str = "2025-26",
        season_type: str = "Regular Season"
    ) -> pd.DataFrame:
        """
        Get a team's game log for a season.
        
        Args:
            team_id: NBA team ID
            season: NBA season string (e.g., "2025-26")
            season_type: "Regular Season" or "Playoffs"
            
        Returns:
            DataFrame with game-by-game team stats
        """
        logger.info(f"Fetching gamelog for team {team_id}, season {season}")
        
        endpoint = self._make_request(
            TeamGameLog,
            team_id=team_id,
            season=season,
            season_type_all_star=season_type,
        )
        
        df = endpoint.get_data_frames()[0]
        
        if "GAME_DATE" in df.columns:
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
            df = df.sort_values("GAME_DATE").reset_index(drop=True)
        
        logger.info(f"Retrieved {len(df)} games for team {team_id}")
        return df
    
    def get_scoreboard(self, game_date: str) -> pd.DataFrame:
        """
        Get the scoreboard/schedule for a specific date.
        
        Args:
            game_date: Date string in format "YYYY-MM-DD"
            
        Returns:
            DataFrame with games scheduled for that date
        """
        logger.info(f"Fetching scoreboard for {game_date}")
        
        # Convert to format expected by API (MM/DD/YYYY)
        date_obj = pd.to_datetime(game_date)
        formatted_date = date_obj.strftime("%m/%d/%Y")
        
        endpoint = self._make_request(
            ScoreboardV2,
            game_date=formatted_date,
            league_id="00",
            day_offset=0,
        )
        
        # GameHeader contains the main game info
        df = endpoint.get_data_frames()[0]  # GameHeader
        logger.info(f"Found {len(df)} games on {game_date}")
        return df
    
    def get_league_team_stats(
        self,
        season: str = "2025-26",
        season_type: str = "Regular Season"
    ) -> pd.DataFrame:
        """
        Get league-wide team statistics (for opponent context features).

        Args:
            season: NBA season string
            season_type: "Regular Season" or "Playoffs"

        Returns:
            DataFrame with team stats including DEF_RATING, PACE, NET_RATING, etc.
            Also includes TEAM_ABBREVIATION for merging with game logs.
        """
        logger.info(f"Fetching league team stats for {season}")

        # Fetch advanced stats (includes DEF_RATING, OFF_RATING, NET_RATING, PACE)
        endpoint = self._make_request(
            LeagueDashTeamStats,
            season=season,
            season_type_all_star=season_type,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
        )

        df = endpoint.get_data_frames()[0]

        # Add team abbreviations by mapping TEAM_ID to abbreviation
        all_teams = teams.get_teams()
        team_abbr_map = {team["id"]: team["abbreviation"] for team in all_teams}
        df["TEAM_ABBREVIATION"] = df["TEAM_ID"].map(team_abbr_map)

        logger.info(f"Retrieved stats for {len(df)} teams")
        return df
    
    @staticmethod
    def get_team_id(team_name: str) -> Optional[int]:
        """
        Get team ID from team name or abbreviation.
        
        Args:
            team_name: Team name (e.g., "Lakers") or abbreviation (e.g., "LAL")
            
        Returns:
            Team ID or None if not found
        """
        all_teams = teams.get_teams()
        name_lower = team_name.lower()
        
        for team in all_teams:
            if (name_lower == team["abbreviation"].lower() or
                name_lower in team["full_name"].lower() or
                name_lower == team["nickname"].lower()):
                logger.info(f"Found team: {team['full_name']} -> ID {team['id']}")
                return team["id"]
        
        logger.error(f"Team not found: {team_name}")
        return None
    
    @staticmethod
    def get_all_teams() -> list[dict]:
        """
        Get list of all NBA teams.
        
        Returns:
            List of team dictionaries with id, name, abbreviation, etc.
        """
        return teams.get_teams()
    
    def clear_cache(self):
        """Clear the request cache."""
        requests_cache.clear()
        logger.info("Cache cleared")


# Convenience function for quick usage
def get_client() -> NBAClient:
    """Get a configured NBAClient instance."""
    return NBAClient()


# Test the client when run directly
if __name__ == "__main__":
    print("Testing NBA Client...")
    print("-" * 50)
    
    client = NBAClient()
    
    # Test 1: Get player ID
    print("\n1. Testing player lookup:")
    player_id = client.get_player_id("Nikola Jokic")
    print(f"   Nikola Jokic ID: {player_id}")
    
    # Test 2: Get player game log
    print("\n2. Testing player gamelog:")
    if player_id:
        games = client.get_player_gamelog(player_id, "2025-26")
        print(f"   Games retrieved: {len(games)}")
        if len(games) > 0:
            print(f"   Latest game: {games.iloc[-1]['GAME_DATE']} - {games.iloc[-1]['PTS']} PTS")
    
    # Test 3: Get team ID
    print("\n3. Testing team lookup:")
    team_id = client.get_team_id("Nuggets")
    print(f"   Nuggets ID: {team_id}")
    
    # Test 4: Get league team stats
    print("\n4. Testing league team stats:")
    team_stats = client.get_league_team_stats("2025-26")
    print(f"   Teams retrieved: {len(team_stats)}")
    print(f"   Columns: {team_stats.columns.tolist()[:10]}...")
    
    # Test 5: Get scoreboard
    print("\n5. Testing scoreboard:")
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    scoreboard = client.get_scoreboard(today)
    print(f"   Games today ({today}): {len(scoreboard)}")
    
    print("\n" + "-" * 50)
    print("All tests completed!")