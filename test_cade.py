"""Quick test to see what data we're getting for Cade Cunningham"""

from src.nba_client import NBAClient

client = NBAClient()

# Get Cade's ID
player_id = client.get_player_id("Cade Cunningham", "2025-26")
print(f"Cade Cunningham ID: {player_id}")

# Get his game log
gamelog = client.get_player_gamelog(player_id, "2025-26")

print(f"\nTotal games found: {len(gamelog)}")
print(f"\nLast 10 games:")
print(gamelog[['GAME_DATE', 'MATCHUP', 'PTS', 'REB', 'AST', 'MIN']].tail(10))

# Check the date range
if len(gamelog) > 0:
    first_game = gamelog['GAME_DATE'].min()
    last_game = gamelog['GAME_DATE'].max()
    print(f"\nDate range: {first_game} to {last_game}")
