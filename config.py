"""Project configuration and constants."""

# Seasons to use for training data
SEASONS = [
    "2022-23",
    "2023-24",
    "2024-25",
]

# Train/test split
TRAIN_SEASONS = ["2022-23", "2023-24"]
TEST_SEASON = "2024-25"

# Feature settings
ROLLING_WINDOWS = [5, 10]

# Target columns
TARGETS = ["PTS", "REB", "AST"]

# Paths
DATA_DIR = "data"
CACHE_DIR = "data/cache"
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
