"""
Model training script for NBA player stats prediction.

This script:
1. Loads the processed dataset
2. Splits into train/test using time-based splits
3. Computes baseline predictions
4. Trains separate CatBoost models for each target (PTS, REB, AST)
5. Evaluates performance vs baseline
6. Saves trained models

Usage:
    python -m src.train
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import configuration
try:
    from config import TRAIN_SEASONS, TEST_SEASON, TARGETS
except ImportError:
    logger.warning("config.py not found, using default values")
    TRAIN_SEASONS = ["2022-23", "2023-24"]
    TEST_SEASON = "2024-25"
    TARGETS = ["PTS", "REB", "AST"]

# Paths
DATA_PATH = Path("data/processed/player_games.parquet")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    """
    Load the processed dataset.

    Returns:
        DataFrame with all player games and features
    """
    logger.info(f"Loading dataset from {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    logger.info(f"Loaded {len(df)} games from {df['player_id'].nunique()} players")
    logger.info(f"Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Get the list of feature columns to use for training.

    Excludes metadata, targets, and other non-feature columns.

    Args:
        df: DataFrame with all columns

    Returns:
        List of feature column names
    """
    # Feature columns are those with specific patterns
    feature_patterns = ["_last_", "opp_", "is_", "days_"]

    feature_cols = []
    for col in df.columns:
        if any(pattern in col for pattern in feature_patterns):
            feature_cols.append(col)

    # Also include home_away as categorical feature
    if "home_away" in df.columns:
        feature_cols.append("home_away")

    # Add player_id as categorical feature (helps model learn player-specific patterns)
    if "player_id" in df.columns:
        feature_cols.append("player_id")

    logger.info(f"Using {len(feature_cols)} features for training")
    logger.debug(f"Features: {feature_cols}")

    return feature_cols


def split_data(df: pd.DataFrame, train_seasons: list, test_season: str):
    """
    Split data into train and test sets using time-based split.

    Args:
        df: Full dataset
        train_seasons: List of seasons to use for training
        test_season: Season to use for testing

    Returns:
        Tuple of (train_df, test_df)
    """
    logger.info(f"Splitting data: train={train_seasons}, test={test_season}")

    train_df = df[df["season"].isin(train_seasons)].copy()
    test_df = df[df["season"] == test_season].copy()

    logger.info(f"Train set: {len(train_df)} games ({train_df['player_id'].nunique()} players)")
    logger.info(f"Test set: {len(test_df)} games ({test_df['player_id'].nunique()} players)")

    return train_df, test_df


def compute_baseline_predictions(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str):
    """
    Compute baseline predictions using simple last-10 average.

    This serves as a baseline to compare model performance against.

    Args:
        train_df: Training data
        test_df: Test data
        target: Target variable name (e.g., "PTS")

    Returns:
        Array of baseline predictions for test set
    """
    # For baseline, use the last_10 feature as prediction
    baseline_col = f"{target.lower()}_last_10"

    if baseline_col not in test_df.columns:
        logger.warning(f"Baseline column {baseline_col} not found, using mean")
        return np.full(len(test_df), train_df[target].mean())

    baseline_predictions = test_df[baseline_col].values

    return baseline_predictions


def train_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    target: str,
    cat_features: list = None
) -> tuple:
    """
    Train a CatBoost model for a single target.

    Args:
        train_df: Training data
        test_df: Test data
        feature_cols: List of feature column names
        target: Target variable name
        cat_features: List of categorical feature names

    Returns:
        Tuple of (model, predictions, metrics_dict)
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Training model for {target}")
    logger.info(f"{'='*70}")

    # Prepare data
    X_train = train_df[feature_cols]
    y_train = train_df[target]
    X_test = test_df[feature_cols]
    y_test = test_df[target]

    # Identify categorical features
    if cat_features is None:
        cat_features = []
        for col in feature_cols:
            if X_train[col].dtype == 'object' or col in ['player_id', 'home_away']:
                cat_features.append(col)

    logger.info(f"Categorical features: {cat_features}")

    # Create CatBoost pools
    train_pool = Pool(
        data=X_train,
        label=y_train,
        cat_features=cat_features
    )

    test_pool = Pool(
        data=X_test,
        label=y_test,
        cat_features=cat_features
    )

    # Configure model
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        loss_function='RMSE',
        eval_metric='MAE',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50
    )

    # Train
    logger.info("Training model...")
    model.fit(
        train_pool,
        eval_set=test_pool,
        use_best_model=True,
        plot=False
    )

    # Predict
    predictions = model.predict(X_test)

    # Compute metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    # Baseline metrics
    baseline_preds = compute_baseline_predictions(train_df, test_df, target)
    baseline_mae = mean_absolute_error(y_test, baseline_preds)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))

    # Improvement
    mae_improvement = ((baseline_mae - mae) / baseline_mae) * 100
    rmse_improvement = ((baseline_rmse - rmse) / baseline_rmse) * 100

    metrics = {
        'target': target,
        'mae': mae,
        'rmse': rmse,
        'baseline_mae': baseline_mae,
        'baseline_rmse': baseline_rmse,
        'mae_improvement': mae_improvement,
        'rmse_improvement': rmse_improvement,
        'best_iteration': model.get_best_iteration(),
        'feature_importance': model.get_feature_importance()
    }

    # Log results
    logger.info(f"\nResults for {target}:")
    logger.info(f"  Model MAE:     {mae:.3f}")
    logger.info(f"  Model RMSE:    {rmse:.3f}")
    logger.info(f"  Baseline MAE:  {baseline_mae:.3f}")
    logger.info(f"  Baseline RMSE: {baseline_rmse:.3f}")
    logger.info(f"  MAE Improvement:  {mae_improvement:+.2f}%")
    logger.info(f"  RMSE Improvement: {rmse_improvement:+.2f}%")
    logger.info(f"  Best iteration: {model.get_best_iteration()}")

    return model, predictions, metrics


def save_model(model: CatBoostRegressor, target: str):
    """
    Save a trained model to disk.

    Args:
        model: Trained CatBoost model
        target: Target variable name
    """
    model_path = MODELS_DIR / f"{target.lower()}_model.cbm"
    model.save_model(str(model_path))
    logger.info(f"Model saved to {model_path}")


def print_feature_importance(model: CatBoostRegressor, feature_cols: list, target: str, top_n: int = 10):
    """
    Print top N most important features.

    Args:
        model: Trained model
        feature_cols: List of feature names
        target: Target variable name
        top_n: Number of top features to display
    """
    importance = model.get_feature_importance()

    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)

    logger.info(f"\nTop {top_n} features for {target}:")
    for idx, row in importance_df.head(top_n).iterrows():
        logger.info(f"  {row['feature']:.<30} {row['importance']:.2f}")


def main():
    """Main training pipeline."""
    logger.info("=" * 70)
    logger.info("NBA STATS PREDICTOR - MODEL TRAINING")
    logger.info("=" * 70)

    # Load data
    df = load_data()

    # Get feature columns
    feature_cols = get_feature_columns(df)

    # Split data
    train_df, test_df = split_data(df, TRAIN_SEASONS, TEST_SEASON)

    # Train models for each target
    models = {}
    all_metrics = []

    for target in TARGETS:
        model, predictions, metrics = train_model(
            train_df,
            test_df,
            feature_cols,
            target
        )

        # Save model
        save_model(model, target)

        # Print feature importance
        print_feature_importance(model, feature_cols, target)

        # Store
        models[target] = model
        all_metrics.append(metrics)

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 70)

    summary_df = pd.DataFrame([
        {
            'Target': m['target'],
            'Model MAE': f"{m['mae']:.3f}",
            'Baseline MAE': f"{m['baseline_mae']:.3f}",
            'Improvement': f"{m['mae_improvement']:+.2f}%"
        }
        for m in all_metrics
    ])

    logger.info("\n" + summary_df.to_string(index=False))

    logger.info("\n✓ Training complete!")
    logger.info(f"Models saved to {MODELS_DIR}/")

    return models, all_metrics


if __name__ == "__main__":
    try:
        models, metrics = main()
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
