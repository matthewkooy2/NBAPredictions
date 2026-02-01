# NBA Stats Predictor

A full-stack machine learning application that predicts NBA player performance for upcoming games. Built with React, FastAPI, and CatBoost to demonstrate modern software engineering and machine learning practices.

**Created by Matthew Kooy**

## Overview

This application analyzes historical NBA data to predict player statistics (points, rebounds, assists) for their next scheduled game. It features a responsive web interface, RESTful API, and production-ready ML pipeline.

## Key Features

- **Machine Learning Predictions**: CatBoost models trained on 10,000+ games from 60 NBA players
- **Smart Player Search**: Real-time autocomplete with Unicode support for international players
- **Context-Aware Forecasting**: Considers opponent strength, rest days, home/away status, and recent performance
- **Interactive Visualizations**: Charts showing recent performance trends and prediction confidence
- **Production-Ready Architecture**: Caching, rate limiting, error handling, and API documentation

## Technical Highlights

### Machine Learning
- Gradient boosting models (CatBoost) with feature engineering
- 19 engineered features including rolling averages, opponent metrics, and rest calculations
- Proper train/test splitting to prevent data leakage
- Models outperform baseline (10-game average) across all metrics

### Backend (Python)
- FastAPI for high-performance REST API
- SQLite caching with 6-hour expiry for external API calls
- Exponential backoff retry logic for reliability
- Pydantic schemas for request/response validation
- Comprehensive logging and error handling

### Frontend (JavaScript)
- React 18 with modern hooks and state management
- Tailwind CSS for responsive, professional UI
- Recharts for data visualization
- Axios for API communication

### Data Engineering
- NBA API client with rate limiting and retry logic
- Automated dataset creation from 3 seasons of data
- Efficient data processing with Pandas and NumPy
- Unicode normalization for player name matching

## Tech Stack

**Frontend**: React, Vite, Tailwind CSS, Recharts
**Backend**: FastAPI, Pydantic, Uvicorn
**ML/Data**: CatBoost, Pandas, NumPy, scikit-learn, nba_api
**Infrastructure**: SQLite caching, requests-cache, tenacity (retries)

## Quick Start

### Backend
```bash
source .venv/bin/activate
uvicorn backend.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm run dev
```

Visit http://localhost:5173 to use the application.

## Project Structure

```
NBAPredictions/
├── backend/              # FastAPI REST API
├── frontend/             # React application
├── src/                  # ML pipeline and data processing
├── models/               # Trained CatBoost models
└── data/                 # Dataset and cache
```

## Model Performance

| Metric | Points | Rebounds | Assists |
|--------|--------|----------|---------|
| MAE | 5.20 | 2.07 | 1.57 |
| Baseline MAE | 5.24 | 2.10 | 1.60 |
| Improvement | +0.62% | +1.73% | +1.90% |

## API Endpoints

- `GET /api/players/search?query={name}` - Search for players
- `GET /api/players/{id}/predict` - Get predictions for next game
- `GET /api/players/{id}/history` - Get recent game history
- `GET /api/health` - Health check

Interactive API documentation available at http://localhost:8000/docs

## License

MIT
