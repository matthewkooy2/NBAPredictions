# NBA Stats Predictor 🏀

A full-stack machine learning web application that predicts NBA player statistics for their next scheduled game using React + FastAPI + CatBoost.

![Tech Stack](https://img.shields.io/badge/React-18-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.128-green) ![Python](https://img.shields.io/badge/Python-3.9+-yellow)

## ✨ Features

- 🔍 **Smart Player Search** - Autocomplete search with Unicode support (handles Jokić, Čančar, etc.)
- 📊 **ML-Powered Predictions** - CatBoost models predict Points, Rebounds, and Assists
- 📈 **Interactive Charts** - Visualize recent performance trends with Recharts
- 🎯 **Context-Aware** - Considers opponent defensive rating, pace, rest days, home/away
- 🎨 **Modern UI** - Beautiful, responsive design with Tailwind CSS
- ⚡ **Fast & Cached** - SQLite caching with 6-hour expiry for NBA API

## 🚀 Quick Start

### Backend
```bash
# Activate virtual environment
source .venv/bin/activate

# Start FastAPI server
uvicorn backend.main:app --reload --port 8000
```

### Frontend
```bash
# From frontend directory
npm run dev
```

Visit **http://localhost:5173** to use the app!

## 📁 Project Structure

```
NBAPredictions/
├── backend/              # FastAPI REST API
│   ├── main.py          # API endpoints
│   └── api/models.py    # Pydantic schemas
├── frontend/            # React + Vite app
│   └── src/App.jsx      # Main UI component
├── src/                 # Python ML pipeline
│   ├── nba_client.py    # NBA API wrapper
│   ├── features.py      # Feature engineering
│   ├── data_builder.py  # Dataset creation
│   ├── train.py         # Model training
│   └── predict.py       # CLI predictions
├── models/              # Trained models (CatBoost)
└── data/processed/      # Training dataset
```

## 🛠️ Tech Stack

**Frontend:**
- React 18 + Vite
- Tailwind CSS
- Recharts (data viz)
- Axios

**Backend:**
- FastAPI
- Pydantic
- Uvicorn

**ML Pipeline:**
- CatBoost (gradient boosting)
- Pandas & NumPy
- scikit-learn
- nba_api

## 📊 Model Performance

| Metric | Points | Rebounds | Assists |
|--------|--------|----------|---------|
| **Model MAE** | 5.20 | 2.07 | 1.57 |
| **Baseline MAE** | 5.24 | 2.10 | 1.60 |
| **Improvement** | +0.62% | +1.73% | +1.90% |

*All models beat the last-10-game average baseline.*

## 🗂️ Dataset

- **60 players** across 3 seasons (2022-25)
- **10,063 games** total
- **19 engineered features:**
  - Rolling stats (5/10 game windows)
  - Rest days & back-to-back flags
  - Opponent metrics (def rating, pace, net rating)
  - Home/away indicators

## 🔌 API Endpoints

```
GET  /api/players/search?query=lebron
GET  /api/players/{id}/predict
GET  /api/players/{id}/history?limit=10
GET  /api/health
```

Visit **http://localhost:8000/docs** for interactive API documentation.

## 💻 Installation

### Prerequisites
- Python 3.9+
- Node.js 16+

### Setup

1. **Clone and install Python deps:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. **Build dataset & train models:**
```bash
python -m src.data_builder  # ~5-10 min
python -m src.train
```

3. **Install frontend deps:**
```bash
cd frontend
npm install
```

## 🎯 Usage

### Web App (Recommended)
1. Start backend: `uvicorn backend.main:app --reload`
2. Start frontend: `cd frontend && npm run dev`
3. Open http://localhost:5173

### CLI (Alternative)
```bash
python -m src.predict --player "Nikola Jokic"
```

## 🔑 Key Features

**Leakage Prevention:**
- Rolling features shifted by 1 game
- Time-based train/test splits only
- No look-ahead bias

**Robust API Client:**
- SQLite caching (6hr expiry)
- Exponential backoff retries
- Rate limiting (0.6s delay)
- Unicode name normalization

**Production Ready:**
- FastAPI with auto docs
- CORS enabled
- Pydantic validation
- Error handling

## 🎨 UI Highlights

- Gradient backgrounds
- Real-time search
- Loading states
- Responsive grid layouts
- Color-coded stats (green/blue/purple)
- Interactive line charts

## 📝 Future Enhancements

- [ ] Injury status integration
- [ ] Player comparison tool
- [ ] Mobile app (React Native)
- [ ] Real-time updates
- [ ] Historical accuracy tracking

## 📄 License

MIT - Feel free to use for your own projects!

## 👨‍💻 Author

Built as a full-stack ML portfolio project demonstrating:
- Machine Learning (CatBoost, feature engineering, model evaluation)
- Backend Development (FastAPI, REST APIs, caching)
- Frontend Development (React, Tailwind, data visualization)
- Data Engineering (NBA API, dataset creation, processing)
- MLOps (training pipelines, model serving)

---

**Note:** Uses historical NBA data for predictions. Not for gambling purposes.
