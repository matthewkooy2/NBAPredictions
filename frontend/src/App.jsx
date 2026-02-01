import { useState } from 'react';
import axios from 'axios';
import {Search, TrendingUp, Calendar, Users} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [selectedPlayer, setSelectedPlayer] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [history, setHistory] = useState(null);
  const [loading, setLoading] = useState(false);

  // Search for players
  const handleSearch = async (query) => {
    if (query.length < 2) {
      setSearchResults([]);
      return;
    }

    try {
      const res = await axios.get(`${API_BASE}/api/players/search?query=${query}`);
      setSearchResults(res.data);
    } catch (error) {
      console.error('Search failed:', error);
    }
  };

  // Get predictions for selected player
  const handleSelectPlayer = async (player) => {
    setSelectedPlayer(player);
    setSearchResults([]);
    setSearchQuery(player.player_name);
    setLoading(true);

    try {
      const [predRes, histRes] = await Promise.all([
        axios.get(`${API_BASE}/api/players/${player.player_id}/predict`),
        axios.get(`${API_BASE}/api/players/${player.player_id}/history?limit=10`)
      ]);

      setPrediction(predRes.data);
      setHistory(histRes.data);
    } catch (error) {
      console.error('Prediction failed:', error);
      alert('Failed to get prediction. Make sure the API server is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500">
            NBA Stats Predictor
          </h1>
          <p className="text-gray-400 text-lg">
            Predict player performance using machine learning
          </p>
        </div>

        {/* Search Bar */}
        <div className="max-w-2xl mx-auto mb-12 relative">
          <div className="relative">
            <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                handleSearch(e.target.value);
              }}
              placeholder="Search for a player (e.g., LeBron James, Nikola Jokic)..."
              className="w-full pl-12 pr-4 py-4 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* Search Results Dropdown */}
          {searchResults.length > 0 && (
            <div className="absolute w-full mt-2 bg-gray-800 border border-gray-700 rounded-lg overflow-hidden shadow-xl z-10">
              {searchResults.map((player) => (
                <button
                  key={player.player_id}
                  onClick={() => handleSelectPlayer(player)}
                  className="w-full px-4 py-3 text-left hover:bg-gray-700 transition-colors flex items-center gap-3"
                >
                  <Users size={18} className="text-blue-400" />
                  <div>
                    <div className="font-semibold">{player.player_name}</div>
                    <div className="text-sm text-gray-400">{player.team}</div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Loading State */}
        {loading && (
          <div className="text-center py-12">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-4 border-gray-700 border-t-blue-500"></div>
            <p className="mt-4 text-gray-400">Loading predictions...</p>
          </div>
        )}

        {/* Prediction Results */}
        {prediction && !loading && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left Column - Next Game Info */}
            <div className="card">
              <h2 className="card-header flex items-center gap-2">
                <Calendar size={24} />
                Next Game
              </h2>
              <div className="space-y-3">
                <div>
                  <p className="text-gray-400 text-sm">Date</p>
                  <p className="text-xl font-semibold">{prediction.next_game.date}</p>
                </div>
                <div>
                  <p className="text-gray-400 text-sm">Opponent</p>
                  <p className="text-xl font-semibold">{prediction.next_game.opponent}</p>
                </div>
                <div>
                  <p className="text-gray-400 text-sm">Location</p>
                  <p className="text-xl font-semibold">
                    {prediction.next_game.home_away === 'HOME' ? '🏠 Home' : '✈️ Away'}
                  </p>
                </div>
                <div>
                  <p className="text-gray-400 text-sm">Days Rest</p>
                  <p className="text-xl font-semibold">{Math.round(prediction.days_rest)}</p>
                </div>
              </div>
            </div>

            {/* Middle Column - Predictions */}
            <div className="card">
              <h2 className="card-header flex items-center gap-2">
                <TrendingUp size={24} />
                Predicted Stats
              </h2>
              <div className="grid grid-cols-3 gap-4">
                <div className="stat-box">
                  <div className="stat-label">Points</div>
                  <div className="stat-value text-green-400">{prediction.predictions.PTS}</div>
                </div>
                <div className="stat-box">
                  <div className="stat-label">Rebounds</div>
                  <div className="stat-value text-blue-400">{prediction.predictions.REB}</div>
                </div>
                <div className="stat-box">
                  <div className="stat-label">Assists</div>
                  <div className="stat-value text-purple-400">{prediction.predictions.AST}</div>
                </div>
              </div>

              <div className="mt-6 pt-6 border-t border-gray-700">
                <h3 className="text-sm font-semibold text-gray-400 mb-3">Recent Form (L10)</h3>
                <div className="grid grid-cols-3 gap-2 text-center">
                  <div>
                    <div className="text-xs text-gray-500">PTS</div>
                    <div className="text-lg font-bold">{prediction.recent_stats.pts_avg.toFixed(1)}</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500">REB</div>
                    <div className="text-lg font-bold">{prediction.recent_stats.reb_avg.toFixed(1)}</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500">AST</div>
                    <div className="text-lg font-bold">{prediction.recent_stats.ast_avg.toFixed(1)}</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Right Column - Opponent Stats */}
            <div className="card">
              <h2 className="card-header">Opponent Context</h2>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Defensive Rating</span>
                  <span className="text-xl font-bold">{prediction.opponent_stats.def_rating.toFixed(1)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Pace</span>
                  <span className="text-xl font-bold">{prediction.opponent_stats.pace.toFixed(1)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Net Rating</span>
                  <span className={`text-xl font-bold ${prediction.opponent_stats.net_rating >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {prediction.opponent_stats.net_rating >= 0 ? '+' : ''}{prediction.opponent_stats.net_rating.toFixed(1)}
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Recent Games Chart */}
        {history && !loading && (
          <div className="card mt-6">
            <h2 className="card-header">Recent Performance</h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={history.games}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="date" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                  labelStyle={{ color: '#9CA3AF' }}
                />
                <Legend />
                <Line type="monotone" dataKey="pts" stroke="#10B981" strokeWidth={2} name="Points" />
                <Line type="monotone" dataKey="reb" stroke="#3B82F6" strokeWidth={2} name="Rebounds" />
                <Line type="monotone" dataKey="ast" stroke="#A78BFA" strokeWidth={2} name="Assists" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
