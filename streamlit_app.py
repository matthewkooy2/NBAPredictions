"""
NBA Stats Predictor - Streamlit Web App

A machine learning application that predicts NBA player statistics for upcoming games.
Created by Matthew Kooy
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from src.nba_client import NBAClient
from src.predict import (
    load_models,
    find_next_game,
    build_prediction_features,
    make_predictions,
)

# Page configuration
st.set_page_config(
    page_title="NBA Stats Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    .stButton>button {
        width: 100%;
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'client' not in st.session_state:
    st.session_state.client = NBAClient()
if 'models' not in st.session_state:
    with st.spinner("Loading ML models..."):
        st.session_state.models = load_models()

# Title
st.title("🏀 NBA Stats Predictor")
st.markdown("*Predict player performance for their next scheduled game using machine learning*")
st.markdown("---")

# Player search
col1, col2 = st.columns([3, 1])
with col1:
    search_query = st.text_input(
        "Search for a player",
        placeholder="e.g., LeBron James, Nikola Jokić, Stephen Curry",
        help="Start typing to search for NBA players"
    )

# Search and display results
if search_query and len(search_query) >= 2:
    try:
        # Get all players and filter
        all_players = st.session_state.client.get_all_players("2025-26")
        active_players = all_players[all_players["ROSTERSTATUS"] == 1]

        # Normalize search
        from src.nba_client import normalize_name
        normalized_query = normalize_name(search_query)
        active_players['normalized_name'] = active_players['DISPLAY_FIRST_LAST'].apply(normalize_name)

        # Search
        mask = active_players['normalized_name'].str.contains(normalized_query, na=False)
        matches = active_players[mask].head(10)

        if len(matches) > 0:
            # Display search results
            st.markdown("### Search Results")
            selected_player = st.selectbox(
                "Select a player",
                options=matches.index,
                format_func=lambda x: f"{matches.loc[x, 'DISPLAY_FIRST_LAST']} ({matches.loc[x, 'TEAM_NAME']})"
            )

            if st.button("Get Predictions", type="primary"):
                player_id = int(matches.loc[selected_player, 'PERSON_ID'])
                player_name = matches.loc[selected_player, 'DISPLAY_FIRST_LAST']

                with st.spinner(f"Analyzing {player_name}'s performance..."):
                    try:
                        # Find next game
                        next_game = find_next_game(st.session_state.client, player_name)

                        # Build features
                        features = build_prediction_features(
                            st.session_state.client,
                            player_id,
                            next_game
                        )

                        # Make predictions
                        predictions = make_predictions(st.session_state.models, features)

                        # Display results
                        st.markdown("---")
                        st.markdown(f"## Predictions for {player_name}")

                        # Next game info
                        st.markdown(f"### Next Game: {next_game['player_team']} vs {next_game['opponent']}")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Date", next_game['date'])
                        with col2:
                            st.metric("Location", next_game['home_away'])
                        with col3:
                            days_rest = int(features['days_rest'].iloc[0])
                            st.metric("Days Rest", days_rest)

                        st.markdown("---")

                        # Predictions
                        st.markdown("### Predicted Stats")
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            pts_pred = round(predictions["PTS"], 1)
                            pts_avg = round(features['pts_last_10'].iloc[0], 1)
                            st.metric(
                                "Points",
                                f"{pts_pred}",
                                f"{pts_pred - pts_avg:+.1f} vs 10-game avg",
                                delta_color="normal"
                            )

                        with col2:
                            reb_pred = round(predictions["REB"], 1)
                            reb_avg = round(features['reb_last_10'].iloc[0], 1)
                            st.metric(
                                "Rebounds",
                                f"{reb_pred}",
                                f"{reb_pred - reb_avg:+.1f} vs 10-game avg",
                                delta_color="normal"
                            )

                        with col3:
                            ast_pred = round(predictions["AST"], 1)
                            ast_avg = round(features['ast_last_10'].iloc[0], 1)
                            st.metric(
                                "Assists",
                                f"{ast_pred}",
                                f"{ast_pred - ast_avg:+.1f} vs 10-game avg",
                                delta_color="normal"
                            )

                        st.markdown("---")

                        # Recent performance
                        st.markdown("### Recent Performance (Last 10 Games)")

                        # Get game history
                        gamelog = st.session_state.client.get_player_gamelog(player_id, "2025-26")
                        recent_games = gamelog.tail(10)

                        # Create chart
                        fig = go.Figure()

                        # X-axis shows games ago (10, 9, 8, ..., 1) so most recent is on the right
                        x_values = list(range(len(recent_games), 0, -1))

                        fig.add_trace(go.Scatter(
                            x=x_values,
                            y=recent_games['PTS'],
                            mode='lines+markers',
                            name='Points',
                            line=dict(color='#22c55e', width=2),
                            marker=dict(size=8)
                        ))

                        fig.add_trace(go.Scatter(
                            x=x_values,
                            y=recent_games['REB'],
                            mode='lines+markers',
                            name='Rebounds',
                            line=dict(color='#3b82f6', width=2),
                            marker=dict(size=8)
                        ))

                        fig.add_trace(go.Scatter(
                            x=x_values,
                            y=recent_games['AST'],
                            mode='lines+markers',
                            name='Assists',
                            line=dict(color='#a855f7', width=2),
                            marker=dict(size=8)
                        ))

                        fig.update_layout(
                            title="Recent Game Statistics",
                            xaxis_title="Games Ago",
                            yaxis_title="Value",
                            hovermode='x unified',
                            template='plotly_dark',
                            height=400
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Opponent context
                        st.markdown("---")
                        st.markdown("### Opponent Context")
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            def_rating = round(features['opp_def_rating'].iloc[0], 1)
                            st.metric("Opponent Defensive Rating", def_rating)

                        with col2:
                            pace = round(features['opp_pace'].iloc[0], 1)
                            st.metric("Opponent Pace", pace)

                        with col3:
                            net_rating = round(features['opp_net_rating'].iloc[0], 1)
                            st.metric("Opponent Net Rating", net_rating)

                    except Exception as e:
                        st.error(f"Error generating predictions: {str(e)}")
                        st.info("Make sure the player has games in the current season and an upcoming game scheduled.")
        else:
            st.warning("No players found matching your search.")

    except Exception as e:
        st.error(f"Search error: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: rgba(255,255,255,0.5); padding: 2rem;'>
    <p>Created by Matthew Kooy | Powered by CatBoost ML Models</p>
    <p style='font-size: 0.8rem;'>Uses historical NBA data. Not for gambling purposes.</p>
    </div>
    """,
    unsafe_allow_html=True
)
