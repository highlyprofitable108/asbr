import sqlite3
import pandas as pd
import numpy as np
import statsmodels.api as sm
from constants import constants
from utils.api_utils import make_api_call

# Establish a connection to the SQLite database
conn = sqlite3.connect(constants['database_path'])


def fetch_data_from_sqlite():
    # Execute a query to fetch player data from the SQLite database
    player_data = pd.read_sql_query('SELECT * FROM player_performance', conn)

    return player_data


def get_field_updates():
    # Fetches the field updates from the API and returns a list of dg_ids
    url = f"{constants['base_url']}/field-updates?tour={constants['tour']}&file_format={constant['file_format']}&key={constants['api_key']}"
    response_content = make_api_call(url)
    content = response_content.decode('utf-8')  # Convert the bytes to a string
    df_updates = pd.read_csv(io.StringIO(content))
    dg_ids = df_updates['dg_id'].tolist()
    return dg_ids


def fit_regression(player_data):
    # Define dependent variable
    Y = player_data['weighted_sg_total']

    # Define independent variables
    X = player_data[['OTT', 'APP', 'ARG', 'PUTT']]

    # Add a constant to the independent variables matrix
    X = sm.add_constant(X)

    # Fit the model
    model = sm.OLS(Y, X).fit()
    
    return model


def simulate_tournament(player_data, num_rounds=4):
    # Create an empty DataFrame to hold the results
    results = pd.DataFrame()

    # Get players in current field
    dg_id_list = get_field_updates()

    # For each player...
    for index, row in player_data.iterrows():
        # Exclude players not in dg_id_list
        if row['dg_id'] not in dg_id_list:
            continue

        # For each round...
        for round in range(num_rounds):
            # Draw a score from a Normal distribution with mean and standard deviation based on the player's past performance
            score = np.random.normal(row['adj_round_score'], np.sqrt(row['variance_round_score']))
            
            # Add the score to the results DataFrame
            results = results.append({'player_name': row['player_name'], 'round': round+1, 'score': score}, ignore_index=True)
            
    return results


def simulate_multiple_tournaments(player_data, num_tournaments=1000, num_rounds=4):
    # Create an empty DataFrame to hold the results
    tournament_results = pd.DataFrame()

    # For each tournament...
    for tournament in range(num_tournaments):
        # Simulate the tournament
        results = simulate_tournament(player_data, num_rounds)
        
        # Sum each player's scores to get their total score for the tournament
        total_scores = results.groupby('player_name')['score'].sum()
        
        # Find the winner of the tournament (the player with the lowest total score)
        winner = total_scores.idxmin()
        
        # Add the winner to the tournament_results DataFrame
        tournament_results = tournament_results.append({'tournament': tournament+1, 'winner': winner}, ignore_index=True)
        
    return tournament_results


def calculate_win_probability(tournament_results, player_name):
    # Count the number of times the player wins
    num_wins = (tournament_results['winner'] == player_name).sum()
    
    # Divide by the total number of tournaments to get the win probability
    win_probability = num_wins / len(tournament_results)
    
    return win_probability


if __name__ == "__main__":
    # Fetch player data from SQLite database
    player_data = fetch_data_from_sqlite()

    # Fit a regression model
    model = fit_regression(player_data)

    # Use the model to make predictions
    predictions = model.predict(player_data)

    # Simulate multiple tournaments
    tournament_results = simulate_multiple_tournaments(player_data)

    # Calculate the win probability for all players
    unique_players = tournament_results['winner'].unique()
    win_probabilities = {}

    for player in unique_players:
        win_probability = calculate_win_probability(tournament_results, player)
        win_probabilities[player] = win_probability

    # Print the win probabilities
    for player, win_probability in win_probabilities.items():
        print(f'{player}: {win_probability}')
