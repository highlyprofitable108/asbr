import os
import datetime
import requests
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax

# Constants
model_dir = '/Users/michaelfuscoletti/Desktop/asbr/data/golf/models/'

# Get the most recent model file
model_files = [file for file in os.listdir(model_dir) if file.endswith('.joblib')]
model_files.sort(reverse=True)
recent_model_file = model_files[0] if model_files else None

if recent_model_file:
    # Load the most recent model
    model_path = os.path.join(model_dir, recent_model_file)
    model = load(model_path)
else:
    print('No model files found.')
    model = None

def get_field_updates(tour='pga', file_format='json', key='getyourown'):
    """
    Fetches the field updates data for a golf tour.

    Parameters:
    tour: The golf tour to fetch field updates for. (default: 'pga')
    file_format: The file format of the data. (default: 'json')
    key: The API key for accessing the data. (default: 'getyourown')

    Returns:
    A DataFrame containing the field updates data.
    """
    url = f"https://feeds.datagolf.com/field-updates?tour={tour}&file_format={file_format}&key={key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as err:
        print("Error fetching field updates:", err)
        return pd.DataFrame()

    data = response.json()
    field_updates = pd.json_normalize(data['field'])
    return field_updates

def get_dg_rankings(file_format='json', key='getyourown'):
    """
    Fetches the skill rankings data for golf players.

    Parameters:
    file_format: The file format of the data. (default: 'json')
    key: The API key for accessing the data. (default: 'getyourown')

    Returns:
    A DataFrame containing the skill rankings data.
    """
    url = f"https://feeds.datagolf.com/preds/skill-ratings?display=value&file_format={file_format}&key={key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as err:
        print("Error fetching skill rankings:", err)
        return pd.DataFrame()

    data = response.json()
    dg_rankings = pd.json_normalize(data['players'])
    return dg_rankings

def get_player_data_by_dg_id(dg_id, df_filtered):
    """
    Fetches player data by DG_ID from the DataFrame and adds a constant round number.

    Parameters:
    dg_id: The DG_ID of the player to fetch data for.
    df_filtered: DataFrame containing player data.

    Returns:
    Player data in the format expected by the predict_tournament function, with a constant round number.
    """
    player_data = df_filtered[df_filtered['dg_id'] == dg_id].copy()
    if not player_data.empty:
        # Add a constant round number column with a value of 5
        player_data['round_num'] = 5
        return player_data
    else:
        return None

def predict_tournament(model, player_dg_ids, upcoming_event_features):
    """
    Predicts tournament performance for players using a trained model.

    Parameters:
    model: Trained machine learning model.
    player_dg_ids: List of DG_IDs for players.
    upcoming_event_features: DataFrame containing upcoming event features.

    Returns:
    A dictionary with predicted probabilities for each player.
    """
    upcoming_event_features = upcoming_event_features[upcoming_event_features['dg_id'].isin(player_dg_ids)]
    
    features = [
        'round_num',
        'sg_putt',
        'sg_arg',
        'sg_app',
        'sg_ott'
    ]

    X = upcoming_event_features[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    simulations = simulate_tournament(model, X_scaled, num_simulations=1000)
    mean_scores = np.mean(simulations, axis=0)
    chances = softmax(-mean_scores)  # Convert scores to probabilities using the softmax function
    
    predicted_probabilities = {}
    for i, dg_id in enumerate(player_dg_ids):
        predicted_probabilities[dg_id] = {
            'mean_score': mean_scores[i],
            'std_score': np.std(simulations[:, i]),
            'chance': chances[i]
        }
    return predicted_probabilities

def simulate_tournament(model, X_scaled, num_simulations):
    """
    Simulates a golf tournament by generating scores for each player.

    Parameters:
    model: Trained machine learning model.
    X_scaled: Scaled input features for the players.
    num_simulations: Number of simulations to run.

    Returns:
    An array of shape (num_simulations, num_players) containing the simulated scores for each player.
    """
    predictions = model.predict(X_scaled)  # Predict scores using the trained model
    scores = np.random.normal(predictions, 1.0, size=(num_simulations, len(predictions)))
    return scores

def simulate_tournament_and_get_odds(tour='pga', num_simulations=100000, file_format='json',
                                    key='195c3cb68dd9f46d7feaafc4829c', save_file_path='/Users/michaelfuscoletti/Desktop/asbr/data/golf/sims'):
    """
    Simulates a golf tournament and calculates the odds of winning for each player.

    Parameters:
    tour: The golf tour to fetch field updates for. (default: 'pga')
    num_simulations: Number of simulations to run for each player.
    file_format: The file format of the data. (default: 'json')
    key: The API key for accessing the data. (default: 'getyourown')
    save_file_path: Path to save the results. (default: None)

    Returns:
    A DataFrame containing the results of the simulation.
    """
    # Fetch field updates and rankings
    print("Fetching field updates...")
    field_updates = get_field_updates(tour, file_format, key)
    print("Fetching skill rankings...")
    dg_rankings = get_dg_rankings(file_format, key)

    df_field_updates = pd.DataFrame(field_updates)
    df_dg_rankings = pd.DataFrame(dg_rankings)

    # Merge field updates and rankings on 'dg_id'
    print("Merging field updates and rankings...")
    df_merged = pd.merge(df_field_updates, df_dg_rankings, on='dg_id')

    # Filter out players not in the field list
    print("Filtering players...")
    df_filtered = df_merged[df_merged['player_name_x'].isin(df_field_updates['player_name'])]

    df_filtered['event_id'] = '26'

    # Initialize empty lists to store results
    expected_scores = []
    standard_deviations = []
    player_results = []

    # Iterate over each player and predict their tournament performance
    print("Simulating tournament...")
    for index, row in df_filtered.iterrows():
        player_dg_id = row['dg_id']
        upcoming_event_features = get_player_data_by_dg_id(player_dg_id, df_filtered)
        if upcoming_event_features is not None:
            print(f"Simulating player {player_dg_id}...")
            player_result = predict_tournament(model, [player_dg_id], upcoming_event_features)
            player_results.append(player_result)

            if isinstance(player_result, dict) and player_dg_id in player_result:
                player_scores = player_result[player_dg_id]
                if isinstance(player_scores, dict):
                    mean_score = player_scores.get('mean_score', np.nan)
                    std_score = player_scores.get('std_score', np.nan)
                else:
                    mean_score = np.nan
                    std_score = np.nan
            else:
                mean_score = np.nan
                std_score = np.nan

            expected_scores.append(mean_score)
            standard_deviations.append(std_score)
        else:
            print(f"No upcoming event features found for player {player_dg_id}")

    df_filtered['expected_score'] = expected_scores
    df_filtered['standard_deviation'] = standard_deviations

    field_mean_score = np.nanmean(df_filtered['expected_score'])
    df_filtered['cumulative_score'] = df_filtered.apply(
        lambda row: 4 * row['expected_score'] - row['sg_putt'] - row['sg_arg'] - row['sg_app'] - row['sg_ott'],
        axis=1
    )

    df_filtered['over_under_par'] = df_filtered['cumulative_score'].apply(lambda score: score - 280)

    field_mean_score = np.nanmean(df_filtered['expected_score'])
    df_filtered['winning_probability'] = (df_filtered['expected_score'] - field_mean_score + 1) / (field_mean_score + 1) * 100

    if save_file_path:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"golf-sim-{timestamp}.csv"
        save_file_path = os.path.join(save_file_path, file_name)
        df_filtered.to_csv(save_file_path, index=False)
        print(f"Data saved to {save_file_path}")

    return df_filtered

result = simulate_tournament_and_get_odds()
