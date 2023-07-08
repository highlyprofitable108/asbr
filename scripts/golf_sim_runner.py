import sqlite3
import pandas as pd
import numpy as np
import os
import io
import datetime
from joblib import load
from pathlib import Path
from constants import constants
from features import features
from utils.api_utils import make_api_call
from utils.file_utils import write_to_file, get_most_recent_file
from concurrent.futures import ProcessPoolExecutor


class SimRunner:
    def __init__(self, output_directory: str):
        self.model_directory = constants['model_path']
        self.url = constants['base_url']
        self.output_directory = output_directory
        self.tour = 'pga'
        self.file_format = 'csv'
        self.key = constants['api_key']
        self.num_iterations = constants['num_iterations']

    def fetch_data_from_sqlite(self):
        print("Fetching player data from SQLite database...")
        conn = sqlite3.connect(constants['database_path'])
        player_data = pd.read_sql_query('SELECT * FROM player_performance', conn)
        conn.close()
        print("Player data fetched.")
        return player_data
    
    def get_field_updates(self) -> list:
        print("Fetching field updates from API...")
        url = f"{self.url}/field-updates?tour={self.tour}&file_format={self.file_format}&key={self.key}"
        response_content = make_api_call(url)
        content = response_content.decode('utf-8')  # Convert the bytes to a string
        df_updates = pd.read_csv(io.StringIO(content))
        dg_ids = df_updates['dg_id'].tolist()
        print("Field updates fetched.")
        return dg_ids
    

    def predict_tournament(self, model, player_dg_ids, player_data, num_rounds=4):
        print("Predicting scores for the tournament players...")
        upcoming_event_features = player_data[player_data['dg_id'].isin(player_dg_ids)]

        # Create an empty DataFrame to store the results
        results = pd.DataFrame(columns=['player_name', 'round', 'score'])

        for index, row in upcoming_event_features.iterrows():
            for round in range(num_rounds):
                # Extract the features from the player's data
                player_features = row[features]

                # Predict the score using the trained regression model
                predicted_scores = model.predict([player_features])[0]
                print(predicted_scores)
                # Create a DataFrame with the predicted scores
                scores_df = pd.DataFrame({'player_name': row['player_name'], 'round': round + 1, 'score': predicted_scores}, index=[0])

                # Concatenate the scores DataFrame with the results DataFrame
                results = pd.concat([results, scores_df], ignore_index=True)

        print("Scores predicted for the tournament players.")
        return results

    def simulate_tournament(self, model, player_data, num_rounds):
        print("Simulating a tournament...")
        # Get the player IDs in the current field
        dg_ids = player_data['dg_id'].tolist()

        # Predict scores for all players in the tournament using the model
        results = self.predict_tournament(model, dg_ids, player_data, num_rounds)

        print("Tournament simulated.")
        return results


    def simulate_multiple_tournaments(self, model, player_data, num_tournaments=1000, num_rounds=4):
        print("Simulating multiple tournaments...")
        tournament_results = pd.DataFrame()

        with ProcessPoolExecutor() as executor:
            futures = []
            for tournament in range(num_tournaments):
                future = executor.submit(self.simulate_tournament, model, player_data, num_rounds)
                futures.append(future)

            for tournament, future in enumerate(futures, start=1):
                results = future.result()

                # Calculate the total score for each player in the tournament
                total_scores = results.groupby('player_name')['score'].sum()

                # Find the winner of the tournament (the player with the lowest total score)
                winner = total_scores.idxmin()

                # Add the winner to the tournament_results DataFrame
                tournament_results = tournament_results.append({'tournament': tournament, 'winner': winner}, ignore_index=True)
                print(f"Simulated tournament {tournament}...")

        print("Multiple tournaments simulated.")
        return tournament_results
    

    def calculate_first_round_leader(self, player_data):
        print("Calculating first round leader statistics...")
        first_round_leader = player_data.groupby(['dg_id', 'player_name']).agg({'first_round_score': ['mean', 'std', 'count']}).reset_index()
        first_round_leader.columns = ['dg_id', 'player_name', 'first_round_score_average', 'first_round_deviation', 'count_of_times_dg_id_low_first_round']
        first_round_leader['american_odds_to_lead'] = first_round_leader['count_of_times_dg_id_low_first_round'].apply(lambda x: '+{:d}'.format(int((1 / x - 1) * 100)) if x > 0 else 0)
        print("First round leader statistics calculated.")
        return first_round_leader

    def get_most_recent_model(self):
        print("Fetching the most recent model...")
        model_filepath = get_most_recent_file(self.model_directory, 'rf_regression*.joblib')
        model = load(model_filepath)
        print("Model fetched.")
        return model

    def create_temp_table_and_append_player_stats(self, dg_ids, db_path):
        print("Creating player performance DataFrame...")
        conn = sqlite3.connect(db_path)
        query = f"SELECT * FROM player_performance WHERE dg_id IN ({','.join(['?']*len(dg_ids))})"
        player_data = pd.read_sql_query(query, conn, params=dg_ids)
        conn.close()
        print("Player performance DataFrame created.")
        return player_data

    def run(self):
        print("Starting the simulation...")
        print("---------------------------")
        player_data = self.fetch_data_from_sqlite()
        print("---------------------------")
        field_player_ids = self.get_field_updates()
        print("---------------------------")
        player_data = self.create_temp_table_and_append_player_stats(field_player_ids, constants['database_path'])
        print("---------------------------")
        self.model = self.get_most_recent_model()
        print("---------------------------")
        tournament_results = self.simulate_multiple_tournaments(self.model, player_data, self.num_iterations)
        print("---------------------------")
        first_round_leader = self.calculate_first_round_leader(tournament_results)
        print("---------------------------")
        print("Saving tournament results and first round leader statistics...")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        tournament_results_file_name = f"golf-sim-tournament-results-{timestamp}.csv"
        first_round_leader_file_name = f"golf-sim-first-round-leader-{timestamp}.csv"
        tournament_results_file_path = Path(os.path.join(self.output_directory, tournament_results_file_name))
        first_round_leader_file_path = Path(os.path.join(self.output_directory, first_round_leader_file_name))
        tournament_results_data = tournament_results.to_csv(index=False).encode()
        write_to_file(tournament_results_file_path, tournament_results_data)
        first_round_leader_data = first_round_leader.to_csv(index=False).encode()
        write_to_file(first_round_leader_file_path, first_round_leader_data)
        print("Simulation completed.")

if __name__ == "__main__":
    output_directory = constants['output_directory']
    os.makedirs(output_directory, exist_ok=True)
    sim_runner = SimRunner(output_directory=output_directory)
    sim_runner.run()
