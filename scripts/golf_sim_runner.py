import io
import os
import time
import sqlite3
import datetime
import cProfile
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load
from constants import constants
from features import features
from utils.api_utils import make_api_call
from utils.file_utils import get_most_recent_file


class SimRunner:
    def __init__(self, output_directory: str):
        self.model_directory = constants['model_path']
        self.url = constants['base_url']
        self.output_directory = output_directory
        self.tour = 'pga'
        self.file_format = 'csv'
        self.key = constants['api_key']
        self.num_iterations = constants['num_iterations']
        self.database_path = constants['database_path']

    def get_players_filtered_by_dg_ids(self, dg_ids):
        conn = sqlite3.connect(self.database_path)

        # Create a string with a placeholder for each ID in dg_ids
        placeholders = ', '.join('?' for _ in dg_ids)

        # Insert the placeholders in the SQL query
        sql_query = f"""
        SELECT *
        FROM players
        WHERE dg_id IN ({placeholders})
        """

        # Execute query and fetch DataFrame
        players_df = pd.read_sql_query(sql_query, conn, params=dg_ids)

        return players_df

    def get_field_updates(self) -> list:
        print("Fetching field updates from API...")
        url = f"{self.url}/field-updates?tour={self.tour}" \
              f"&file_format={self.file_format}&key={self.key}"
        response_content = make_api_call(url)
        content = response_content.decode('utf-8')
        df_updates = pd.read_csv(io.StringIO(content))

        dg_ids = df_updates['dg_id'].tolist()

        # Get the players DataFrame from SQLite
        players_df = self.get_players_filtered_by_dg_ids(dg_ids)
        print("Field updates fetched.")

        return players_df

    def get_player_scores(self, dg_id, years=3):
        conn = sqlite3.connect(self.database_path)

        # Calculate the date from three years ago
        three_years_ago = datetime.datetime.now() - datetime.timedelta(
            days=years*365
        )
        three_years_ago_str = three_years_ago.strftime('%Y-%m-%d')

        # SQL query to fetch all the round scores for
        # the player over the last 3 years
        sql_query = """
            SELECT sg_total
            FROM rounds
            WHERE dg_id = ? AND date >= ?
            ORDER BY date DESC
        """

        player_scores_df = pd.read_sql_query(
            sql_query, conn, params=(dg_id, three_years_ago_str)
        )

        # If the player has less than 10 rounds,
        #   provide the mean sg_total of all rounds played by any player
        if len(player_scores_df) < 10:
            sql_query_all_players = """
            SELECT AVG(sg_total) as avg_sg_total
            FROM rounds
            WHERE date >= ? AND sg_total BETWEEN -10 AND 10
            """
            avg_sg_total_df = pd.read_sql_query(
                sql_query_all_players, conn, params=(three_years_ago_str,)
            )
            player_scores_df = pd.DataFrame(
                {'sg_total': [avg_sg_total_df['avg_sg_total'].values[0]] * 10}
            )

        return player_scores_df['sg_total']

    def preload_player_stats(self, dg_ids, years=3):
        # Fetch all the necessary player scores and calculate
        # the mean and standard deviation before starting simulations]
        print("Calculating player stats...")

        player_stats_data = {}
        for dg_id in dg_ids:
            player_scores = self.get_player_scores(dg_id, years)
            player_scores = pd.to_numeric(player_scores, errors='coerce')
            player_scores = player_scores.dropna()
            player_stats_data[dg_id] = {
                'mean_score': np.mean(player_scores.values),
                'std_dev': np.std(player_scores.values)
            }

        print("Player stats calculated.")
        return player_stats_data

    # New method to preload player predicted scores
    def preload_player_predicted_scores(self, model, player_data, features):
        print("Calculating player predicted scores...")

        player_predicted_scores = {}
        for _, player_row in player_data.iterrows():
            dg_id = player_row['dg_id']
            player_features = player_row[features].tolist()
            predicted_score = model.predict([player_features])[0]
            player_predicted_scores[dg_id] = predicted_score

        print("Player predicted scores calculated.")
        return player_predicted_scores

    # Updated method to use preloaded player predicted scores
    def simulate_tournament(
        self,
        model,
        player_data,
        num_rounds,
        sim_num,
        player_stats_data,  # use preloaded player stats data
        player_predicted_scores,  # use preloaded player predicted scores
        features=None
    ):
        dg_ids = player_data['dg_id'].tolist()
        tournament_results = []

        for sim, dg_id in enumerate(dg_ids, start=1):
            player_row = player_data[player_data['dg_id'] == dg_id].iloc[0]
            player_name = player_row['player_name']

            # Use the pre-calculated mean score and standard deviation
            mean_score = player_stats_data[dg_id]['mean_score']
            std_dev = player_stats_data[dg_id]['std_dev']

            # Use the pre-calculated predicted score
            predicted_score = player_predicted_scores[dg_id]

            round_scores = []
            total_score = 0
            for round_num in range(1, num_rounds + 1):
                # Generate a random score based on the mean
                #   and standard deviation
                random_score = np.random.normal(mean_score, std_dev)-mean_score
                predicted_score = int(predicted_score)
                generated_score = predicted_score-random_score
                round_scores.append(generated_score)
                total_score += generated_score

            tournament_results.append({
                'sim': sim_num,
                'dg_id': dg_id,
                'player_name': player_name,
                'total_score': total_score,
                **{f'round{i}': score for i, score in enumerate(
                    round_scores, start=1
                )}
            })

        return pd.DataFrame(tournament_results)

    def simulate_multiple_tournaments(
        self,
        model,
        player_data,
        player_stats_data,
        player_predicted_scores,  # use preloaded player predicted scores
        num_tournaments=10,
        num_rounds=4,
        features=None
    ):
        print(f"Simulating {num_tournaments} tournaments...")
        tournament_results = pd.DataFrame()

        start_time = time.time()

        for tournament in range(num_tournaments):
            try:
                results = self.simulate_tournament(
                    model=model,
                    player_data=player_data,
                    num_rounds=num_rounds,
                    sim_num=tournament,
                    player_stats_data=player_stats_data,
                    player_predicted_scores=player_predicted_scores,
                    features=features
                )

                tournament_results = pd.concat(
                    [tournament_results, results], ignore_index=True
                )
                if tournament % 1000 == 0:
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(
                        f"Simmed tournaments {tournament}-{tournament+1000}. "
                        f"Elapsed time: {elapsed_time:.2f} seconds"
                    )
                    start_time = time.time()
            except Exception as e:
                print(f"Error in simulating tournament {tournament+1}: {e}")

        print("Multiple tournaments simulated.")
        return tournament_results

    def get_most_recent_model(self):
        print("Fetching the most recent model...")
        model_filepath = get_most_recent_file(
            self.model_directory, 'random_forest*.joblib')
        model = load(model_filepath)
        print("Model fetched.")
        return model

    def run(self):
        profiler = cProfile.Profile()
        profiler.enable()

        print("Starting the simulation...")
        print("---------------------------")
        player_data = self.get_field_updates()
        print("---------------------------")

        # Preload player stats data
        dg_ids = player_data['dg_id'].tolist()
        player_stats_data = self.preload_player_stats(dg_ids)
        print("---------------------------")
        self.model = self.get_most_recent_model()
        print("---------------------------")
        player_predicted_scores = self.preload_player_predicted_scores(
            self.model, player_data, features
        )
        print("---------------------------")
        tournament_results = self.simulate_multiple_tournaments(
            model=self.model,
            player_data=player_data,
            player_stats_data=player_stats_data,
            player_predicted_scores=player_predicted_scores,
            num_tournaments=self.num_iterations,
            num_rounds=4,
            features=features
        )
        print("---------------------------")
        print("Saving tournament results and statistics...")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        tournament_results_file_name = (
            f"golf-sim-tournament-results-{timestamp}.csv"
        )
        tournament_results_file_path = Path(
            os.path.join(self.output_directory, tournament_results_file_name)
        )
        tournament_results.to_csv(tournament_results_file_path, index=False)
        print("Simulation completed.")

        profiler.disable()
        # profiler.print_stats(sort='time')


if __name__ == "__main__":
    output_directory = constants['output_directory']
    os.makedirs(output_directory, exist_ok=True)
    sim_runner = SimRunner(output_directory=output_directory)
    sim_runner.run()
