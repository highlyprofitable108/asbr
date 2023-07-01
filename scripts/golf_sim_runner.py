import sqlite3
import os
import io
import datetime
import pandas as pd
import numpy as np
from joblib import load
from pathlib import Path
import random
import warnings
from constants import constants
from features import features
from utils.api_utils import make_api_call
from utils.file_utils import write_to_file, get_most_recent_file


class SimRunner:
    def __init__(self, output_directory: str):
        self.model_directory = constants['model_path']
        self.url = constants['base_url']
        self.output_directory = output_directory  # Use the argument passed during initialization
        self.tour = 'pga'
        self.file_format = 'csv'
        self.key = constants['api_key']
        self.num_iterations = constants['num_iterations']  # Updated variable name


    def get_field_updates(self) -> list:
        # Fetches the field updates from the API and returns a list of dg_ids
        url = f"{self.url}/field-updates?tour={self.tour}&file_format={self.file_format}&key={self.key}"
        response_content = make_api_call(url)
        content = response_content.decode('utf-8')  # Convert the bytes to a string
        df_updates = pd.read_csv(io.StringIO(content))
        dg_ids = df_updates['dg_id'].tolist()
        return dg_ids


    def get_most_recent_model(self):
        # Loads the most recent random forest regression model
        model_filepath = get_most_recent_file(self.model_directory, 'rf_regression*.joblib')
        model = load(model_filepath)
        return model


    def create_temp_table_and_append_course_stats(self, dg_ids, course_name, db_path):
        # Creates a temporary table and appends course statistics for player data
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Step 1: Create a temporary table to store player data
        print("Creating temporary table...")
        cursor.execute("CREATE TEMPORARY TABLE temp_player_data AS "
                       "SELECT * FROM players WHERE dg_id IN ({})".format(','.join(['?'] * len(dg_ids))), dg_ids)
        print("Temporary table created.")

        # Step 2: Get the course stats
        print("Fetching course stats...")
        cursor.execute("SELECT * FROM courses WHERE course_name = ?", (course_name,))  # Use parameterized query
        course_stats = dict(zip([column[0] for column in cursor.description], cursor.fetchone()))
        print("Course stats fetched.")

        # Step 3: Add the course_num and course_par columns to the temporary player table
        print("Adding columns to the temporary table...")
        cursor.execute("ALTER TABLE temp_player_data ADD COLUMN course_num INTEGER")
        cursor.execute("ALTER TABLE temp_player_data ADD COLUMN course_par INTEGER")
        cursor.execute("UPDATE temp_player_data SET course_num = ?", (course_stats['course_num'],))
        cursor.execute("UPDATE temp_player_data SET course_par = ?", (course_stats['course_par'],))
        print("Columns added to the temporary table.")

        # Step 4: Add the course stats columns to the temporary player table
        print("Adding course stats columns...")
        for column, value in course_stats.items():
            if column not in ['course_num', 'course_par']:
                cursor.execute("ALTER TABLE temp_player_data ADD COLUMN c_{} REAL".format(column))
                cursor.execute("UPDATE temp_player_data SET c_{} = ?".format(column), (value,))
        print("Course stats columns added.")

        # Step 5: Fetch the updated player data with course stats
        print("Fetching updated player data with course stats...")
        query = "SELECT * FROM temp_player_data"
        player_data = pd.read_sql_query(query, conn)

        # Rename the columns to match the model's column names
        column_mapping = {
            'sg_putt': 'r_sg_putt',
            'sg_arg': 'r_sg_arg',
            'sg_app': 'r_sg_app',
            'sg_ott': 'r_sg_ott',
            'sg_t2g': 'r_sg_t2g',
            'driving_dist': 'r_driving_dist',
            'driving_acc': 'r_driving_acc',
            'gir': 'r_gir',
            'scrambling': 'r_scrambling',
            'prox_rgh': 'r_prox_rgh',
            'prox_fw': 'r_prox_fw'
        }

        player_data.rename(columns=column_mapping, inplace=True)
        print("Updated player data fetched.")

        # Step 6: Clean up and close the connection
        cursor.close()
        conn.close()

        return player_data


    def simulate_tournament_and_get_odds(self, player_data, model):
        # Simulates the tournament and calculates the odds for each player
        player_dg_ids = player_data['dg_id'].tolist()
        # Initialize dictionaries to store the simulation results
        player_scores = {dg_id: [] for dg_id in player_dg_ids}

        # Create a temporary DataFrame to store all simulation results
        temp_results = pd.DataFrame()

        # Initialize dictionary to store the player with the lowest score in each simulation
        lowest_scores = {}

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)  # Ignore the specific warning
            for sim in range(self.num_iterations):
                print(f"Running simulation {sim + 1}/{self.num_iterations}...")
                # Fetch the predictions and odds for each simulation
                player_results = self.predict_tournament(model, player_dg_ids, player_data)
                lowest_score = float('inf')
                lowest_dg_id = None
                for dg_id in player_dg_ids:
                    if dg_id in player_results:
                        score = player_results[dg_id]['expected_scores']

                        # Calculate the variance
                        variability = random.uniform(0.9, 1.2)
                        score = score * variability
                        # Append the score to the respective player's list
                        player_scores[dg_id].append({'sim_num': sim, 'score': score})

                        # Create a DataFrame with simulation results for the player
                        player_sim_results = pd.DataFrame({'dg_id': [dg_id],
                                                            'player_name': player_data[player_data['dg_id'] == dg_id]['player_name'].values[0],
                                                            'sim_num': [sim],
                                                            'expected_score': [score]})  # Convert score to a single-item list

                        # Concatenate player's simulation results to the temporary DataFrame
                        temp_results = pd.concat([temp_results, player_sim_results], ignore_index=True)

                        # Check if this player's score is the lowest in this simulation
                        if score < lowest_score:
                            lowest_score = score
                            lowest_dg_id = dg_id

                # Update the player with the lowest score in this simulation
                if lowest_dg_id is not None:
                    if lowest_dg_id not in lowest_scores:
                        lowest_scores[lowest_dg_id] = 0
                    lowest_scores[lowest_dg_id] += 1

            # Consolidate the results by calculating mean and standard deviation
            consolidated_results = temp_results.groupby(['dg_id', 'player_name']).agg({'expected_score': 'mean'}).reset_index()
            consolidated_results.rename(columns={'expected_score': 'mean_expected_score'}, inplace=True)

            # Calculate lowest round count and odds to lead
            consolidated_results['lowest_round_count'] = consolidated_results['dg_id'].map(lowest_scores.get).fillna(0)
            consolidated_results['odds_to_lead'] = consolidated_results['lowest_round_count'] / self.num_iterations
            consolidated_results['odds_to_lead'] = consolidated_results['odds_to_lead'].apply(
                lambda x: '+{:d}'.format(int((1 / x - 1) * 100)) if x > 0 else 0)

            # Calculate the standard deviation for each dg_id
            for dg_id in player_dg_ids:
                scores = [score['score'] for score in player_scores[dg_id]]
                consolidated_results.loc[consolidated_results['dg_id'] == dg_id, 'standard_deviation'] = np.std(scores)

            return consolidated_results


    def calculate_odds(self, lowest_round_count, total_rounds):
        # Calculates the odds based on the lowest round count and total rounds
        if lowest_round_count == 0:
            return 0

        odds = total_rounds / lowest_round_count

        if odds > 1:
            return '+{:d}'.format(int((odds - 1) * 100))
        else:
            return '-{:d}'.format(int((1 / odds - 1) * 100))


    def predict_tournament(self, model, player_dg_ids, player_data):
        # Predicts scores for all players in the tournament using the model
        upcoming_event_features = player_data[player_data['dg_id'].isin(player_dg_ids)]

        # Features for the model
        X = upcoming_event_features[features]

        # Use the model to predict scores for all players at once
        scores = model.predict(X)

        # Create a dictionary to store the predicted scores for each player
        player_results = {dg_id: {'expected_scores': []} for dg_id in player_dg_ids}

        for j, dg_id in enumerate(player_dg_ids):
            player_results[dg_id]['expected_scores'] = scores[j]

        return player_results


    def run(self):
        # Runs the simulation
        print("Fetching the most recent model...")
        reg_model = self.get_most_recent_model()
        print("Model fetched.")

        print("Fetching field updates...")
        field_player_ids = self.get_field_updates()
        print("Field updates fetched.")
        course_name = 'Detroit Golf Club'

        print("Creating temporary table and appending course stats...")
        player_data = self.create_temp_table_and_append_course_stats(field_player_ids, course_name, constants['database_path'])
        print("Temporary table created and course stats appended.")

        print(f"Running simulation with {self.num_iterations} iterations...")
        odds = self.simulate_tournament_and_get_odds(player_data, reg_model)
        print("Simulation completed.")

        # Define file path for the consolidated results
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"golf-sim-consolidated-{timestamp}.csv"
        file_path = Path(os.path.join(self.output_directory, file_name))  # convert the path string to a Path object

        # Save the consolidated results to a single CSV file
        print(f"Saving consolidated results to {file_path}...")
        data = odds.to_csv(index=False).encode()  # Convert DataFrame to bytes
        write_to_file(file_path, data)  # Use common_utils.write_to_file()
        print("Consolidated results saved.")


if __name__ == "__main__":
    output_directory = constants['output_directory']
    os.makedirs(output_directory, exist_ok=True)  # Create the sims directory if it doesn't exist
    sim_runner = SimRunner(output_directory=output_directory)
    sim_runner.run()
