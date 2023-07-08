import sqlite3
import datetime
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from constants import constants


def establish_connection():
    # Establish connection to the SQLite database using the path provided in constants
    return sqlite3.connect(constants['database_path'])


def retrieve_rounds_data(db_conn):
    # Retrieve necessary data
    rounds_data = pd.read_sql_query(
        "SELECT date, dg_id, player_name, round_score, course_num, course_name, sg_total, sg_ott, sg_app, sg_arg, sg_putt, weight_calculated FROM rounds",
        db_conn
    )

    return rounds_data


def filter_rounds_data(rounds_data):
    # Convert round_score column to numeric type
    rounds_data['round_score'] = pd.to_numeric(rounds_data['round_score'], errors='coerce')

    # Drop rows with round_score outside the range of 55 to 100
    rounds_data = rounds_data[(rounds_data['round_score'] >= 55) & (rounds_data['round_score'] <= 100)]

    # Convert 'date' column to datetime format
    rounds_data.loc[:, 'date'] = pd.to_datetime(rounds_data['date'])

    return rounds_data


def filter_eligible_players(rounds_data):
    # Set reference dates for the last 365 and 730 days
    ref_date_365 = pd.to_datetime('today') - pd.DateOffset(days=365)
    ref_date_730 = pd.to_datetime('today') - pd.DateOffset(days=730)

    # Filter rounds in the last 365 and 730 days
    rounds_last_365 = rounds_data[rounds_data['date'] >= ref_date_365]
    rounds_last_730 = rounds_data[rounds_data['date'] >= ref_date_730]

    # Count the number of rounds for each player in the last 365 and 730 days
    rounds_per_player_365 = rounds_last_365['player_name'].value_counts()
    rounds_per_player_730 = rounds_last_730['player_name'].value_counts()

    # Get a list of player names who have played 20 or more rounds in the last 365 days and log increase up to 1100 days
    eligible_players_365 = rounds_per_player_365[rounds_per_player_365 >= 20].index
    eligible_players_730 = rounds_per_player_730[rounds_per_player_730 >= np.log(1101)].index

    # Get a list of players who are eligible in both time frames
    eligible_players = list(set(eligible_players_365) & set(eligible_players_730))

    # Filter rounds_data to only include rounds from eligible players
    rounds_data = rounds_data[rounds_data['player_name'].isin(eligible_players)]

    return rounds_data


def update_course_difficulty_table(db_conn, rounds_data):
    # Create a DataFrame with course_num, course_name, difficulty, and variance
    overall_mean = rounds_data['round_score'].mean()  # Calculate the mean of all round scores
    course_data = rounds_data.groupby('course_num').agg({'course_name': 'first', 'round_score': 'mean'}).reset_index()
    course_data['avg_difficulty'] = course_data['round_score'] - overall_mean
    course_data['variance_round_score'] = rounds_data.groupby('course_num')['round_score'].var().values

    # Store the DataFrame in a new table in the database
    course_data.to_sql('course_difficulty', db_conn, if_exists='replace', index=False)

def calculate_golf_time(rounds_data):
    # Calculate golf time function (T)
    rounds_data['date'] = pd.to_datetime(rounds_data['date'])
    rounds_data['days_since_round'] = (pd.to_datetime('today') - rounds_data['date']).dt.days
    rounds_data['days_since_round'] = rounds_data['days_since_round'].fillna(1)
    rounds_data_full = rounds_data
    rounds_data = rounds_data[rounds_data['days_since_round'] <= 1100]

    # Add 'rounds_played' and 'days_since_last_round' to 'player_data'
    date_agg = rounds_data.groupby('player_name').agg({'date': ['count', 'max']})
    date_agg.columns = ['_'.join(col).strip() for col in date_agg.columns.values]
    rounds_data = pd.merge(rounds_data, date_agg, left_on='player_name', right_index=True, how='left')
    rounds_data['rounds_played'] = rounds_data['date_count']
    rounds_data['date_max'] = pd.to_datetime(rounds_data['date_max'])
    rounds_data['days_since_last_round'] = (pd.to_datetime('today') - rounds_data['date_max']).dt.days

    # Drop 'date_count' and 'date_max' columns
    rounds_data = rounds_data.drop(columns=['date_count', 'date_max'])

   # Create weight: 1 for the last 400 days, log decreasing from 401-1100, 0 beyond 1101 days
    # Increase weight linearly with rounds_played and decrease with days_since_last_round
    rounds_data['weight'] = np.where(rounds_data['days_since_round'] <= 400, 1 + rounds_data['rounds_played'] - 0.01 * rounds_data['days_since_last_round'],
                                    np.where(rounds_data['days_since_round'] > 1100, 0,
                                            np.log(1101 - (1101 - rounds_data['days_since_round'])) + rounds_data[
                                                'rounds_played'] - 0.01 * rounds_data['days_since_last_round']))

    return rounds_data, rounds_data_full


def perform_regression(rounds_data, sg_columns):
    # List all column names starting with 'sg_'
    columns = ['round_score'] + sg_columns

    # Dictionary to hold results for each column
    adjusted_difficulty = {}

    # Perform regression for each column
    for column in columns:
        # Check if the column is available and is numerical
        if column in rounds_data.columns and np.issubdtype(rounds_data[column].dtype, np.number):
            # Exclude rows with NaN or inf in the column
            valid_data = rounds_data[~rounds_data[column].isnull() & ~np.isinf(rounds_data[column])]

            # Add constant to independent variable
            X = sm.add_constant(valid_data[['weight', 'rounds_played', 'days_since_last_round']])

            # Prepare the dependent variable
            y = valid_data[column]

            # Perform the regression analysis
            model = sm.OLS(y, X)
            results = model.fit()

            # Store the constant parameter (adjusted difficulty) in the dictionary
            adjusted_difficulty[column] = results.params['const']

    return adjusted_difficulty


def calculate_weighted_averages(rounds_data, adjusted_difficulty):
    # Convert 'weight' to numeric
    rounds_data['weight'] = pd.to_numeric(rounds_data['weight'], errors='coerce')

    # Convert all columns in `columns` to numeric
    columns = ['round_score'] + [col for col in rounds_data.columns if col.startswith('sg_')]
    for col in columns:
        rounds_data[col] = pd.to_numeric(rounds_data[col], errors='coerce')

    # Calculate the weighted sum for each column
    weighted_sums = rounds_data.groupby('player_name').apply(
        lambda df: (df[columns].T * df['weight']).T.sum())

    # Calculate the sum of weights for each player
    weight_sums = rounds_data.groupby('player_name')['weight'].sum()

    # Calculate the weighted averages
    weighted_avgs = weighted_sums.div(weight_sums, axis=0)

    # Add dg_id to the DataFrame
    weighted_avgs['dg_id'] = rounds_data.groupby('player_name')['dg_id'].first()

    # Reset the index
    player_data = weighted_avgs.reset_index()

    # Make 'dg_id' the first column
    cols = player_data.columns.tolist()
    cols.insert(0, cols.pop(cols.index('dg_id')))
    player_data = player_data[cols]

    # Add 'weighted_' prefix to 'round_score' and 'sg_' columns
    player_data.columns = ['weighted_' + col if 'round_score' in col or col.startswith('sg_') else col
                           for col in player_data.columns]

    return player_data


def create_player_performance_table(db_conn, player_data):
    # Store the DataFrame in a new table in the database
    player_data.to_sql('player_performance', db_conn, if_exists='replace', index=False)

    return player_data


def create_missing_columns(db_conn, rounds_data, weighted_sg_columns):
    # Get the column names of the rounds table
    rounds_columns = pd.read_sql_query("PRAGMA table_info(rounds)", db_conn)['name'].values

    # Create the missing columns in the rounds table if they don't exist
    for column in weighted_sg_columns:
        if column not in rounds_columns:
            # Alter the rounds table to add the missing column
            alter_query = f"ALTER TABLE rounds ADD COLUMN {column} REAL"
            db_conn.execute(alter_query)


def update_rounds_table(db_conn, rounds_data_full):
    print("Start updating rounds table...\n")
    
    for column in ['sg_putt', 'sg_ott', 'sg_app', 'sg_arg', 'sg_total']:
        rounds_data_full['weighted_' + column] = np.nan

    for i, (index, round_row) in enumerate(rounds_data_full.iterrows()):
        print(f"\n--- Processing index {index}, iteration {i} ---")
        print("Initial values: ", round_row)

        round_date = round_row['date']
        player_name = round_row['player_name']
        weight_calculation = round_row['weight_calculated']

        rounds_data_copy = rounds_data_full.copy()

        current_date = datetime.datetime.now().date()
        if pd.isnull(round_date):
            round_date = datetime.datetime.combine(current_date, datetime.time()) + datetime.timedelta(days=(3 - current_date.weekday() + 7) % 7)

        if round_date.year >= 2018 and pd.notnull(round_row['sg_total']) and weight_calculation != 1:
            print(f"\nProcessing player: {player_name}, round date: {round_date}")
            player_rounds_index = rounds_data_full[(rounds_data_full['player_name'] == player_name) & (
                        rounds_data_full['date'] < round_date)].index

            if len(player_rounds_index) > 0:
                print(f"Found {len(player_rounds_index)} round indices for player: {player_name}")
                player_rounds_index_copy = rounds_data_copy[(rounds_data_copy['player_name'] == player_name) & (
                rounds_data_copy['date'] < round_date)].index

                player_rounds_data = rounds_data_copy.loc[player_rounds_index_copy]

                if len(player_rounds_data) > 1:
                    print(f"Player {player_name} has more than one round. Proceeding with further calculation...")
                    most_recent_round = player_rounds_data.sort_values('date', ascending=False).iloc[0]

                    player_rounds_data = player_rounds_data[player_rounds_data['date'] < round_date]

                    print("Calculating days since round...")
                    rounds_data_copy.loc[player_rounds_index_copy, 'days_since_round'] = (
                            round_date - rounds_data_copy.loc[player_rounds_index_copy, 'date']).dt.days
                    print(f"Days since round for player {player_name}: \n {rounds_data_copy.loc[player_rounds_index_copy, 'days_since_round']}")

                    print("Calculating weights...")
                    player_rounds_data_copy = player_rounds_data.copy()
                    player_rounds_data_copy.reset_index(drop=True, inplace=True)
                    rounds_data_copy.loc[player_rounds_index_copy, 'rounds_played'] = player_rounds_data_copy.groupby('player_name').cumcount() + 1
                    rounds_data_copy.loc[player_rounds_index_copy, 'weight'] = np.where(
                        rounds_data_copy.loc[player_rounds_index_copy, 'days_since_round'] <= 400, 1,
                        (1101 - rounds_data_copy.loc[player_rounds_index_copy, 'days_since_round']) / (1101 - 401)
                    )
                    rounds_data_copy.loc[player_rounds_index_copy, 'weight'] = rounds_data_copy.loc[
                        player_rounds_index_copy, 'weight'].clip(lower=0)
                    print(f"Weights for player {player_name}: \n {rounds_data_copy.loc[player_rounds_index_copy, 'weight']}")

                    player_rounds_data = rounds_data_copy.loc[player_rounds_index_copy]
                    sg_columns = ['sg_putt', 'sg_ott', 'sg_app', 'sg_arg', 'sg_total']
                    player_rounds_data = player_rounds_data.dropna(subset=['weight'])
                    player_rounds_data['weight'] = pd.to_numeric(player_rounds_data['weight'], errors='coerce')

                    for column in sg_columns:
                        print(f"\nProcessing column: {column}")
                        if pd.notnull(round_row[column]):
                            y = player_rounds_data.loc[player_rounds_data[column].notna(), column]
                            X = sm.add_constant(player_rounds_data.loc[player_rounds_data[column].notna(), 'weight'])
                            y = pd.to_numeric(y, errors='coerce')

                            y_temp = y.dropna()
                            X_temp = X.loc[y_temp.index]

                            if not y_temp.empty and not X_temp.isnull().values.any():
                                print(f"Running OLS for column {column}...")
                                model = sm.OLS(y_temp, X_temp)
                                results = model.fit()
                                weighted_sg_value = results.predict([1, 1])[0]

                                print(f"\nPlayer: {player_name}, Date: {round_date}, Column: {column}")
                                print(f"Y values: \n {y_temp}")
                                print(f"X values: \n {X_temp}")
                                print(f"Parameters: {results.params}")

                                if weighted_sg_value > 5 or weighted_sg_value < -5:
                                    print(f"Outlier detected: Weighted {column} for round on {round_date} for player {player_name}: {weighted_sg_value}")
                                    rounds_data_full.at[index, 'is_outlier'] = True
                                else:
                                    print(f"Updating weighted value for column {column}")
                                    rounds_data_full.at[index, 'weighted_' + column] = weighted_sg_value
                            else:
                                print(f"No valid data for running OLS for column {column}. Using most recent round data.")
                                rounds_data_full.at[index, 'weighted_' + column] = most_recent_round[column]
                else:
                    print(f"Player {player_name} has only one round. No further calculation required.")
                    for column in sg_columns:
                        rounds_data_full.at[index, 'weighted_' + column] = round_row[column]
            else:
                print(f"No previous rounds found for player {player_name}.")
                for column in sg_columns:
                    rounds_data_full.at[index, 'weighted_' + column] = round_row[column]

        if weight_calculation != 1:
            print(f"\nPreparing to update rounds data for player: {player_name}, round date: {round_date}")
            date_str = round_date.strftime('%Y-%m-%d')
            dg_id = round_row['dg_id']

            update_query = """
                UPDATE rounds
                SET
                    weighted_sg_ott = ?,
                    weighted_sg_app = ?,
                    weighted_sg_arg = ?,
                    weighted_sg_putt = ?,
                    weighted_sg_total = ?,
                    weight_calculated = 1  -- Set weight_calculated to 1
                WHERE date = ? AND dg_id= ?
            """
            round_data = (
                rounds_data_full.at[index, 'weighted_sg_ott'],
                rounds_data_full.at[index, 'weighted_sg_app'],
                rounds_data_full.at[index, 'weighted_sg_arg'],
                rounds_data_full.at[index, 'weighted_sg_putt'],
                rounds_data_full.at[index, 'weighted_sg_total'],
                date_str,
                dg_id
            )
            print("Prepared round data: ", round_data)
            db_conn.execute(update_query, round_data)
            # Commit every 1000 iterations
            if i % 1000 == 0 and i > 0:
                print(f"Committing updates after {i} iterations...")
                db_conn.commit()

    print("Final commit...")
    # Commit the changes to the database one final time after the loop
    db_conn.commit()


def main():
    # Establish connection to the SQLite database
    db_conn = establish_connection()

    # Retrieve rounds data from the database
    rounds_data = retrieve_rounds_data(db_conn)

    # Filter rounds data
    rounds_data = filter_rounds_data(rounds_data)

    # Filter eligible players
    rounds_data = filter_eligible_players(rounds_data)

    # Update the course difficulty table
    update_course_difficulty_table(db_conn, rounds_data)

    # Calculate golf time
    rounds_data, rounds_data_full = calculate_golf_time(rounds_data)

    # Perform regression
    sg_columns = ['sg_putt', 'sg_ott', 'sg_app', 'sg_arg', 'sg_total']
    adjusted_difficulty = perform_regression(rounds_data, sg_columns)

    # Calculate weighted averages
    player_data = calculate_weighted_averages(rounds_data, adjusted_difficulty)

    # Create the player performance table
    player_data = create_player_performance_table(db_conn, player_data)

    # Define the list of weighted_sg columns
    weighted_sg_columns = ['weighted_sg_ott', 'weighted_sg_app', 'weighted_sg_arg', 'weighted_sg_putt',
                            'weighted_sg_total']

    # Create missing columns in the rounds table if necessary
    create_missing_columns(db_conn, rounds_data, weighted_sg_columns)

    # Update the rounds table with calculated weighted sg values
    update_rounds_table(db_conn, rounds_data_full)

    # Close the database connection
    db_conn.close()


if __name__ == '__main__':
    main()
