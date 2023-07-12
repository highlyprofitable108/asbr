import sqlite3
from constants import constants
from utils.database_utils import (
    get_round_scoring,
    get_historical_raw_data_event_ids,
    get_player_stats,
    populate_course_data,
    backup_db,
    populate_weather_table,
    create_model_data_table
)


def main():
    # Establish a connection to the SQLite database
    #   specified in the 'constants' dictionary
    db_path = constants['database_path']
    db_conn = sqlite3.connect(db_path)
    print("Connected to database.")

    # Backup the database before making changes to preserve data integrity
    backup_db(db_path)
    print("Database backed up.")

    # Fetch and preprocess the historical raw data event IDs
    get_historical_raw_data_event_ids(db_conn)
    print("Historical data populated.")

    # Fetch and preprocess the round scoring statistics
    #   and strokes gained data for analysis
    get_round_scoring(db_conn)
    print("Round data populated.")

    # Fetch and preprocess player statistics,
    #   which are a critical factor in our predictive model
    get_player_stats(db_conn)
    print("Player data populated.")

    # Fetch and preprocess course data,
    #   which includes details about the venue of the tournament
    populate_course_data(db_conn)
    print("Course data populated.")

    # Fetch and preprocess weather data,
    #   as it can influence the game and affect predictions
    populate_weather_table(db_conn)
    print("Weather data populated.")

    # Create a table to store the data used in the predictive model
    create_model_data_table(db_conn)

    # Close the connection to the SQLite database to conserve system resources
    db_conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
