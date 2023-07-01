import sqlite3
from constants import constants
from utils.database_utils import get_round_scoring, get_historical_raw_data_event_ids, \
    get_player_stats, populate_course_data, backup_db, populate_first_round_model, \
    populate_tournament_table

#, populate_weather_table
 

def main():
    # Establish connection to the SQLite database using the path provided in constants
    db_path = constants['database_path']
    db_conn = sqlite3.connect(db_path)
    print("Connected to database.")

    # Backing up the database is a good practice before making large scale changes or additions
    # This function would copy the current state of the database to a backup file
    backup_db(db_path)
    print("Database backed up.")

    # Retrieve and process historical raw data event IDs
    # This could include any preprocessing required such as parsing or type conversion
    get_historical_raw_data_event_ids(db_conn)
    print("Historical data populated.")

    # Retrieve and process round scoring stats and strokes gained data
    # This data would be important for later analysis
    get_round_scoring(db_conn)
    print("Round data populated.")

    # Retrieve and process player stats
    # Player performance data could be a crucial factor in our predictions
    get_player_stats(db_conn)
    print("Player data populated.")

    # Retrieve and process course data
    # This data would contain information about the golf course where the tournament is held
    populate_course_data(db_conn)
    print("Course data populated.")

    # Retrieve and process weather data
    # Weather could influence the game, so this data is useful for making predictions
    # populate_weather_table(db_conn)
    # print("Weather data populated.")

    # Retrieve and process first round data
    # This data contains information about the first round of the game
    populate_first_round_model(db_conn)
    print("First round data populated.")

    # Retrieve and process tournament data
    # This function would gather data on each individual tournament
    populate_tournament_table(db_conn)
    print("Tournament table populated.")

    # Close the database connection to free up resources
    db_conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
