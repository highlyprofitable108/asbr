import sqlite3
import gzip
import os
import csv
import shutil
from io import StringIO
from datetime import datetime
from constants import constants
from utils.api_utils import make_api_call, get_weather_data


def backup_db(db_path_with_name: str) -> None:
    """
    Creates a backup of the SQLite database and compresses it.

    Args:
        db_path_with_name (str): The path to the database file
            along with the database filename.
    """
    # Database directory and name are extracted from the provided path
    db_dir, db_name = os.path.split(db_path_with_name)

    # Current timestamp is used to create a unique name for the backup file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f'{os.path.splitext(db_name)[0]}_backup_{timestamp}.bak'
    backup_path = os.path.join(db_dir, backup_name)

    # Try to backup the database. If it fails, print the error
    try:
        # The 'source_db' is opened in read-only mode,
        #   and its content is copied to 'backup_db'
        with sqlite3.connect(
            f'file:{db_path_with_name}?mode=ro', uri=True
        ) as source_db:
            with sqlite3.connect(backup_path) as backup_db:
                source_db.backup(backup_db)
    except sqlite3.Error as e:
        print(f"Error occurred during DB backup: {e}")

    # Try to compress the backup. If it fails, print the error
    try:
        # The backup file is opened in read-binary mode
        # `and its content is copied to a gzip file
        with open(backup_path, 'rb') as f_in, gzip.open(
            f'{backup_path}.gz', 'wb'
        ) as f_out:
            shutil.copyfileobj(f_in, f_out)
        # Once compressed, the uncompressed backup is removed
        os.remove(backup_path)
    except Exception as e:
        print(f"Error occurred during backup compression: {e}")


def read_sql_file(file_name):
    """
    Reads an SQL script from a file and returns it as a string.

    Args:
        file_name (str): The name of the SQL file.

    Returns:
        sql (str): The SQL script as a string.
    """
    # File path is determined by adding the file name to a predefined SQL
    #   scripts directory path
    file_path = constants['sql_path'] + file_name

    # Open the file, read lines, join them into a single string and return
    with open(file_path, 'r') as file:
        sql = " ".join(file.readlines())
    return sql


def csv_to_dict_list(csv_string):
    """
    Converts a CSV string into a list of dictionaries.

    Args:
        csv_string (str): The CSV string.

    Returns:
        list: A list of dictionaries where each dictionary corresponds to
            a row in the CSV.
    """
    # csv.DictReader takes a file-like object,
    #   here we use StringIO to transform string to file-like object
    csv_reader = csv.DictReader(StringIO(csv_string.decode()))

    # Convert the CSV reader object to a list and return
    return list(csv_reader)


def get_historical_raw_data_event_ids(cursor: sqlite3.Cursor) -> None:
    """
    Retrieves historical raw data event IDs and inserts them into the database.

    Args:
        cursor (sqlite3.Cursor): A SQLite cursor to execute queries.
    """
    # The API endpoint is built using predefined constants
    endpoint = (
        f"{constants['base_url']}/historical-raw-data/event-list?"
        f"file_format={constants['file_format']}&"
        f"key={constants['api_key']}"
    )

    # Data from API call is converted from CSV to a list of dictionaries
    data = csv_to_dict_list(make_api_call(endpoint))

    # Read SQL script for inserting data from the file
    sql_script = read_sql_file("historical_raw_event_ids.sql")

    # Prepare data for insertion into the database
    data_to_insert = [
        (
            row['calendar_year'],
            row['date'],
            row['event_id'],
            row['sg_categories'],
            row['event_name']
        ) for row in data
    ]

    # Try to insert data into the database. If it fails, print the error
    try:
        cursor.executemany(sql_script, data_to_insert)
    except sqlite3.Error as e:
        print(f"Error occurred during data insertion: {e}")


def get_round_scoring(cursor: sqlite3.Cursor) -> None:
    """
    Retrieves round scoring data and inserts it into the database.

    Args:
        cursor (sqlite3.Cursor): A SQLite cursor to execute queries.
    """
    # For each combination of tour and year...
    for tour in constants['tours']:
        for year in constants['years']:
            # ...build an API endpoint...
            endpoint = (
                f"{constants['base_url']}/historical-raw-data/rounds?"
                f"tour={tour}&"
                f"event_id={constants['event_id']}&"
                f"year={year}&"
                f"file_format={constants['file_format']}&"
                f"key={constants['api_key']}"
            )

            # ...fetch data from the API and convert it from CSV
            #   to a list of dictionaries...
            data = csv_to_dict_list(make_api_call(endpoint))

            # ...read SQL script for inserting data from the file...
            sql_script = read_sql_file("get_round_scoring.sql")

            # ...and prepare data for insertion into the database.
            data_to_insert = [
                (
                    row.get('tour'),
                    row.get('year'),
                    row.get('date'),
                    row.get('event_id'),
                    row.get('event_name'),
                    row.get('dg_id'),
                    row.get('player_name'),
                    row.get('fin_text'),
                    row.get('round_num'),
                    row.get('course_num'),
                    row.get('course_name'),
                    row.get('course_par'),
                    row.get('round_score'),
                    row.get('sg_putt'),
                    row.get('sg_arg'),
                    row.get('sg_app'),
                    row.get('sg_ott'),
                    row.get('sg_t2g'),
                    row.get('sg_total'),
                    row.get('driving_dist'),
                    row.get('driving_acc'),
                    row.get('gir'),
                    row.get('scrambling'),
                    row.get('prox_rgh'),
                    row.get('prox_fw')
                ) for row in data
            ]

            # Try to insert data into the database.
            #   If it fails, print the error
            try:
                cursor.executemany(sql_script, data_to_insert)
            except sqlite3.Error as e:
                print(f"Error occurred during data insertion: {e}")


def get_player_stats(db_conn: sqlite3.Connection) -> None:
    """
    Retrieves raw player stats,
        calculates averages and updates the players table in the database.

    Args:
        db_conn (sqlite3.Connection): A SQLite connection to the database.
    """
    # Build the API endpoint
    endpoint = (
        f"{constants['base_url']}/preds/skill-ratings?"
        f"display=value&"
        f"file_format={constants['file_format']}&"
        f"key={constants['api_key']}"
    )

    # Fetch data from the API and convert it from CSV to a list of dictionaries
    data = csv_to_dict_list(make_api_call(endpoint))

    cursor = db_conn.cursor()

    try:
        # Clear temporary player averages table for fresh data
        cursor.execute("DROP TABLE IF EXISTS player_averages_temp")
        db_conn.commit()

        # Create temporary table to hold player averages
        sql_script = read_sql_file("get_player_stats_create_tmp.sql")
        cursor.execute(sql_script)
        db_conn.commit()

        # Calculate and insert new averages into temporary table
        sql_script = read_sql_file("get_player_stats_insert_tmp.sql")
        cursor.execute(sql_script)
        db_conn.commit()

        # Clear the players table for new stats
        cursor.execute("DROP TABLE IF EXISTS players")

        # Create or update the players table schema
        sql_script = read_sql_file("get_player_stats_create.sql")
        cursor.execute(sql_script)
        db_conn.commit()

        # Insert player stats into the players table
        sql_script = read_sql_file("get_player_stats_insert.sql")
        data_to_insert = [
            (
                row['dg_id'],
                row['player_name'],
                row['sg_putt'],
                row['sg_arg'],
                row['sg_app'],
                row['sg_ott'],
                (
                    float(row['sg_arg'])
                    + float(row['sg_app'])
                    + float(row['sg_ott'])
                ),
                row['sg_total'],
                row['dg_id']
            )
            for row in data
        ]

        cursor.executemany(sql_script, data_to_insert)
        db_conn.commit()

    except sqlite3.Error as e:
        print(f"Error occurred during player stats processing: {e}")


def populate_course_data(db_conn: sqlite3.Connection) -> None:
    """
    Retrieves and updates course data,
    and fetches location data for courses if necessary.

    Args:
        db_conn (sqlite3.Connection): A SQLite connection to the database.
    """
    cursor = db_conn.cursor()

    try:
        # Insert new courses into the courses table
        #   or ignore if they already exist
        sql_script = read_sql_file("populate_course_data_insert.sql")
        cursor.execute(sql_script)
        db_conn.commit()

        # Update the statistical data for existing courses
        sql_script = read_sql_file("populate_course_data_update.sql")
        cursor.execute(sql_script)
        db_conn.commit()

        # Fetch courses whose location data hasn't been retrieved yet
        """
        cursor.execute("SELECT DISTINCT course_name
            FROM courses WHERE location_fetched = 0")
        courses = cursor.fetchall()

        for course in courses:
            course_name = course[0]
            lat, lon = get_location_data(course_name)

            if lat is not None and lon is not None:
                # Update course location data in the courses table
                cursor.execute(
                    UPDATE courses
                    SET latitude = ?, longitude = ?, location_fetched = 1
                    WHERE course_name = ?
                , (lat, lon, course_name))
                db_conn.commit()
        """

    except sqlite3.Error as e:
        print(f"Error occurred during course data processing: {e}")


def populate_weather_table(db_conn: sqlite3.Connection) -> None:
    """
    Retrieves weather data for each unique date
    and course combination
    and updates the weather table accordingly.

    Args:
        db_conn (sqlite3.Connection): A SQLite connection to the database.
    """
    cursor = db_conn.cursor()

    try:
        # Fetch unique date and course combinations
        cursor.execute("SELECT DISTINCT date, course_name FROM rounds")
        unique_date_course_combinations = cursor.fetchall()

        for date_course in unique_date_course_combinations:
            if date_course[0] is None:  # check if date string is None
                continue  # skip this iteration if it is None

            try:
                # Check if date string is in correct format
                date = datetime.strptime(date_course[0].split()[0], '%Y-%m-%d')
            except ValueError:
                print(f"Unexpected date format for {date_course[0]}")
                continue
            except Exception as e:
                print(f"Unexpected error: {e}")
                continue

            course_name = date_course[1]

            # Fetch course location data
            sql_script = read_sql_file("get_location_data.sql")
            cursor.execute(sql_script, (course_name,))
            course_data = cursor.fetchone()

            if course_data is None:
                continue

            lat = course_data[0]
            lon = course_data[1]

            # Check if weather data for the given date
            #   and course already exists
            sql_script = read_sql_file("get_weather_data.sql")
            cursor.execute(sql_script, (date, course_name))
            existing_weather_data = cursor.fetchone()

            if existing_weather_data is None:
                # Fetch and insert new weather data
                weather_data = get_weather_data(date, lat, lon)
                if weather_data is not None:
                    cursor.execute(
                        sql_script,
                        (date, course_name, *weather_data)
                    )
                    db_conn.commit()
    except sqlite3.Error as e:
        print(f"Error occurred during weather data processing: {e}")


def create_model_data_table(db_conn):
    cursor = db_conn.cursor()

    # Drop the model_data table if it already exists
    cursor.execute("""
        DROP TABLE IF EXISTS model_data
    """)

    # Create the new table by joining the rounds and weather tables
    sql_script = read_sql_file("populate_model_data.sql")
    cursor.execute(sql_script)
    db_conn.commit()
