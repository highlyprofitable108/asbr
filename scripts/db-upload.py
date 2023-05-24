import os
import glob
import sqlite3
import csv

# Path to SQLite database file
database_path = '/Users/michaelfuscoletti/Desktop/data/pgatour.db'

# Path to CSV files
csv_file_path = '/Users/michaelfuscoletti/Desktop/data'

# Increase field size limit
csv.field_size_limit(1000000)

# Connect to the SQLite database
conn = sqlite3.connect(database_path)
cursor = conn.cursor()

# Table creation statements (replace with your own table definitions)
table_statements = [
    '''
    CREATE TABLE IF NOT EXISTS raw_data (
        tour TEXT,
        year INTEGER,
        season INTEGER,
        event_completed TEXT,
        event_name TEXT,
        event_id INTEGER,
        player_name TEXT,
        dg_id INTEGER,
        fin_text TEXT,
        round_num INTEGER,
        course_name TEXT,
        course_num INTEGER,
        course_par INTEGER,
        round_score INTEGER,
        sg_putt REAL,
        sg_arg REAL,
        sg_app REAL,
        sg_ott REAL,
        sg_t2g REAL,
        sg_total REAL,
        driving_dist REAL,
        driving_acc REAL,
        gir REAL,
        scrambling REAL,
        prox_rgh REAL,
        prox_fw REAL
    )
    ''',
    '''
    CREATE TABLE IF NOT EXISTS matchups (
        p3_outcome_text TEXT,
        p3_close REAL,
        p3_player_name TEXT,
        p2_outcome_text TEXT,
        p2_outcome REAL,
        close_time TEXT,
        bet_type TEXT,
        p1_outcome_text TEXT,
        p3_open REAL,
        p2_open REAL,
        p3_dg_id INTEGER,
        p1_outcome REAL,
        tie_rule TEXT,
        p2_close REAL,
        p1_open REAL,
        p2_dg_id INTEGER,
        p3_outcome REAL,
        p1_dg_id INTEGER,
        p1_player_name TEXT,
        open_time TEXT,
        p2_player_name TEXT,
        p1_close REAL,
        book TEXT,
        event_completed TEXT,
        event_name TEXT,
        season INTEGER,
        year INTEGER,
        event_id INTEGER
    )
    ''',
    '''
    CREATE TABLE IF NOT EXISTS outright (
        outcome TEXT,
        close_time TEXT,
        open_time TEXT,
        open_odds REAL,
        close_odds REAL,
        player_name TEXT,
        bet_outcome_text TEXT,
        bet_outcome_numeric REAL,
        dg_id INTEGER,
        book TEXT,
        event_completed TEXT,
        event_name TEXT,
        market TEXT,
        season INTEGER,
        year INTEGER,
        event_id INTEGER
    )
    '''
]

# Execute table creation statements
for table_statement in table_statements:
    cursor.execute(table_statement)

# Validate and fix CSV files
def validate_and_fix_csv(file_path, expected_columns):
    try:
        with open(file_path, 'r', newline='') as file:
            rows = file.readlines()

        # Check if any rows are empty or have incorrect number of columns
        invalid_rows = [row for row in rows if not row.strip() or len(row.strip().split(',')) != expected_columns]

        if invalid_rows:
            print(f"Invalid rows found in {file_path}. Fixing formatting issues...")

            # Remove invalid rows from the list
            cleaned_rows = [row for row in rows if row not in invalid_rows]

            # Write the cleaned rows back to the CSV file
            with open(file_path, 'w', newline='') as file:
                file.writelines(cleaned_rows)

            print("Formatting issues fixed.")
        else:
            print(f"No formatting issues found in {file_path}.")

    except IOError:
        print(f"Error reading or writing to {file_path}.")

# Function to fix formatting issues in a row
def fix_formatting(row, expected_columns):
    if len(row) != expected_columns:
        # Fix any formatting issues in the row
        row = [column.strip() for column in row]

    return row

# Process CSV files
for file_path in glob.glob(os.path.join(csv_file_path, '*.csv')):
    print(f"Processing file: {file_path}")

    # Determine the table name based on the file name
    table_name = os.path.splitext(os.path.basename(file_path))[0]

    # Truncate the table before inserting new data
    # cursor.execute(f"DELETE FROM {table_name}")

    # Get the expected number of columns for the table
    if table_name == 'raw_data':
        expected_columns = 23
    elif table_name == 'matchups':
        expected_columns = 28
    elif table_name == 'outright':
        expected_columns = 15
    else:
        expected_columns = 0  # Adjust based on your table's column count

    # Validate and fix CSV file
    validate_and_fix_csv(file_path, expected_columns)

    # Insert data into the table
    with open(file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)
        columns = ', '.join(header)
        placeholders = ', '.join(['?'] * len(header))
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        cursor.executemany(query, reader)

    # Commit the changes to the database
    conn.commit()

# Close the database connection
conn.close()
