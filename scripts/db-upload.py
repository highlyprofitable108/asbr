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
    CREATE TABLE IF NOT EXISTS outrights (
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

# Get the latest version of each CSV file
latest_files = {}
for file_path in glob.glob(os.path.join(csv_file_path, '*.csv')):
    file_name = os.path.basename(file_path)
    file_name_no_ext = os.path.splitext(file_name)[0]  # Remove the file extension
    if file_name_no_ext not in latest_files:
        latest_files[file_name_no_ext] = file_path
    else:
        current_mtime = os.path.getmtime(file_path)
        previous_mtime = os.path.getmtime(latest_files[file_name_no_ext])
        if current_mtime > previous_mtime:
            latest_files[file_name_no_ext] = file_path

# Process CSV files
for file_path in latest_files.values():
    print(f"Processing file: {file_path}")

    # Determine the table name based on the file name without the timestamp
    table_name = os.path.splitext(os.path.basename(file_path))[0].rsplit('_', 1)[0]

    # Get the expected number of columns for the table
    if table_name == 'raw_data':
        expected_columns = 27
    elif table_name == 'matchups':
        expected_columns = 28
    elif table_name == 'outrights':
        expected_columns = 16
    else:xs\\
        expected_columns = 0  # Adjust based on your table's column count

    # Insert data into the table
    with open(file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader, None)  # Use next() function with a default value of None
        if header is not None and len(header) == expected_columns:
            # Remove the _timestamp portion from the header
            header = [column.split('_')[0] for column in header]
            
            columns = ', '.join(header)
            placeholders = ', '.join(['?'] * len(header))
            query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
            cursor.executemany(query, reader)

    # Commit the changes to the database
    conn.commit()

# Close the database connection
conn.close()
