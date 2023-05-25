import os
import glob
import sqlite3
import csv
from collections import defaultdict

# Path to SQLite database file
database_path = '/Users/michaelfuscoletti/Desktop/data/pgatour.db'

# Path to CSV files
csv_file_path = '/Users/michaelfuscoletti/Desktop/data'

# Connect to the SQLite database
conn = sqlite3.connect(database_path)
cursor = conn.cursor()

# Function to create table
def create_table(table_name, table_statement):
    cursor.execute(f'CREATE TABLE IF NOT EXISTS {table_name} {table_statement}')

# Table creation definitions
table_definitions = {
    'raw_data': '''
    (
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
    'matchups': '''
    (
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
    'outrights': '''
    (
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
}

# Execute table creation statements
for table_name, table_statement in table_definitions.items():
    create_table(table_name, table_statement)

# Dictionary for the files 
table_files = defaultdict(list)

# Get all CSV files that match table names
for root, dirs, files in os.walk(csv_file_path):
    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join(root, file)
            
            # Continue to next file if current file size is 0
            if os.path.getsize(file_path) == 0:
                print(f"Skipping empty file: {file_path}")
                continue

            for table_name in table_definitions.keys():
                if table_name in file:
                    table_files[table_name].append(file_path)
                    break

# Process CSV files
for table_name, file_paths in table_files.items():
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        if table_name in table_definitions:
            with open(file_path, 'r', newline='', encoding='utf-8-sig') as file:
                reader = csv.reader(file, delimiter=',', quotechar='"')
                header = next(reader, None) 

                # If raw_data file has 28 columns, ignore the last two columns
                if table_name == "raw_data" and len(header) == 28:
                    header = header[:-2]
                    reader = (row[:-2] + [None]*(len(header)-len(row[:-2])) for row in reader)

                # Check for expected number of columns based on table definitions
                expected_columns = len(table_definitions[table_name].split(',')) 
                if header is not None and len(header) == expected_columns:
                    columns = ', '.join(header)
                    placeholders = ', '.join(['?'] * len(header))
                    query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
                    cursor.executemany(query, reader)
                else:
                    print(f"Unexpected number of columns in {file_path}. Expected {expected_columns}, but found {len(header)}.")

        else:
            print(f"Unknown table: {table_name}. Skipping this file.")

# Commit the changes to the database
conn.commit()

# Close the database connection
conn.close()