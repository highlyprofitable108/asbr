import os
import glob
import sqlite3
import csv
import json
from collections import defaultdict

# Path to SQLite database file
database_path = '/Users/michaelfuscoletti/Desktop/data/{sport}.db'

# Path to CSV files
csv_file_path = '/Users/michaelfuscoletti/Desktop/data/{sport}'

def get_sport_type():
    # List of valid sports
    valid_sports = ['pgatour', 'nfl', 'ncaaf', 'ncaab', 'horse racing']

    # Ask for user input
    sport = input("Please enter the sport type (pgatour, nfl, ncaaf, ncaab, horse racing): ")

    # Validate user input
    while sport not in valid_sports:
        print("Invalid sport type. Please try again.")
        sport = input("Please enter the sport type (pgatour, nfl, ncaaf, ncaab, horse racing): ")

    return sport

def create_table(cursor, table_name, table_statement, unique_columns):
    cursor.execute(f'CREATE TABLE IF NOT EXISTS {table_name} {table_statement}')
    if unique_columns:
        cursor.execute(f'CREATE UNIQUE INDEX IF NOT EXISTS idx_{table_name}_unique ON {table_name} ({", ".join(unique_columns)})')

def insert_data_from_csv(cursor, table_name, table_definition, file_path):
    with open(file_path, 'r', newline='', encoding='utf-8-sig') as file:
        reader = csv.reader(file, delimiter=',', quotechar='"')
        header = next(reader, None) 
        expected_columns = len(table_definition.split(',')) 
        if header is not None and len(header) == expected_columns:
            columns = ', '.join(header)
            placeholders = ', '.join(['?'] * len(header))
            query = f"INSERT OR IGNORE INTO {table_name} ({columns}) VALUES ({placeholders})"
            cursor.executemany(query, reader)
        else:
            print(f"Unexpected number of columns in {file_path}. Expected {expected_columns}, but found {len(header)}.")

def process_files(cursor, table_definitions):
    table_files = defaultdict(list)
    for root, dirs, files in os.walk(csv_file_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                if os.path.getsize(file_path) == 0:
                    print(f"Skipping empty file: {file_path}")
                    continue
                for table_name in table_definitions.keys():
                    if table_name in file:
                        table_files[table_name].append(file_path)
                        break
    for table_name, file_paths in table_files.items():
        for file_path in file_paths:
            print(f"Processing file: {file_path}")
            if table_name in table_definitions:
                insert_data_from_csv(cursor, table_name, table_definitions[table_name], file_path)
            else:
                print(f"Unknown table: {table_name}. Skipping this file.")

def main():
    sport = get_sport_type()
    
    # Update the paths
    global database_path
    global csv_file_path
    database_path = database_path.format(sport=sport)
    csv_file_path = csv_file_path.format(sport=sport)
    
    with open(f'{sport}_table_definitions.json') as f:
        table_definitions = json.load(f)
    
    # Connect to the SQLite database
    with sqlite3.connect(database_path) as conn:
        cursor = conn.cursor()
        for table_name, table_statement in table_definitions.items():
            create_table(cursor, table_name, table_statement)
        process_files(cursor, table_definitions)
        conn.commit()

def main():
    sport = get_sport_type()

    # Update the paths
    global database_path
    global csv_file_path
    database_path = database_path.format(sport=sport)
    csv_file_path = csv_file_path.format(sport=sport)
    
    with open(f'{sport}_table_definitions.json') as f:
        table_definitions = json.load(f)
    
    # Connect to the SQLite database
    with sqlite3.connect(database_path) as conn:
        cursor = conn.cursor()
        for table_name, data in table_definitions.items():
            create_table(cursor, table_name, data['table_definition'], data.get('unique_columns', []))
        process_files(cursor, table_definitions)
        conn.commit()