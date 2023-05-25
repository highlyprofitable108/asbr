import os
import glob
import requests
import pandas as pd
from datetime import datetime
import time
import csv

# API key, base URL, CSV path, and other constants
api_key = '195c3cb68dd9f46d7feaafc4829c'
base_url = "https://feeds.datagolf.com"
csv_file_path = '/Users/michaelfuscoletti/Desktop/data'
tours = ['pga', 'euro', 'kft', 'alt']
years = list(range(2017, 2023))
market = 'win'
file_format = 'csv'
books = ['betmgm', 'draftkings', 'Fanduel']
odds_format = 'american'
event_id = 'all'
request_delay = 60
max_requests_per_minute = 50

requests_counter = 0
last_request_time = None
error_occurred = False

# Function to create backups of old files
def backup_old_files(directory, file_name):
    current_bak_files = sorted(glob.glob(os.path.join(directory, f"{file_name}*.bak")))
    if len(current_bak_files) >= 3:
        os.remove(current_bak_files[0])
        current_bak_files.pop(0)
    os.rename(os.path.join(directory, f"{file_name}.csv"), os.path.join(directory, f"{file_name}_{len(current_bak_files) + 1}.bak"))

# Function to handle errors during API requests
def handle_error(error_message):
    global error_occurred
    error_occurred = True
    print(f"An error occurred: {error_message}")
    # Additional error handling logic or notifications can be added here

# Function to sanitize the CSV files returned by the API
def sanitize_csv_file(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"The file {file_path} does not exist.")
        return

    # Check if the file is empty
    if os.stat(file_path).st_size == 0:
        print(f"The file {file_path} is empty.")
        return

    # Open the file, replace the characters and write it back
    with open(file_path, 'r') as file:
        file_data = file.read()

    # Replace all instances of """ with "
    file_data = file_data.replace('"""', '"')

    # Patterns to replace with just a comma
    replace_patterns = ['","', '", Jr.,"', '", Sr.,"', '", Jr,"', '", Sr,"', '", IV,"']

    for pattern in replace_patterns:
        file_data = file_data.replace(pattern, ',')

    with open(file_path, 'w') as file:
        file.write(file_data)

# Function to get data from the API and write it to a CSV file
def get_data_from_api(endpoint, file_path):
    global requests_counter, last_request_time
    try:
        # Check rate limiting
        if last_request_time is not None and (datetime.now() - last_request_time).total_seconds() < 60:
            if requests_counter >= max_requests_per_minute:
                # Delay if the maximum number of requests has been reached
                print(f"Reached maximum requests per minute. Waiting for {request_delay} seconds...")
                time.sleep(request_delay)
                requests_counter = 0
        else:
            requests_counter = 0  # Reset the counter if a minute has passed since the last request

        response = requests.get(endpoint)
        response.raise_for_status()

        # Update requests counter and timestamp
        requests_counter += 1
        last_request_time = datetime.now()

        # Parse response content as CSV
        csv_content = [row.split(',') for row in response.text.split('\n') if row]

        # Save parsed content as CSV file
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(csv_content)

    except requests.exceptions.HTTPError as err:
        handle_error(f"HTTP error occurred: {err}")
    except requests.exceptions.RequestException as err:
        handle_error(f"An exception occurred: {err}")


def get_round_scoring_stats_strokes_gained(tour, event_id, year, file_format, raw_data_file_path):
    endpoint = f"{base_url}/historical-raw-data/rounds?tour={tour}&event_id={event_id}&year={year}&file_format={file_format}&key={api_key}"
    return get_data_from_api(endpoint, raw_data_file_path)

def get_historical_outrights(tour, event_id, year, market, book, odds_format, file_format, outrights_file_path):
    endpoint = f"{base_url}/historical-odds/outrights?tour={tour}&event_id={event_id}&year={year}&market={market}&book={book}&odds_format={odds_format}&file_format={file_format}&key={api_key}"
    return get_data_from_api(endpoint, outrights_file_path)

def get_historical_matchups(tour, event_id, year, book, odds_format, file_format, matchups_file_path):
    endpoint = f"{base_url}/historical-odds/matchups?tour={tour}&event_id={event_id}&year={year}&book={book}&odds_format={odds_format}&file_format={file_format}&key={api_key}"
    return get_data_from_api(endpoint, matchups_file_path)

def main():
    # Get raw data once and store in a separate directory
    raw_data_directory = os.path.join(csv_file_path, 'raw_data')
    if not os.path.exists(raw_data_directory):
        os.makedirs(raw_data_directory)
    for tour in tours:
        for year in years:
            raw_data_file_path = os.path.join(raw_data_directory, f'{tour}_{year}_raw_data.csv')
            if os.path.isfile(raw_data_file_path):
                backup_old_files(raw_data_directory, f'{tour}_{year}_raw_data')
            get_round_scoring_stats_strokes_gained(tour, event_id, year, file_format, raw_data_file_path)

            # Sanitize the downloaded CSV files
            sanitize_csv_file(raw_data_file_path)

    # Loop through all books, tours, and years
    for book in books:
        book_directory = os.path.join(csv_file_path, book)
        for tour in tours:
            tour_directory = os.path.join(book_directory, tour)
            for year in years:
                year_directory = os.path.join(tour_directory, str(year))
                # If directory doesn't exist, create it
                if not os.path.exists(year_directory):
                    os.makedirs(year_directory)
                # Set file paths for outrights and matchups
                outrights_file_path = os.path.join(year_directory, 'outrights.csv')
                matchups_file_path = os.path.join(year_directory, 'matchups.csv')
                # Backup old files if they exist
                if os.path.isfile(outrights_file_path):
                    backup_old_files(year_directory, 'outrights')
                if os.path.isfile(matchups_file_path):
                    backup_old_files(year_directory, 'matchups')
                # Print debugging information
                print(f"Fetching data for book: {book}, tour: {tour}, year: {year}")
                # Call API and retrieve data
                get_historical_outrights(tour, event_id, year, market, book, odds_format, file_format, outrights_file_path)
                get_historical_matchups(tour, event_id, year, book, odds_format, file_format, matchups_file_path)

                # Sanitize the downloaded CSV files
                sanitize_csv_file(outrights_file_path)
                sanitize_csv_file(matchups_file_path)

if __name__ == "__main__":
    main()