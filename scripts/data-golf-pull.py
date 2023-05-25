import os
import requests
import pandas as pd
from datetime import datetime
import time
import csv

# API key
# api_key = os.environ['API_KEY']
api_key = '195c3cb68dd9f46d7feaafc4829c'

# Base URL
base_url = "https://feeds.datagolf.com"

# Define the path for CSV files
csv_file_path = '/Users/michaelfuscoletti/Desktop/data'

# Check if the directory exists, if not, create it
if not os.path.exists(csv_file_path):
    os.makedirs(csv_file_path)

# Your tour list
tours = ['pga', 'euro', 'kft', 'alt']

# Your years list
years = list(range(2017, 2023))

# Your market
market = 'win'

# File format
file_format = 'csv'

# Your book list
books = ['betmgm', 'draftkings', 'Fanduel']

# Your odds format
odds_format = 'american'

# Set event_id as 'all'
event_id = 'all'

# Rate limiting parameters
request_delay = 1  # Delay between API requests in seconds
max_requests_per_minute = 50  # Maximum number of API requests per minute

# Counter for tracking requests
requests_counter = 0

def handle_error(error_message):
    print(f"An error occurred: {error_message}")
    # Additional error handling logic or notifications can be added here

def sanitize_csv_file(file_path):
    temp_file_path = f"{file_path}.tmp"

    # Read the CSV file using pandas
    df = pd.read_csv(file_path, quoting=csv.QUOTE_ALL, escapechar="\\")
    
    # Remove leading and trailing double quotes from each cell
    df = df.applymap(lambda x: x.strip('"') if isinstance(x, str) else x)

    # Write the sanitized DataFrame to a new CSV file
    df.to_csv(temp_file_path, index=False, header=False, quoting=csv.QUOTE_ALL, escapechar="\\")
    
    # Replace the original file with the sanitized file
    os.replace(temp_file_path, file_path)
    
    # Remove the first character of the first line
    with open(file_path, 'r+') as file:
        lines = file.readlines()
        lines[0] = lines[0][1:]
        file.seek(0)
        file.writelines(lines)
        file.truncate()

    # Remove the last character of the last line
    with open(file_path, 'r+') as file:
        lines = file.readlines()
        lines[-1] = lines[-1][:-1]
        file.seek(0)
        file.writelines(lines)
        file.truncate()

    # Replace "" occurrences with "
    with open(file_path, 'r+') as file:
        content = file.read()
        file.seek(0)
        file.write(content.replace('""', '"'))
        file.truncate()

def get_data_from_api(endpoint, file_path):
    global requests_counter
    try:
        # Check rate limiting
        if requests_counter >= max_requests_per_minute:
            # Delay if the maximum number of requests has been reached
            print(f"Reached maximum requests per minute. Waiting for {request_delay} seconds...")
            time.sleep(request_delay)
            requests_counter = 0

        response = requests.get(endpoint)
        response.raise_for_status()

        # Update requests counter
        requests_counter += 1

        # Save response content as CSV file
        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([response.text])

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
    # Create timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # Set file paths
    raw_data_file_name = f'raw_data_{timestamp}.csv'
    raw_data_file_path = os.path.join(csv_file_path, raw_data_file_name)

    outrights_file_name = f'outrights_{timestamp}.csv'
    outrights_file_path = os.path.join(csv_file_path, outrights_file_name)

    matchups_file_name = f'matchups_{timestamp}.csv'
    matchups_file_path = os.path.join(csv_file_path, matchups_file_name)

    # Save column headers to the CSV files
    with open(raw_data_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Raw Data'])

    with open(outrights_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Outrights'])

    with open(matchups_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Matchups'])

    # Loop through all tours and years
    for tour in tours:
        for year in years:

            for book in books:
                # Print debugging information
                print(f"Fetching data for tour: {tour}, event_id: {event_id}, year: {year}, book: {book}")

                # Call API and retrieve data
                get_round_scoring_stats_strokes_gained(tour, event_id, year, file_format, raw_data_file_path)
                get_historical_outrights(tour, event_id, year, market, book, odds_format, file_format, outrights_file_path)
                get_historical_matchups(tour, event_id, year, book, odds_format, file_format, matchups_file_path)

                # Sanitize the downloaded CSV files
                sanitize_csv_file(raw_data_file_path)
                sanitize_csv_file(outrights_file_path)
                sanitize_csv_file(matchups_file_path)

if __name__ == "__main__":
    main()
