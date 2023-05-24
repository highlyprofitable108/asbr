import os
import requests
import pandas as pd
from datetime import datetime

# API key
api_key = os.environ['API_KEY']

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

# Your book list
books = ['betmgm', 'draftkings', 'Fanduel']

# Your odds format
odds_format = 'american'

# Set event_id as 'all'
event_id = 'all'


def handle_error(error_message):
    print(f"An error occurred: {error_message}")
    # Additional error handling logic or notifications can be added here


def get_data_from_api(endpoint):
    try:
        response = requests.get(endpoint)
        response.raise_for_status()
        print("Status Code:", response.status_code)
        print("Response Text:", response.text)
        return response.json()
    except requests.exceptions.HTTPError as err:
        handle_error(f"HTTP error occurred: {err}")
    except requests.exceptions.RequestException as err:
        handle_error(f"An exception occurred: {err}")
    except ValueError as err:
        handle_error(f"Error parsing JSON: {err}")
    return None


def get_round_scoring_stats_strokes_gained(tour, event_id, year, file_format):
    endpoint = f"{base_url}/historical-raw-data/rounds?tour={tour}&event_id={event_id}&year={year}&file_format={file_format}&key={api_key}"
    return get_data_from_api(endpoint)


def get_historical_outrights(tour, event_id, year, market, book, odds_format, file_format):
    endpoint = f"{base_url}/historical-odds/outrights?tour={tour}&event_id={event_id}&year={year}&market={market}&book={book}&odds_format={odds_format}&file_format={file_format}&key={api_key}"
    return get_data_from_api(endpoint)


def get_historical_matchups(tour, event_id, year, book, odds_format, file_format):
    endpoint = f"{base_url}/historical-odds/matchups?tour={tour}&event_id={event_id}&year={year}&book={book}&odds_format={odds_format}&file_format={file_format}&key={api_key}"
    return get_data_from_api(endpoint)


# Loop through all tours and years
for tour in tours:
    for year in years:
        # Initialize empty DataFrames
        df_raw_data = pd.DataFrame()
        df_outrights = pd.DataFrame()
        df_matchups = pd.DataFrame()

        for book in books:
            # Print debugging information
            print(f"Fetching data for tour: {tour}, event_id: {event_id}, year: {year}, book: {book}")

            # Call API and retrieve data
            raw_data = get_round_scoring_stats_strokes_gained(tour, event_id, year, file_format)
            outrights = get_historical_outrights(tour, event_id, year, market, book, odds_format, file_format)
            matchups = get_historical_matchups(tour, event_id, year, book, odds_format, file_format)

            # Check if data retrieval was successful
            if raw_data is None or outrights is None or matchups is None:
                # Skip this iteration if data retrieval failed
                continue

            # Optionally, you can also parse the data as DataFrames if needed
            # df_raw_data = pd.DataFrame(raw_data)
            # df_outrights = pd.DataFrame(outrights)
            # df_matchups = pd.DataFrame(matchups)

            # Save raw data to CSV with timestamp in file name
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            raw_data_file_name = f'raw_data_{timestamp}.csv'
            raw_data_file_path = os.path.join(csv_file_path, raw_data_file_name)
            with open(raw_data_file_path, 'w') as file:
                file.write(str(raw_data))

            # Save outrights data to CSV with timestamp in file name
            outrights_file_name = f'outrights_{timestamp}.csv'
            outrights_file_path = os.path.join(csv_file_path, outrights_file_name)
            with open(outrights_file_path, 'w') as file:
                file.write(str(outrights))

            # Save matchups data to CSV with timestamp in file name
            matchups_file_name = f'matchups_{timestamp}.csv'
            matchups_file_path = os.path.join(csv_file_path, matchups_file_name)
            with open(matchups_file_path, 'w') as file:
                file.write(str(matchups))

        # You can further process the DataFrames if needed
        # Concatenate or merge the DataFrames, perform additional operations, etc.

        # Print some data for debugging
        print(df_raw_data.head())
        print(df_outrights.head())
        print(df_matchups.head())
