import os
import requests
import logging
import pandas as pd

# Enable logging
logging.basicConfig(filename='warnings.log', level=logging.WARNING)

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

# Your file format
file_format = 'csv'

# Set event_id as 'all'
event_id = 'all'

def fetch_api_data(endpoint):
    response = requests.get(endpoint)
    try:
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
    except json.JSONDecodeError:
        logging.warning(f"Bad data received from API for endpoint: {endpoint}")
        return None

def get_historical_raw_data_event_ids(file_format='json'):
    endpoint = f"{base_url}/historical-raw-data/event-list?file_format={file_format}&key={api_key}"
    return fetch_api_data(endpoint)

def get_round_scoring_stats_strokes_gained(tour, event_id, year, file_format='json'):
    endpoint = f"{base_url}/historical-raw-data/rounds?tour={tour}&event_id={event_id}&year={year}&file_format={file_format}&key={api_key}"
    return fetch_api_data(endpoint)

def get_historical_odds_data_event_ids(tour='pga', file_format='json'):
    endpoint = f"{base_url}/historical-odds/event-list?tour={tour}&file_format={file_format}&key={api_key}"
    return fetch_api_data(endpoint)

def get_historical_outrights(tour, event_id, year, market, book, odds_format='decimal', file_format='json'):
    endpoint = f"{base_url}/historical-odds/outrights?tour={tour}&event_id={event_id}&year={year}&market={market}&book={book}&odds_format={odds_format}&file_format={file_format}&key={api_key}"
    return fetch_api_data(endpoint)

def get_historical_matchups(tour, event_id, year, book, odds_format='decimal', file_format='json'):
    endpoint = f"{base_url}/historical-odds/matchups?tour={tour}&event_id={event_id}&year={year}&book={book}&odds_format={odds_format}&file_format={file_format}&key={api_key}"
    return fetch_api_data(endpoint)

# Loop through all tours and years
for tour in tours:
    for year in years:
        # Initialize empty DataFrame
        df_total = pd.DataFrame()

        for book in books:
            # Print debugging information
            print(f"Fetching data for tour: {tour}, event_id: {event_id}, year: {year}, book: {book}")

            # Call API
            raw_data = get_round_scoring_stats_strokes_gained(tour, event_id, year, file_format)
            outrights = get_historical_outrights(tour, event_id, year, market, book, odds_format, file_format)
            matchups = get_historical_matchups(tour, event_id, year, book, odds_format, file_format)

            # Convert to DataFrame
            df_raw_data = pd.DataFrame(raw_data)
            df_outrights = pd.DataFrame(outrights)
            df_matchups = pd.DataFrame(matchups)

            # Concatenate DataFrames
            df_temp = pd.concat([df_raw_data, df_outrights, df_matchups], axis=0)
            df_total = pd.concat([df_total, df_temp], axis=0)

        # Save to CSV
        df_total.to_csv(f'{csv_file_path}/historical_data_{tour}_{year}.csv', index=False)

        # Print some data for debugging
        print(df_total.head())
