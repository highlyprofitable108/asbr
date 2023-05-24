import os
import requests
import pandas as pd

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

# Your event ids list
event_ids = list(range(1, 101))  # Replace with your actual event ids

# Your years list
years = list(range(2017, 2024))

# Your market
market = 'win'

# Your book list
books = ['betmgm', 'draftkings', 'Fanduel']

# Your odds format
odds_format = 'american'

# Your file format
file_format = 'csv'

# Loop through all tours, events and years
for tour in tours:
    for event_id in event_ids:
        for year in years:
            for book in books:
                # Print debugging information
                print(f"Fetching data for tour: {tour}, event_id: {event_id}, year: {year}, book: {book}")

                # Call API
                outrights = get_historical_outrights(tour, event_id, year, market, book, odds_format, file_format)
                matchups = get_historical_matchups(tour, event_id, year, book, odds_format, file_format)

                # Convert to DataFrame
                df_outrights = pd.DataFrame(outrights)
                df_matchups = pd.DataFrame(matchups)

                # Save to CSV
                df_outrights.to_csv(f'{csv_file_path}/historical_outrights_{tour}_{event_id}_{year}_{book}.csv', index=False)
                df_matchups.to_csv(f'{csv_file_path}/historical_matchups_{tour}_{event_id}_{year}_{book}.csv', index=False)

                # Print some data for debugging
                print(df_outrights.head())
                print(df_matchups.head())
