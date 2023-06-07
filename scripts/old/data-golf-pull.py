import os
import glob
import requests
import time
import csv
from datetime import datetime
from pathlib import Path
from ratelimiter import RateLimiter
from typing import List

# Constants (you can choose to place these in a configuration file)
CONSTANTS = {
    'api_key': '195c3cb68dd9f46d7feaafc4829c',
    'base_url': "https://feeds.datagolf.com",
    'csv_file_path': Path('/Users/michaelfuscoletti/Desktop/data'),
    'tours': ['pga', 'euro', 'kft', 'alt'],
    'years': list(range(2017, 2023)),
    'market': 'win',
    'file_format': 'csv',
    'books': ['betmgm', 'draftkings', 'Fanduel'],
    'odds_format': 'american',
    'event_id': 'all',
    'max_requests_per_minute': 50
}

class DataRetriever:
    def __init__(self):
        self.requests_counter = 0
        self.last_request_time = None
        self.error_occurred = False

    @staticmethod
    def backup_old_files(directory: Path, file_name: str):
        current_bak_files = sorted(directory.glob(f"{file_name}*.bak"))
        if len(current_bak_files) >= 3:
            os.remove(current_bak_files[0])
            current_bak_files.pop(0)
        os.rename(directory / f"{file_name}.csv", directory / f"{file_name}_{len(current_bak_files) + 1}.bak")

    @staticmethod
    def sanitize_csv_file(file_path: Path):
        if not file_path.exists():
            print(f"The file {file_path} does not exist.")
            return
        if file_path.stat().st_size == 0:
            print(f"The file {file_path} is empty.")
            return
        with open(file_path, 'r') as file:
            file_data = file.read()
        file_data = file_data.replace('"""', '"')
        replace_patterns = ['","', '", Jr.,"', '", Sr.,"', '", Jr,"', '", Sr,"', '", IV,"']
        for pattern in replace_patterns:
            file_data = file_data.replace(pattern, ',')
        with open(file_path, 'w') as file:
            file.write(file_data)

    @RateLimiter(max_calls=CONSTANTS['max_requests_per_minute'], period=60)
    def get_data_from_api(self, endpoint: str, file_path: Path):
        try:
            response = requests.get(endpoint)
            response.raise_for_status()
            self.requests_counter += 1
            self.last_request_time = datetime.now()
            csv_content = [row.split(',') for row in response.text.split('\n') if row]
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(csv_content)
        except requests.exceptions.HTTPError as err:
            print(f"HTTP error occurred: {err}")
            self.error_occurred = True
        except requests.exceptions.RequestException as err:
            print(f"An exception occurred: {err}")
            self.error_occurred = True

    def run(self):
        raw_data_directory = CONSTANTS['csv_file_path'] / 'raw_data'
        raw_data_directory.mkdir(parents=True, exist_ok=True)
        for tour in CONSTANTS['tours']:
            for year in CONSTANTS['years']:
                raw_data_file_path = raw_data_directory / f'{tour}_{year}_raw_data.csv'
                if raw_data_file_path.is_file():
                    self.backup_old_files(raw_data_directory, f'{tour}_{year}_raw_data')
                self.get_data_from_api(f"{CONSTANTS['base_url']}/historical-raw-data/rounds?tour={tour}&event_id={CONSTANTS['event_id']}&year={year}&file_format={CONSTANTS['file_format']}&key={CONSTANTS['api_key']}", raw_data_file_path)
                self.sanitize_csv_file(raw_data_file_path)
        for book in CONSTANTS['books']:
            book_directory = CONSTANTS['csv_file_path'] / book
            for tour in CONSTANTS['tours']:
                tour_directory = book_directory / tour
                for year in CONSTANTS['years']:
                    year_directory = tour_directory / str(year)
                    year_directory.mkdir(parents=True, exist_ok=True)
                    outrights_file_path = year_directory / 'outrights.csv'
                    matchups_file_path = year_directory / 'matchups.csv'
                    if outrights_file_path.is_file():
                        self.backup_old_files(year_directory, 'outrights')
                    if matchups_file_path.is_file():
                        self.backup_old_files(year_directory, 'matchups')
                    print(f"Fetching data for book: {book}, tour: {tour}, year: {year}")
                    self.get_data_from_api(f"{CONSTANTS['base_url']}/historical-odds/outrights?tour={tour}&event_id={CONSTANTS['event_id']}&year={year}&market={CONSTANTS['market']}&book={book}&odds_format={CONSTANTS['odds_format']}&file_format={CONSTANTS['file_format']}&key={CONSTANTS['api_key']}", outrights_file_path)
                    self.get_data_from_api(f"{CONSTANTS['base_url']}/historical-odds/matchups?tour={tour}&event_id={CONSTANTS['event_id']}&year={year}&book={book}&odds_format={CONSTANTS['odds_format']}&file_format={CONSTANTS['file_format']}&key={CONSTANTS['api_key']}", matchups_file_path)
                    self.sanitize_csv_file(outrights_file_path)
                    self.sanitize_csv_file(matchups_file_path)

if __name__ == "__main__":
    data_retriever = DataRetriever()
    data_retriever.run()