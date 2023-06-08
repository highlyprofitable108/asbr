import os
import requests
from pathlib import Path

script_dir = Path(__file__).resolve().parent
root_dir = script_dir.parent

constants = {
    'api_key': '195c3cb68dd9f46d7feaafc4829c',
    'base_url': "https://feeds.datagolf.com",
    'file_path': root_dir / 'data/raw_data/golf/csv',  # absolute path
    'tours': ['pga'],
    'years': list(range(2017, 2023)),
    'file_format': 'csv',
    'event_id': 'all',
    'max_requests_per_minute': 84
}

def get_round_scoring_stats_strokes_gained():
    for tour in constants['tours']:
        for year in constants['years']:
            endpoint = f"{constants['base_url']}/historical-raw-data/rounds?tour={tour}&event_id={constants['event_id']}&year={year}&file_format={constants['file_format']}&key={constants['api_key']}"
            response = requests.get(endpoint)
            if response.status_code == 200:
                data = response.content
                file_path = constants['file_path'] / f'round_scoring_stats_strokes_gained_{tour}_{year}.csv'
                if not file_path.exists() or os.path.getsize(file_path) != len(data):
                    with open(file_path, 'wb') as f:
                        f.write(data)

def get_historical_raw_data_event_ids():
    endpoint = f"{constants['base_url']}/historical-raw-data/event-list?file_format={constants['file_format']}&key={constants['api_key']}"
    response = requests.get(endpoint)
    if response.status_code == 200:
        data = response.content
        file_path = constants['file_path'] / 'historical_raw_data_event_ids.csv'
        if not file_path.exists() or os.path.getsize(file_path) != len(data):
            with open(file_path, 'wb') as f:
                f.write(data)

if __name__ == "__main__":
    get_round_scoring_stats_strokes_gained()
    get_historical_raw_data_event_ids()
