import requests
from time import sleep
import pandas as pd
from io import StringIO

api_key = "195c3cb68dd9f46d7feaafc4829c"
base_url = "https://feeds.datagolf.com"

tours = ['pga', 'euro', 'kft', 'alt']
years = range(2017, 2023)
market = 'win'
books = ['betmgm', 'draftkings', 'Fanduel']
odds_format = 'american'
file_format = 'csv'

# Get event IDs
def get_event_ids(tour):
    endpoint = f"{base_url}/historical-odds/event-list?tour={tour}&file_format={file_format}&key={api_key}"
    response = requests.get(endpoint)
    if response.status_code == 200:
        data = StringIO(response.text)
        df = pd.read_csv(data)
        return df['event_id'].unique()
    else:
        return []

def fetch_data():
    with open('event_ids.csv', 'w') as event_ids_file, open('odds_data.csv', 'w') as odds_data_file, open('raw_data.csv', 'w') as raw_data_file:
        for tour in tours:
            events = get_event_ids(tour)
            event_ids_file.write(f"{tour},{','.join(map(str, events))}\n")

            for event in events:
                for year in years:
                    for book in books:
                        sleep(1)  # To avoid overloading the server

                        # Get Outrights Data
                        endpoint = f"{base_url}/historical-odds/outrights?tour={tour}&event_id={event}&year={year}&market={market}&book={book}&odds_format={odds_format}&file_format={file_format}&key={api_key}"
                        response = requests.get(endpoint)
                        if response.status_code == 200:
                            data = response.text
                            odds_data_file.write(f"{tour},{event},{year},{book},outright,{data}\n")

                        # Get Matchups Data
                        endpoint = f"{base_url}/historical-odds/matchups?tour={tour}&event_id={event}&year={year}&book={book}&odds_format={odds_format}&file_format={file_format}&key={api_key}"
                        response = requests.get(endpoint)
                        if response.status_code == 200:
                            data = response.text
                            odds_data_file.write(f"{tour},{event},{year},{book},matchup,{data}\n")

                        # Get Rounds Data
                        endpoint = f"{base_url}/historical-raw-data/rounds?tour={tour}&event_id={event}&year={year}&file_format={file_format}&key={api_key}"
                        response = requests.get(endpoint)
                        if response.status_code == 200:
                            data = response.text
                            raw_data_file.write(f"{tour},{event},{year},{book},{data}\n")

fetch_data()
