import requests
from constants import constants
import urllib
import csv
from io import StringIO
from opencage.geocoder import OpenCageGeocode


def make_api_call(url):
    """
    Makes a GET request to a given URL.
    Returns the response content if the status code is 200,
        otherwise raises an Exception.

    Args:
        url (str): The URL to which the GET request will be sent.

    Returns:
        response.content (bytes): The content of the response.

    Raises:
        Exception: If the status code of the response is not 200.
    """
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(
            f'GET request failed with status {response.status_code}'
        )


def get_location_data(course_name):
    """
    Gets the latitude and longitude of a location given its name.

    Args:
        course_name (str): The name of the location.

    Returns:
        lat (float), lon (float): The latitude and longitude of the location.
    """
    # Retrieve API key from constants
    key = constants['location_key']
    geocoder = OpenCageGeocode(key)

    results = geocoder.geocode(course_name)

    if results and len(results):
        latitude = results[0]['geometry']['lat']
        longitude = results[0]['geometry']['lng']

        return latitude, longitude
    return None, None


def get_elevation_data(lat, lon):
    """
    Gets the elevation of a location given its latitude and longitude.

    Args:
        lat (float): The latitude of the location.
        lon (float): The longitude of the location.

    Returns:
        elevation (float): The elevation of the location in meters.
    """
    # Build the URL query string
    query = urllib.parse.urlencode({
        "output": "json",
        "x": lon,
        "y": lat,
        "units": "Meters"
    })
    url = f"https://nationalmap.gov/epqs/pqs.php?{query}"

    response = requests.get(url)

    data = response.json()

    return data['USGS_Elevation_Point_Query_Service']['Elevation']


def get_weather_data(date, lat, lon):
    """
    Gets weather data for a location on a specific date.

    Args:
        date (datetime): The date for which to get weather data.
        lat (float): The latitude of the location.
        lon (float): The longitude of the location.

    Returns:
        weather_data (dict): A dictionary containing various weather data.
    """
    # Retrieve API key from constants
    weather_key = constants['weather_key']

    # Format the date as YYYY-MM-DD
    date_str = date.strftime('%Y-%m-%d')

    # Create the API request URL
    url = (
        f"https://weather.visualcrossing.com/VisualCrossingWebServices/"
        f"rest/services/timeline/{lat}%2C{lon}/{date_str}/{date_str}?"
        f"unitGroup=us&include=days&key={weather_key}&contentType=csv"
    )

    # Make a GET request to the API
    response = make_api_call(url)

    # Parse the CSV response
    reader = csv.reader(StringIO(response.decode('utf-8')), delimiter=',')
    next(reader)  # skip the headers
    data = next(reader)

    weather_data = {
        'min_temp': float(data[3]),
        'max_temp': float(data[2]),
        'temperature': float(data[4]),
        'precipitation': float(data[10]),
        'wind_gust': float(data[16]) if data[16] != '' else float(data[17]),
        'wind_speed': float(data[17]),
    }

    return weather_data
