import os
import fnmatch
from datetime import datetime


def write_to_file(file_path, data):
    """
    Writes data to a file if it doesn't exist or
        if the size of the file is different from the size of the data.

    Parameters:
    file_path (str): The path to the file.
    data (bytes): The data to write to the file.

    """
    if not file_path.exists() or os.path.getsize(file_path) != len(data):
        with open(file_path, 'wb') as f:  # Open the file in binary write mode
            f.write(data)  # Write the data to the file


def generate_file_path(directory, filename):
    """
    Generates a file path with a timestamp attached to the filename.

    Parameters:
    directory (str): The directory of the file.
    filename (str): The name of the file.

    Returns:
    str: The file path with timestamp attached to the filename.
    """
    timestamp = get_timestamp()
    return directory / f'{filename}_{timestamp}.csv'


def get_timestamp():
    """
    Returns the current timestamp in the format: YYYY-MM-DD_HH-MM-SS
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # format the timestamp


def get_most_recent_file(directory, pattern):
    """
    Load the most recent file with a given pattern from a specified directory.

    Parameters:
    directory (str): The directory in which to search for files.
    pattern (str): The pattern to look for.

    Returns:
    str: The most recent file name if any are found, otherwise None.
    """

    # Search and find the necessary files
    files = [file for file in os.listdir(
        directory
    ) if fnmatch.fnmatch(file, pattern)]  # add "*" around pattern
    files.sort(reverse=True)
    recent_file = files[0] if files else None

    # Return full path to the file
    if recent_file:
        return os.path.join(directory, recent_file)
    else:
        print(f"No {pattern} files found.")
        return None
