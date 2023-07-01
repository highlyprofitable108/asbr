import os
import fnmatch
from joblib import dump
from datetime import datetime


def write_to_file(file_path, data):
    """
    Writes data to a file if it doesn't exist or if the size of the file is different from the size of the data.

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
    timestamp = get_timestamp()  # get the current timestamp
    return directory / f'{filename}_{timestamp}.csv'  # append the timestamp to the filename


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
    files = [file for file in os.listdir(directory) if fnmatch.fnmatch(file, pattern)]  # Get all files that match the pattern
    files.sort(reverse=True)  # Sort the files in reverse order (most recent first)
    recent_file = files[0] if files else None  # Get the most recent file

    if recent_file:
        return os.path.join(directory, recent_file)  # Return full path to the file
    else:
        print(f"No {pattern} files found.")
        return None
    

def save_model(model, model_path, model_filename, feature_names):
    """
    Saves the trained model to a Joblib file and the feature names to a text file.

    Parameters:
    model (object): The trained model.
    model_path (str): The path where the model will be saved.
    model_filename (str): The filename of the model.
    feature_names (list): The names of the features used to train the model.
    """
    # Save the model to a Joblib file
    dump(model, os.path.join(model_path, model_filename + '.joblib'))

    # Save the feature names to a text file
    with open(os.path.join(model_path, model_filename + '_features.txt'), 'w') as f:
        f.write('\n'.join(feature_names))  # write the feature names line by line
