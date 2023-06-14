import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Define the directory of the raw data files
directory = Path('/Users/michaelfuscoletti/Desktop/asbr/data/golf/csv')

historical_files = []
other_files = []

# Segregate files into historical and other files
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        if 'historical' in filename:
            historical_files.append(filename)
        else:
            other_files.append(filename)

print(f"Found {len(historical_files)} historical files and {len(other_files)} other files")

# Sort the files by the most recent date stamp
historical_files.sort(reverse=True)
other_files.sort(reverse=True)

# Use only the most recent file
historical_files = [historical_files[0]] if historical_files else []
other_files = [other_files[0]] if other_files else []

print(f"Processing {len(historical_files)} historical file and {len(other_files)} other file")

# Initialize an empty DataFrame for historical files
df_historical = pd.DataFrame()

# Read and process historical files
for file in historical_files:
    filepath = directory / file
    print(f"Processing historical file: {file}")
    df_temp = pd.read_csv(filepath)  # Read csv as pandas DataFrame
    df_historical = pd.concat([df_historical, df_temp])  # Concatenate with the main DataFrame

# Initialize a list for other files
other_dfs = []

# Read and process other files
for file in other_files:
    filepath = directory / file
    print(f"Processing other file: {file}")
    df_temp = pd.read_csv(filepath)  # Read csv as pandas DataFrame
    other_dfs.append(df_temp)  # Append DataFrame to the list

# Concatenate all DataFrames in the list into a single DataFrame
df_other = pd.concat(other_dfs)

# Handle missing values for numerical columns
print("Handling missing values for numerical columns in the other DataFrame...")
numeric_columns = df_other.select_dtypes(include=np.number).columns
df_other[numeric_columns] = df_other[numeric_columns].fillna(df_other[numeric_columns].mean())

# Handle missing values for categorical columns (optional)
print("Handling missing values for categorical columns in the other DataFrame...")
categorical_columns = df_other.select_dtypes(include=['object']).columns
df_other[categorical_columns] = df_other[categorical_columns].fillna('Unknown')

# Convert categorical values into numerical ones
print("Converting categorical values into numerical ones for historical DataFrame...")
df_historical['sg_categories'] = df_historical['sg_categories'].map({'yes': 1, 'no': 0})
df_historical['traditional_stats'] = df_historical['traditional_stats'].map({'yes': 1, 'no': 0})

# One-hot encoding for 'tour' column
print("Performing one-hot encoding for 'tour' column for both DataFrames...")
df_historical = pd.get_dummies(df_historical, columns=['tour'])
df_other = pd.get_dummies(df_other, columns=['tour'])

# Convert date column to datetime type
print("Converting date column to datetime type for both DataFrames...")
df_historical['date'] = pd.to_datetime(df_historical['date'])
df_other['event_completed'] = pd.to_datetime(df_other['event_completed'])

# Define the output directory
output_dir = Path('/Users/michaelfuscoletti/Desktop/asbr/data/golf/pkl')

# Create a timestamp for the current date and time
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Save the dataframes to pickle files with the timestamp
print("Saving dataframes to pickle files...")
df_historical.to_pickle(output_dir / f'df_historical_{timestamp}.pkl')
df_other.to_pickle(output_dir / f'df_other_{timestamp}.pkl')
print("All done!")
