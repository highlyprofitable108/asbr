import os
import pandas as pd
import numpy as np

directory = '/Users/michaelfuscoletti/Desktop/data/raw_data/golf/csv'

historical_files = []
other_files = []

# segregate files into historical and other files
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        if 'historical' in filename:
            historical_files.append(filename)
        else:
            other_files.append(filename)

# initialize an empty DataFrame for historical files
df_historical = pd.DataFrame()

# read and process historical files
for file in historical_files:
    filepath = os.path.join(directory, file)
    df_temp = pd.read_csv(filepath)  # read csv as pandas DataFrame
    df_historical = pd.concat([df_historical, df_temp])  # concatenate with the main DataFrame 

# initialize a list for other files
other_dfs = []

# read and process other files
for file in other_files:
    filepath = os.path.join(directory, file)
    df_temp = pd.read_csv(filepath)  # read csv as pandas DataFrame
    other_dfs.append(df_temp)  # append DataFrame to the list

# concatenate all DataFrames in the list into a single DataFrame
df_other = pd.concat(other_dfs)

# Handle missing values
# for simplicity, we are going to fill NaNs with the mean of the respective column for numerical columns
df_other = df_other.fillna(df_other.mean())

# Convert categorical values into numerical ones
df_historical['sg_categories'] = df_historical['sg_categories'].map({'yes': 1, 'no': 0})
df_historical['traditional_stats'] = df_historical['traditional_stats'].map({'yes': 1, 'no': 0})

# One-hot encoding for 'tour' column
df_historical = pd.get_dummies(df_historical, columns=['tour'])
df_other = pd.get_dummies(df_other, columns=['tour'])

# Convert date column to datetime type
df_historical['date'] = pd.to_datetime(df_historical['date'])
df_other['event_completed'] = pd.to_datetime(df_other['event_completed'])

# Save the dataframes to pickle files
df_historical.to_pickle('/Users/michaelfuscoletti/Desktop/data/processed_data/golf/pkl/df_historical.pkl')
df_other.to_pickle('/Users/michaelfuscoletti/Desktop/data/processed_data/golf/pkl/df_other.pkl')
