import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from constants import constants


def load_data():
    """
    Load data from the SQLite database.
    
    Returns:
    DataFrame: The loaded data.
    """

    # Connect to the database
    db_conn = sqlite3.connect(constants['database_path'])

    # Retrieve the necessary data from the database
    df = pd.read_sql_query("SELECT * FROM round1_data", db_conn)

    # Close the database connection
    db_conn.close()

    return df


def convert_dtypes(df):
    """
    Convert the data types of the numeric columns.

    Parameters:
    df (DataFrame): The data.

    Returns:
    DataFrame: The data with converted data types.
    """

    numeric_columns = constants['numerical_columns']

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def plot_categorical(df, column):
    """
    Plot the distribution of a categorical variable.

    Parameters:
    df (DataFrame): The data.
    column (str): The column to plot.
    """

    plt.figure(figsize=(10, 6))
    sns.countplot(y=column, data=df, order=df[column].value_counts().index)
    plt.title(f"Distribution of {column}")
    plt.show()


def plot_numerical(df, column):
    """
    Plot the distribution of a numerical variable.

    Parameters:
    df (DataFrame): The data.
    column (str): The column to plot.
    """

    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.show()


def plot_time_series(df, x_column, y_column):
    """
    Plot a time series.

    Parameters:
    df (DataFrame): The data.
    x_column (str): The x column.
    y_column (str): The y column.
    """

    plt.figure(figsize=(12, 6))
    df = df.reset_index()  # Reset the index to avoid duplicate labels
    sns.lineplot(x=x_column, y=y_column, data=df)
    plt.title(f"Time Series of {y_column}")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()


def print_top_correlations(correlations, n):
    """
    Print the top n positive correlations with 'round_score'.

    Parameters:
    correlations (Series): The correlations.
    n (int): The number of correlations to print.
    """

    print(f"Top {n} positive correlations with 'round_score':")
    for feature, correlation in correlations[:n].items():
        print(f"{feature}: {correlation:.4f}")


def print_bottom_correlations(correlations, n):
    """
    Print the top n negative correlations with 'round_score'.

    Parameters:
    correlations (Series): The correlations.
    n (int): The number of correlations to print.
    """

    print(f"Top {n} negative correlations with 'round_score':")
    for feature, correlation in correlations[-n:].items():
        print(f"{feature}: {correlation:.4f}")


def main():
    """
    The main function that executes the data analysis process.
    """

    # Load data
    df = load_data()

    # Print first 5 rows
    print(df.head())

    # Print info
    print(df.info())

    # Print description of numeric data
    print(df.describe(include=[np.number]))

    # Convert columns to appropriate data types
    df = convert_dtypes(df)

    # Print description of non-numeric data
    print(df.describe(include=['object']))

    # Plot distributions of some numerical columns
    # for column in constants['numerical_columns']:
    #    plot_numerical(df, column)

    # Plot time series of round score
    # plot_time_series(df, 'date', 'round_score')

    # Calculate correlations with round_score
    numeric_df = df.select_dtypes(include=[np.number])
    correlations = numeric_df.corr()['round_score'].drop('round_score')

    # Sort correlations in descending order
    sorted_correlations = correlations.sort_values(ascending=False)

    # Print top 15 positive and top 15 negative correlations with round_score
    print_top_correlations(sorted_correlations, 15)
    print_bottom_correlations(sorted_correlations, 15)


if __name__ == "__main__":
    main()
