import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from features import features
from constants import constants


def load_and_process_data():
    """
    Load data from the SQLite database and process it.

    Returns:
    DataFrame: The loaded data.
    """
    conditions = [f"{col}" for col in features]
    where_clause = " AND ".join(conditions)
    where_clause += " AND " + " AND ".join(
        [f"{col} IS NOT NULL" for col in features]
    )

    with sqlite3.connect(constants['database_path']) as db_conn:
        df = pd.read_sql_query(
            f"SELECT * FROM model_data WHERE {where_clause}", db_conn
        )
        for col in df.columns:
            if col in features + [constants['target_variable']]:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def plot_numerical(df, column):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.show()


def plot_time_series(df, x_column, y_column):
    plt.figure(figsize=(12, 6))
    df = df.reset_index()
    sns.lineplot(x=x_column, y=y_column, data=df)
    plt.title(f"Time Series of {y_column}")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()


def print_correlations(correlations, n):
    print("Correlations with 'round_score':")
    for feature, correlation in correlations[:n].items():
        print(f"{feature}: {correlation:.4f}")


def main():
    df = load_and_process_data()
    print(df.head())
    print(df.info())
    print(df.describe())

    # for column in features:
    #    plot_numerical(df, column)

    # plot_time_series(df, 'date', constants['target_variable'])

    numeric_df = df.select_dtypes(include=[np.number])
    correlations = numeric_df.corr()[constants['target_variable']].drop(
        constants['target_variable']
    )
    sorted_correlations = correlations.abs().sort_values(ascending=False)

    print_correlations(sorted_correlations, 100)


if __name__ == "__main__":
    main()
