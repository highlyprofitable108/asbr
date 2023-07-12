import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from typing import List
from pathlib import Path


def read_and_concatenate(directory: Path, file_list: List[str]):
    """
    Read and concatenate data from multiple CSV files into a single dataframe.

    Args:
        directory (Path): Path of the directory where the CSV files are stored.
        file_list (List[str]): List of CSV file names.

    Returns:
        pd.DataFrame: Concatenated dataframe.
    """
    df = pd.DataFrame()

    # Format path and read file
    #   at location
    for file in file_list:
        filepath = directory / file
        df_temp = pd.read_csv(filepath)
        df = pd.concat([df, df_temp])
    return df


def handle_missing_values(df: pd.DataFrame):
    """
    Handle missing values in a dataframe. For numeric columns,
        rows with missing values are dropped.
    For categorical columns, missing values are replaced with 'Unknown'.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with handled missing values.
    """
    # Get columns with numeric data type
    numeric_columns = df.select_dtypes(include=np.number).columns
    # Drop rows where any of the numeric columns has missing values
    df.dropna(subset=numeric_columns, inplace=True)

    # Get columns with object data type (usually string/categorical)
    categorical_columns = df.select_dtypes(include=['object']).columns
    # Replace missing values in categorical columns with 'Unknown'
    df[categorical_columns] = df[categorical_columns].fillna('Unknown')

    return df


def perform_one_hot_encoding(df: pd.DataFrame, columns: List[str]):
    """
    Perform one-hot encoding on specified columns of a dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.
        columns (List[str]): Columns to perform one-hot encoding on.

    Returns:
        pd.DataFrame: Dataframe with one-hot encoded columns.
    """
    # pd.get_dummies function is used for one-hot encoding
    df = pd.get_dummies(df, columns=columns)
    return df


def convert_to_datetime(df: pd.DataFrame, columns: List[str]):
    """
    Convert specified columns of a dataframe to datetime format.

    Args:
        df (pd.DataFrame): Input dataframe.
        columns (List[str]): Columns to convert to datetime format.

    Returns:
        pd.DataFrame:
            Dataframe with specified columns converted to datetime format.
    """
    # Convert the column to datetime format
    for column in columns:
        df[column] = pd.to_datetime(df[column])
    return df


def perform_knn_imputation(df: pd.DataFrame):
    """
    Perform K-Nearest Neighbors (KNN) imputation on a dataframe.
    The imputation is performed only on columns with numeric values.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with imputed values.
    """
    # Identify columns with missing values
    columns_with_blanks = df.columns[df.isnull().any()].tolist()

    # Replace blank values with NaN
    df.replace('', np.nan, inplace=True)

    # Perform KNN imputation only on columns with numeric values
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    columns_to_impute = list(set(numeric_columns) & set(columns_with_blanks))

    # Create a copy of the DataFrame for imputation
    df_imputed = df.copy()

    # Impute missing values using KNN imputation
    for column in columns_to_impute:
        imputer = KNNImputer(n_neighbors=5, weights='uniform')
        df_imputed[column] = imputer.fit_transform(df_imputed[[column]])

    return df_imputed
