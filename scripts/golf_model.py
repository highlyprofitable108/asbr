import sqlite3
import pandas as pd
from datetime import date
from sklearn.ensemble import RandomForestRegressor
from constants import constants
from features import features
from columns import columns
from utils.file_utils import save_model
from utils.model_utils import train_and_evaluate_model, split_data
from utils.dataframe_utils import handle_missing_values, perform_knn_imputation


def main():
    # Establish connection to the SQLite database using the path provided in constants
    db_conn = sqlite3.connect(constants['database_path'])

    # Create a cursor object which allows SQL commands to be executed
    cursor = db_conn.cursor()
    # Execute SQL command to select all data from the table 'round1_data'
    cursor.execute("""
        SELECT * FROM round1_data;
    """)

    # Fetch all the results from the executed SQL command
    data = cursor.fetchall()

    # If data exists, proceed with the following operations
    if data:
        # Define the column names for the dataframe
        

        # Create a pandas DataFrame using the fetched data and defined column names
        df = pd.DataFrame(data, columns=columns)

        # Call a function from 'utils.dataframe_utils' to perform KNN imputation on the dataframe
        print("Performing KNN imputation...")
        df = perform_knn_imputation(df)

        # Handle missing values in the dataframe
        print("Handling missing values...")
        df = handle_missing_values(df)

        # Prepare the train and test datasets for regression model
        print("Preparing train/test datasets for the regression model...")
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = split_data(
            df,
            [col for col in column_names if col in features],
            'round_score',
            'regression'
        )

        # Define parameters for RandomForestRegressor
        param_grid_regression = {
            'n_estimators': [1000],
            'max_depth': [None],
            'min_samples_split': [2],
            'min_samples_leaf': [1]
        }

        # Train and evaluate RandomForestRegressor model
        print("Training and evaluating the regression model...")
        CV_rf_regression, error_regression, _ = train_and_evaluate_model(
            RandomForestRegressor(random_state=0), 
            param_grid_regression,
            X_train_reg,
            X_test_reg,
            y_train_reg,
            y_test_reg,
            'regression'  # Specify the model type
        )
        
        # Save the trained model to a specified path with current date as part of filename
        model_filename_regression = f"rf_regression_{date.today().strftime('%Y-%m-%d')}"
        save_model(CV_rf_regression, constants['model_path'], model_filename_regression, column_names[:-3])

        # Prepare performance data to be saved for further evaluation
        performance_data = {
            'X_test_reg': X_test_reg,
            'y_test_reg': y_test_reg,
            'error_regression': error_regression
        }

        # Save the performance data to a specified path with current date as part of filename
        performance_filename = f"performance_data_{date.today().strftime('%Y-%m-%d')}"
        save_model(performance_data, constants['model_path'], performance_filename, column_names[:-3])

    # Close the database connection to free resources
    db_conn.close()


if __name__ == "__main__":
    main()
