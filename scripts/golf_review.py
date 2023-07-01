import os
import matplotlib.pyplot as plt
from datetime import date
from joblib import load
from constants import constants
from features import features
from utils.file_utils import get_most_recent_file
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Directory where the models are saved
model_dir = constants['model_path']

# These are the features that have been used in training the model
trained_features = [features]

"""
    "minimum_temperature",
    "maximum_temperature",
    "temperature",
    "precipitation_amount",
    "wind_gust",
    "wind_speed",
"""


def print_model_details(model_data, trained_features, X_test, y_test):
    # This function prints the details of the model including its parameters,
    # the feature importance scores, and its performance metrics.

    # Unpack the model and player data from the dictionary
    best_estimator = model_data.best_estimator_

    # Prints the raw data of the model
    print("Model data:")
    print(best_estimator)

    # Prints the parameters of the model
    print("\nModel parameters:")
    print(best_estimator.get_params())

    # Prints specific attributes of the model depending on the model type
    if isinstance(best_estimator, RandomForestClassifier) or isinstance(best_estimator, RandomForestRegressor):
        print("\nMax depth:", best_estimator.max_depth)
        print("Min samples leaf:", best_estimator.min_samples_leaf)
        print("Number of estimators:", best_estimator.n_estimators)

    # Prints the coefficients of the model if it has any (applicable for linear models)
    if hasattr(best_estimator, 'coef_'):
        coefficients = best_estimator.coef_
        coefficients_dict = {name: coef for name, coef in zip(trained_features, coefficients)}
        print("\nCoefficients:", coefficients_dict)

    # Prints and plots the feature importance scores if the model has any (applicable for tree-based models)
    if hasattr(best_estimator, 'feature_importances_'):
        importances = best_estimator.feature_importances_
        importances_dict = {name: imp for name, imp in zip(trained_features, importances)}
        print("\nFeature importances:", importances_dict)
        
        # Plot feature importances
        plt.barh(range(len(importances_dict)), list(importances_dict.values()), align='center')
        plt.yticks(range(len(importances_dict)), list(importances_dict.keys()))
        plt.xlabel('Importance')
        plt.title('Feature importances')
        plt.show()

    # Evaluate model performance on the test set
    if isinstance(best_estimator, RandomForestClassifier):
        y_pred = best_estimator.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    elif isinstance(best_estimator, RandomForestRegressor):
        if X_test is not None and y_test is not None:
            y_pred = best_estimator.predict(X_test)
            print("\nMean Absolute Error:", mean_absolute_error(y_test, y_pred))
        else:
            print("\nRegression evaluation not available.")


def main():
    # Load the most recent regression model
    recent_model_files_regression = get_most_recent_file(model_dir, 'rf_regression_*.joblib')

    if recent_model_files_regression:
        model_filename_regression = recent_model_files_regression
        model_path_regression = os.path.join(model_dir, model_filename_regression)

        # Load the trained model from the file
        best_estimator = load(model_path_regression)

        # Load the test data for the model
        test_data_filename = f"performance_data_{date.today().strftime('%Y-%m-%d')}.joblib"
        test_data_path = os.path.join(model_dir, test_data_filename)
        test_data = load(test_data_path)

        # Apply scaling to the test data
        scaler = StandardScaler()
        X_test_reg = scaler.fit_transform(test_data['X_test_reg'])
        y_test_reg = test_data['y_test_reg']

        # Prints the details of the regression model
        print("Regression Model Details:")
        print_model_details(best_estimator, trained_features, X_test_reg, y_test_reg)
    else:
        print(f"No rf_regression_*.rf files found in {model_dir}.")


if __name__ == "__main__":
    main()