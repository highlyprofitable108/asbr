import os
import matplotlib.pyplot as plt
from joblib import load
from constants import constants
from features import features
from utils.file_utils import get_most_recent_file
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Directory where the models are saved
model_dir = constants['model_path']

# These are the features that have been used in training the model
trained_features = features

# Define the scaler object
scaler = StandardScaler()


def print_model_details(model_data, trained_features, X_test, y_test):
    # This function prints the details of the model including its parameters,
    # the feature importance scores, and its performance metrics.

    # Prints the raw data of the model
    print("Model data:")
    print(model_data)

    # Prints the parameters of the model
    if hasattr(model_data, 'get_params'):
        print("\nModel parameters:")
        print(model_data.get_params())
    else:
        print("\nThe model doesn't have get_params method.")

    # Prints specific attributes of the model depending on the model type
    if isinstance(model_data, RandomForestRegressor):
        print("\nMax depth:", model_data.max_depth)
        print("Min samples leaf:", model_data.min_samples_leaf)
        print("Number of estimators:", model_data.n_estimators)

    # Prints the coefficients of the model if it has any
    # (applicable for linear models)
    if hasattr(model_data, 'coef_'):
        coefficients = model_data.coef_
        coefficients_dict = {name: coef for name, coef in zip(
            trained_features, coefficients
        )}
        print("\nCoefficients:", coefficients_dict)

    # Prints and plots the feature importance scores if the model has any
    # (applicable for tree-based models)
    if hasattr(model_data, 'feature_importances_'):
        importances = model_data.feature_importances_
        importances_dict = {name: imp for name, imp in zip(
            trained_features, importances
        )}
        print("\nFeature importances:", importances_dict)

        # Plot feature importances
        plt.barh(
            range(len(importances_dict)),
            list(importances_dict.values()),
            align='center'
        )
        plt.yticks(range(len(importances_dict)), list(importances_dict.keys()))
        plt.xlabel('Importance')
        plt.title('Feature importances')
        plt.show()

    # Evaluate model performance on the test set
    if isinstance(model_data, RandomForestRegressor):
        if X_test is not None and y_test is not None:
            y_pred = model_data.predict(X_test)
            print(
                "\nMean Absolute Error:", mean_absolute_error(y_test, y_pred)
            )
        else:
            print("\nRegression evaluation not available.")
    elif isinstance(model_data, RandomForestClassifier):
        if X_test is not None and y_test is not None:
            y_pred = model_data.predict(X_test)
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
        else:
            print("\nClassification evaluation not available.")
    elif isinstance(model_data, MLPRegressor):
        if X_test is not None:
            y_pred = model_data.predict(X_test)
            print(
                "\nMean Absolute Error:", mean_absolute_error(y_test, y_pred)
            )
        else:
            print("\nRegression evaluation not available.")


def main():
    # Define all the different patterns for the model files
    # Add more patterns as needed
    model_patterns = [
        'neural_network',
        'decision_tree',
        'linear_regression',
        'ordinary_least_squares',
        'gradient_boosting',
        'random_forest'
    ]

    for pattern in model_patterns:
        model_file = get_most_recent_file(model_dir, f"{pattern}*.joblib")
        if model_file is not None:
            print(f"Loading model from {model_file}...")

            # Load the trained model from the file
            best_estimator = load(model_file)

            # Extract the date from the filename
            model_date = os.path.basename(
                model_file
            ).split('_')[2].split('.')[0]

            # Load the test data for the model
            test_data_filename = f"{pattern}_{model_date}.joblib"
            test_data_path = os.path.join(model_dir, test_data_filename)

            if pattern == 'neural_network':
                test_data = load(test_data_path)
                X_test = test_data
                y_test = None
            else:
                test_data = load(test_data_path)
                X_test = scaler.fit_transform(test_data['X_test'])
                y_test = test_data['y_test']

            # Prints the details of the model
            print("Model Details:")
            print_model_details(
                best_estimator, trained_features, X_test, y_test

            )
        else:
            print(f"No {pattern} files found in {model_dir}.")


if __name__ == "__main__":
    main()
