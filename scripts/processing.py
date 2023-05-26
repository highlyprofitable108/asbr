import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Connect to SQLite database
database_path = '/Users/michaelfuscoletti/Desktop/data/pgatour.db'
conn = sqlite3.connect(database_path)
cursor = conn.cursor()

def train_model(data, target_column):
    # Separate target variable from the rest of the data
    y = data[target_column]
    X = data.drop(target_column, axis=1)

    # Add 'round_score_std' as a feature
    X = X.join(data['round_score_std'])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline with StandardScaler and Ridge regression
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge())
    ])

    # Define the hyperparameters and their possible values for fine-tuning
    params = {
        'ridge__alpha': [1e-3, 1e-2, 1e-1, 1, 10, 100]
    }

    # Perform a grid search with cross-validation to find the best hyperparameters
    grid_search = GridSearchCV(model, params, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Train the model with the best hyperparameters on the entire training set
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'Mean Absolute Error: {mae:.2f}')

    return best_model

def main():
    # Load data from the database
    query = "SELECT * FROM raw_data"
    data = pd.read_sql_query(query, conn)
    
    # Preprocess the data
    data = preprocess_data(data)
    
    # Train the model
    target_column = 'round_score_sum'
    model = train_model(data, target_column)

    # Close the database connection
    conn.close()

if __name__ == "__main__":
    main()
