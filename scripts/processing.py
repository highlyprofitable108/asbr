import sqlite3
import requests
from joblib import dump
from datetime import datetime
import numpy as np
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

def connect_to_database(database_name):
    try:
        conn = sqlite3.connect(database_name)
        print("Database connection successful")
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")

def retrieve_data(connection, table_name):
    try:
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, connection)
        print(f"Data retrieval from {table_name} successful")
        return df
    except Exception as e:
        print(f"Data retrieval error: {e}")

def explore_and_preprocess_data(df):
    df_copy = df.copy()

    # Check and filter round_score column
    df_copy['round_score'] = pd.to_numeric(df_copy['round_score'], errors='coerce').fillna(-999)
    df_copy = df_copy[(df_copy['round_score'] >= 50) & (df_copy['round_score'] <= 90) | (df_copy['round_score'] == -999)]
    df_copy['avg_round_score'] = df_copy.groupby('player_name')['round_score'].transform('mean')

    # Check and filter sg columns
    sg_columns = ['sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g']
    for column in sg_columns:
        df_copy[column] = pd.to_numeric(df_copy[column], errors='coerce').fillna(-999).round(3)
        df_copy = df_copy[(df_copy[column] >= -10) & (df_copy[column] <= 10) | (df_copy[column] == -999)]

    # Check percentage columns
    percentage_columns = ['driving_acc', 'gir', 'scrambling']
    for column in percentage_columns:
        df_copy[column] = pd.to_numeric(df_copy[column], errors='coerce').fillna(-999)
        df_copy = df_copy[(df_copy[column] >= 0) & (df_copy[column] <= 100) | (df_copy[column] == -999)]
    
    # Check course_par column
    df_copy['course_par'] = pd.to_numeric(df_copy['course_par'], errors='coerce').fillna(-999)
    df_copy = df_copy[(df_copy['course_par'] >= 69) & (df_copy['course_par'] <= 75) | (df_copy['course_par'] == -999)]
    
    # Calculate player progression
    df_copy['player_progression'] = df_copy.groupby('player_name')['avg_round_score'].transform(lambda x: x.pct_change())

    # Handle categorical data for course_name
    df_copy['course_name'] = df_copy['course_name'].astype('category').cat.codes
    
    # Handle fin_text column
    df_copy['fin_text'] = df_copy['fin_text'].str.lstrip('T')
    df_copy['fin_text'] = pd.to_numeric(df_copy['fin_text'], errors='coerce').fillna(999).astype(int)

    # Normalize selected features
    scaler = MinMaxScaler()
    columns_to_scale = ['avg_round_score', 'driving_acc', 'gir', 'scrambling', 'course_par', 'player_progression']
    df_copy[columns_to_scale] = scaler.fit_transform(df_copy[columns_to_scale])

    print("Data exploration and preprocessing complete")  # Debugging line

    return df_copy

def dimensionality_reduction(df):
    print("Starting dimensionality reduction")  # Debugging line
    
    # Select the relevant features for PCA
    features = df.drop(['sg_total'], axis=1)

    # Replace missing or infinite values with a large negative number
    features = features.replace([np.inf, -np.inf], np.nan)
    features.fillna(-999, inplace=True)

    # Perform PCA
    pca = PCA(n_components=7)  # Adjust the number of components as needed
    features_reduced = pca.fit_transform(features)
    
    # Create a DataFrame with the reduced features
    df_reduced = pd.DataFrame(features_reduced, columns=[f'pc_{i+1}' for i in range(pca.n_components_)])
    
    # Concatenate the reduced features with the target variable
    df_processed = pd.concat([df[['fin_text']], df_reduced], axis=1)
    
    print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    print(f"Cumulative Explained Variance: {cumulative_variance}")

    plt.plot(pca.explained_variance_ratio_)
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)') # for each component
    plt.title('Explained Variance')
    plt.show()

    print(df_processed.head())

    print("Dimensionality reduction complete")  # Debugging line
    return df_processed

def feature_engineering(df):
    print("Feature Engineering starting")  # Debugging line

    # Define the columns that should contain numeric values
    numeric_cols = ['round_score', 'sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g', 'sg_total', 'year', 'season', 
                    'driving_dist', 'driving_acc', 'gir', 'scrambling', 'prox_rgh', 'prox_fw']

    string_cols = [col for col in df.columns if col not in numeric_cols]

    # Attempt to convert all relevant columns to numeric data types and average them
    for col in numeric_cols:
        df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
        df.loc[:, f'average_{col}'] = df.groupby('player_name')[col].transform('mean')

    # Processing for string columns can be added here
    # for col in string_cols:
        # your processing here

    print("Feature Engineering complete")  # Debugging line
    return df

def create_model(optimizer='adam'):
    print("Create Model")  # Debugging line
    model = Sequential()
    model.add(Dense(12, input_dim=7, activation='relu'))  # Adjusted the input_dim to match PCA components
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def algorithm_development(df):
    X = df.drop(['fin_text'], axis=1)
    y = df['fin_text']

    # Define preprocessing for numeric columns (scale them)
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    # Define preprocessing for categorical features (encode them)
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # Create a pipeline that combines the preprocessor with the estimator
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', LogisticRegression(solver='liblinear'))])

    param_grid = {'classifier__C': [0.1, 1, 10, 100], 'classifier__penalty': ['l1', 'l2']}

    stratified_k_fold = StratifiedKFold(n_splits=2)

    grid = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=stratified_k_fold, verbose=2)
    grid_result = grid.fit(X, y)

    # Save the model as a pickle in a file with the timestamp in the name
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    dump(grid_result.best_estimator_, f'/Users/michaelfuscoletti/Desktop/data/best_estimator_{timestamp}.joblib')

    print("Modeling complete")  # Debugging line

    return grid_result.best_estimator_

def get_player_data(tour='pga', file_format='csv'):
    url = f"https://feeds.datagolf.com/field-updates?tour={tour}&file_format={file_format}&key=195c3cb68dd9f46d7feaafc4829c"
    response = requests.get(url)
    response.raise_for_status()
    if file_format.lower() == 'csv':
        data = pd.read_csv(StringIO(response.text))
    return data

def predict_future_tournaments(players_list, df_historical, model):
    df_future = df_historical[df_historical['player_name'].isin(players_list)]
    df_future_fe = feature_engineering(df_future)
    df_future_fe = df_future_fe.drop(['fin_text'], axis=1)
    scaler = StandardScaler()
    df_future_fe = scaler.fit_transform(df_future_fe)
    predictions = model.predict(df_future_fe)
    return predictions

def evaluate_predictions(predictions, actual_results):
    mae = mean_absolute_error(actual_results, predictions)
    mse = mean_squared_error(actual_results, predictions)
    r2 = r2_score(actual_results, predictions)
    print(f'MAE: {mae}, MSE: {mse}, R2: {r2}')

def main():
    try:
        conn = connect_to_database('/Users/michaelfuscoletti/Desktop/data/pgatour.db')  # Replace 'your_database_name' with the name of your database
        df_historical = retrieve_data(conn, 'raw_data')  # Replace 'your_table_name' with the name of your table
        df_historical = explore_and_preprocess_data(df_historical)
        df_historical_fe = feature_engineering(df_historical)
        model = algorithm_development(df_historical_fe)
        players_list = get_player_data(tour='pga', file_format='csv')['player_name'].tolist()
        predictions = predict_future_tournaments(players_list, df_historical, model)
        evaluate_predictions(predictions, df_historical['fin_text'])
        print("Execution completed successfully")
    except Exception as e:
        print(f"Execution error: {e}")

if __name__ == "__main__":
    main()