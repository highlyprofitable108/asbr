import sqlite3
import requests
import numpy as np
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils import resample
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Database connection function
def connect_to_database(database_name):
    try:
        conn = sqlite3.connect(database_name)
        print("Database connection successful.")  # Debugging line
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")  # Debugging line

# Data retrieval function
def retrieve_data(connection, table_name):
    try:
        query = f"SELECT * FROM {table_name}"
        cursor = connection.cursor()
        cursor.execute(query)
        data = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(data, columns=columns)
        print(f"Data retrieval from professional tours successful.")  # Debugging line
        return df
    except Exception as e:
        print(f"Data retrieval error: {e}")  # Debugging line

# Data exploration and preprocessing function
def explore_and_preprocess_data(df):
    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    df_copy = df.copy()

    # Check and filter round_score column
    df_copy.loc[:, 'round_score'] = pd.to_numeric(df_copy['round_score'], errors='coerce').fillna(-999)
    df_copy = df_copy[(df_copy['round_score'] >= 50) & (df_copy['round_score'] <= 90) | (df_copy['round_score'] == -999)]

    # Check and filter sg columns
    sg_columns = ['sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g', 'sg_total']
    for column in sg_columns:
        df_copy.loc[:, column] = pd.to_numeric(df_copy[column], errors='coerce').fillna(-999).round(3)
        df_copy = df_copy[(df_copy[column] >= -10) & (df_copy[column] <= 10) | (df_copy[column] == -999)]

    # Check round_num column
    df_copy.loc[:, 'round_num'] = pd.to_numeric(df_copy['round_num'], errors='coerce').fillna(-999)
    df_copy = df_copy[(df_copy['round_num'] >= 1) & (df_copy['round_num'] <= 4) | (df_copy['round_num'] == -999)]

    # Check course_par column
    df_copy.loc[:, 'course_par'] = pd.to_numeric(df_copy['course_par'], errors='coerce').fillna(-999)
    df_copy = df_copy[(df_copy['course_par'] >= 69) & (df_copy['course_par'] <= 75) | (df_copy['course_par'] == -999)]

    # Check percentage columns
    percentage_columns = ['gir', 'driving_acc', 'scrambling']
    for column in percentage_columns:
        df_copy.loc[:, column] = pd.to_numeric(df_copy[column], errors='coerce').fillna(-999)
        df_copy = df_copy[(df_copy[column] >= 0) & (df_copy[column] <= 100) | (df_copy[column] == -999)]

    # Normalize sg data relative to field strength
    sg_columns_relative = ['sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g', 'sg_total']
    for column in sg_columns_relative:
        df_copy[column] = pd.to_numeric(df_copy[column], errors='coerce')
        df_copy[column].fillna(-999, inplace=True)
        df_copy[column] = df_copy.groupby('event_id')[column].transform(lambda x: MinMaxScaler().fit_transform(x.values.reshape(-1, 1)).flatten())

    # Selecting features based on chi-square test for round scores
    round_scores = df_copy[['round_score']]
    round_scores = round_scores.astype(int)  # Ensure round scores are integer values
    round_scores = round_scores.apply(lambda x: pd.qcut(x, q=4, labels=False, duplicates='drop'))
    round_scores = round_scores.astype(str)  # Convert back to string type for chi-square test
    round_scores = round_scores.apply(lambda x: x.replace('NaN', 'Missing'))  # Handle missing values

    chi2_selector = SelectKBest(chi2, k='all')

    # Remove leading 'T' from the 'fin_text' column
    df_copy['fin_text'] = df_copy['fin_text'].str.lstrip('T')

    # Replace non-numeric values with NaN and then with '999'
    df_copy['fin_text'] = pd.to_numeric(df_copy['fin_text'], errors='coerce').fillna(999)

    # Convert 'fin_text' column to int
    df_copy['fin_text'] = df_copy['fin_text'].astype(int)

    round_scores_selected = chi2_selector.fit_transform(round_scores, df_copy['fin_text'])
    round_scores_selected = pd.DataFrame(round_scores_selected, columns=['round_score'])

    # Normalize tour and course_name columns if needed
    df_copy['tour'] = df_copy['tour'].str.strip()  # Remove leading/trailing spaces

    # Applying ridge regression for feature selection on golf course information
    course_info = df_copy[['course_name', 'course_par', 'course_num']]
    course_info_encoded = pd.get_dummies(course_info, drop_first=True)
    course_info_encoded_normalized = StandardScaler().fit_transform(course_info_encoded)

    ridge_selector = Ridge(alpha=1.0)
    ridge_selector.fit(course_info_encoded_normalized, df_copy['fin_text'])

    # Concatenating selected features back to the original dataframe
    df_selected = pd.concat([df_copy[['fin_text']], round_scores_selected, course_info_encoded], axis=1)
    
    # Step 3.5: Dimensionality Reduction
    df_preprocessed = dimensionality_reduction(df_selected)

    # Step 3.4: Encoding Categorical Variables
    # categorical_columns = ['tour', 'year', 'season', 'event_name', 'round_num']

    # One-Hot Encoding with column dropping
    # df_encoded = pd.get_dummies(df_selected, columns=categorical_columns, drop_first=True)

    # Handling missing values in categorical variables
    # df_encoded.fillna('Missing', inplace=True)

    # Concatenating the encoded categorical variables back to the dataframe
    # df_preprocessed = pd.concat([df_encoded, df_selected.drop(categorical_columns, axis=1)], axis=1)

    # Step 3.6: Data Sampling
    # Placeholder for data sampling techniques
    # Add your code here for handling imbalanced datasets if needed

    # Step 3.7: Handling Skewed Data
    # Placeholder for handling skewed data
    # Add your code here for handling skewed features

    # Step 3.8: Outlier Detection and Treatment
    # Placeholder for outlier detection and treatment
    # Add your code here for outlier detection and treatment techniques

    # Step 3.9: Optimal Simulation
    # Placeholder for optimal simulation
    # Add your code here for optimal simulation techniques

    # Step 3.10: Mutual Information Feature Selection
    # Placeholder for mutual information feature selection
    # Add your code here for mutual information feature selection techniques

    # Placeholder for additional preprocessing steps or enhancements
    # Add your code or placeholders for any other important preprocessing steps

    print("Data exploration and preprocessing complete.")  # Debugging line

    return df_preprocessed

def dimensionality_reduction(df):
    print("Starting dimensionality reduction.")  # Debugging line
    
    # Select the relevant features for PCA
    features = df.drop(['fin_text'], axis=1)

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

    print("Dimensionality reduction complete.")  # Debugging line
    return df_processed

def feature_engineering(df):
    print("Starting feature engineering.")  # Debugging line

    # Historical Performance
    df['avg_round_score'] = df.groupby('player_name')['round_score'].transform('mean')

    # Performance on Specific Course
    df['avg_round_score_course'] = df.groupby(['player_name', 'course_name'])['round_score'].transform('mean')

    # Player Consistency
    df['std_round_score'] = df.groupby('player_name')['round_score'].transform('std')

    # Overall Game Skills
    df['overall_skill_score'] = df[['sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g', 'sg_total']].sum(axis=1)

    # Seasonal Performance
    # Assuming that there is a 'season' column in the data
    df['season_avg_round_score'] = df.groupby(['player_name', 'season'])['round_score'].transform('mean')

    # Player Progression
    # Assuming that there is a 'year' column in the data
    df['yearly_avg_round_score'] = df.groupby(['player_name', 'year'])['round_score'].transform('mean')
    df['player_progression'] = df.groupby('player_name')['yearly_avg_round_score'].transform(lambda x: x.pct_change())

    print("Feature engineering complete.")  # Debugging line
    return df

def create_model(optimizer='adam'):
    model = Sequential()
    model.add(Dense(12, input_dim=7, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def algorithm_development(df):
    print("Starting algorithm development.")  # Debugging line

    # Split data into features and target
    X = df.drop(['fin_text'], axis=1)
    y = df['fin_text']
    
    # Apply feature scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define logistic regression model
    lr = LogisticRegression()

    # Define the grid search parameters
    param_grid = [{'C': [0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}]

    # Grid search for hyperparameters
    grid = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5)
    grid_result = grid.fit(X_train, y_train)

    # Get best hyperparameters
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    # Define the keras classifier
    model = KerasClassifier(build_fn=create_model, epochs=50, batch_size=10, verbose=0)

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test))

    # Evaluate the model
    scores = model.evaluate(X_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    # Finalize the model with chosen hyperparameters
    final_model = LogisticRegression(**grid_result.best_params_)
    final_model.fit(X_train, y_train)

    print("Algorithm development complete.")  # Debugging line
    return final_model

def get_player_data(tour='pga', file_format='csv'):
    url = f"https://feeds.datagolf.com/field-updates?tour={tour}&file_format={file_format}&key=195c3..."

    # Send a GET request to the API
    response = requests.get(url)

    # Raise an exception if the request was unsuccessful
    response.raise_for_status()

    # If the requested file format is CSV
    if file_format.lower() == 'csv':
        data = pd.read_csv(StringIO(response.text))

    # Return the data
    return data

def predict_future_tournaments(players_list, df_historical, model):
    print("Starting sim of upcoming tournament.")  # Debugging line

    # Fetch historical data for the players in the future tournament
    df_future = df_historical[df_historical['player_name'].isin(players_list)]

    # Preprocessing the players' historical data
    df_future_preprocessed = explore_and_preprocess_data(df_future)
    
    # Feature engineering
    df_future_fe = feature_engineering(df_future_preprocessed)

    # Drop the 'fin_text' column as it is the target variable and wouldn't be present in future data
    df_future_fe = df_future_fe.drop(['fin_text'], axis=1)
    
    # Apply feature scaling
    scaler = StandardScaler()
    df_future_fe_scaled = scaler.fit_transform(df_future_fe)

    # Use the model to make predictions
    predictions = model.predict(df_future_fe_scaled)
    
    print("Prediction of future tournaments complete.")  # Debugging line
    
    return predictions

def analyze_and_evaluate_predictions(predictions, actual_results):
    print("Starting analysis of player finishes.")  # Debugging line

    # Compute and print the MAE
    mae = mean_absolute_error(actual_results, predictions)
    print(f"Mean Absolute Error: {mae}")

    # Compute and print the MSE
    mse = mean_squared_error(actual_results, predictions)
    print(f"Mean Squared Error: {mse}")

    # Compute and print the RMSE
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error: {rmse}")

    # Compute and print the R2 score
    r2 = r2_score(actual_results, predictions)
    print(f"R2 Score: {r2}")

    # Return the metrics
    return mae, mse, rmse, r2

# Main execution flow
if __name__ == '__main__':
    try:
        conn = connect_to_database('/Users/michaelfuscoletti/Desktop/data/pgatour.db')
        data = retrieve_data(conn, 'raw_data')
        data = explore_and_preprocess_data(data)
        create_model()
        data = feature_engineering(data)
        model = algorithm_development(data)
        players_list = get_player_data
        predictions = predict_future_tournaments(players_list, data, model)
        evaluation_metrics = analyze_and_evaluate_predictions(predictions, actual_results)
        print("Execution complete.")  # Debugging line
    except Exception as e:
        print(f"Execution error: {e}")  # Debugging line
