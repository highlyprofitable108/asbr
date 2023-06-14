import os
import pandas as pd
import numpy as np
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from datetime import date
from scipy.stats import norm

# Constants
constants = {
    'file_path': '/Users/michaelfuscoletti/Desktop/asbr/data/golf/csv/',
    'model_path': '/Users/michaelfuscoletti/Desktop/asbr/data/golf/models/',
    'pkl_path': '/Users/michaelfuscoletti/Desktop/asbr/data/golf/pkl/',
}

def load_data():
    # Get the most recent date-stamped pickle file
    pkl_files = [file for file in os.listdir(constants['pkl_path']) if file.endswith('.pkl')]
    pkl_files.sort(reverse=True)
    recent_file = pkl_files[0] if pkl_files else None

    if recent_file:
        df_path = os.path.join(constants['pkl_path'], recent_file)
        df = pd.read_pickle(df_path)
        return df
    else:
        print("No pickle files found.")
        return None

def perform_knn_imputation(df):
    imputer = KNNImputer(n_neighbors=5, weights='uniform')

    df_numeric = df.select_dtypes(include=[np.number])
    df_non_numeric = df.select_dtypes(exclude=[np.number]).reset_index(drop=True)  # Reset index

    df_numeric_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)

    df = pd.concat([df_numeric_imputed, df_non_numeric], axis=1)

    return df

def set_up_features_and_targets(df):
    features = ['round_num', 'sg_putt', 'sg_arg', 'sg_app', 'sg_ott']
    target = 'round_score'
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=108)
    return X_train, X_test, y_train, y_test

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_features': [1.0, None, 'log2'],
        'max_depth': [4, 5, 6, 7, 8],
        'criterion': ['squared_error'],
        'min_samples_split': [2]
    }

    CV_rfc = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    CV_rfc.fit(X_train_scaled, y_train)
    print(f"Best params: {CV_rfc.best_params_}")

    y_pred = CV_rfc.predict(X_test_scaled)
    error = mean_absolute_error(y_test, y_pred)

    return CV_rfc, error

def save_model(CV_rfc):
    model_filename = f"rfc_best_{date.today().strftime('%Y-%m-%d')}.joblib"
    model_path = os.path.join(constants['model_path'], model_filename)
    dump(CV_rfc.best_estimator_, model_path)

def main():
    df = load_data()
    if df is not None:
        df = perform_knn_imputation(df)
        print(df.columns)
        X_train, X_test, y_train, y_test = set_up_features_and_targets(df)
        CV_rfc, error = train_and_evaluate_model(X_train, X_test, y_train, y_test)
        save_model(CV_rfc)

if __name__ == "__main__":
    main()
