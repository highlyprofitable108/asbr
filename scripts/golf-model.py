import pandas as pd
import numpy as np
from joblib import dump
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.utils import resample
import os
from datetime import date

# Assuming that your script is in the root project directory
root_project_dir = os.getcwd() # gets current working directory

# Load your data
relative_path = "data/processed_data/golf/pkl/df_other.pkl"
absolute_path = os.path.join(root_project_dir, relative_path)
df = pd.read_pickle(absolute_path)

# Create pivot tables for all columns that have 1-4 rows of data
# Replace 'columns_to_pivot' with your actual column names
print("Creating pivot tables...")
columns_to_pivot = ['round_score', 'sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g', 'sg_total', 'driving_dist', 'driving_acc', 'gir', 'scrambling', 'prox_rgh', 'prox_fw']

# Preserve static columns
static_columns = ['year', 'season', 'event_completed', 'event_name', 'event_id', 'player_name', 'dg_id', 'fin_text']
df_static = df[static_columns].drop_duplicates()

for column in columns_to_pivot:
    print(f"Pivoting column: {column}")
    pivot = df.pivot_table(index=['event_id', 'dg_id'], columns='round_num', values=column)
    pivot.columns = [f"{column}_{i}" for i in pivot.columns]
    df_static = df_static.join(pivot, on=['event_id', 'dg_id'])

# For total_score and sg_total, check if all 4 rounds exist, if so, calculate the sum
print("Calculating total scores...")
round_columns = [f'round_score_{i}' for i in range(1, 4)]
df_static['total_score'] = df_static[round_columns].sum(axis=1)
df_static.loc[df_static[round_columns].isnull().any(axis=1), 'total_score'] = np.nan

sg_columns = [f'sg_total_{i}' for i in range(1, 4)]
df_static['sg_total_all'] = df_static[sg_columns].sum(axis=1)
df_static.loc[df_static[sg_columns].isnull().any(axis=1), 'sg_total_all'] = np.nan
df = df_static

# Model-based imputation
print("Performing model-based imputation...")
imputer = IterativeImputer()

# Separating non-numeric columns
original_index = df.index
df_non_numeric = df.select_dtypes(exclude=[np.number])

# Dropping non-numeric columns from df
df_numeric = df.select_dtypes(include=[np.number])

# Perform imputation on df_numeric
df_numeric = pd.DataFrame(imputer.fit_transform(df_numeric), columns = df_numeric.columns, index = original_index)

# Joining non-numeric columns back to the imputed dataframe
df_non_numeric.reindex(df_numeric.index)
df = pd.concat([df_numeric, df_non_numeric], axis=1)

# Convert to datetime format
df['event_completed'] = pd.to_datetime(df['event_completed'])

# Calculate the difference in days from the current date
df['days_since_event'] = (pd.Timestamp('today') - df['event_completed']).dt.days

# Implement time decay
df['weight'] = np.exp(-df['days_since_event'] / 120)

# Multiply the weights with your target variable
df['total_score_weighted'] = df['total_score'] * df['weight']


# Set your features and target variables
print("Setting up features and targets...")
features = [
    'dg_id', 
    'event_id', 
    'sg_putt_1', 'sg_putt_2', 'sg_putt_3', 'sg_putt_4',
    'sg_arg_1', 'sg_arg_2', 'sg_arg_3', 'sg_arg_4',
    'sg_app_1', 'sg_app_2', 'sg_app_3', 'sg_app_4',
    'sg_ott_1', 'sg_ott_2', 'sg_ott_3', 'sg_ott_4',
    'sg_t2g_1', 'sg_t2g_2', 'sg_t2g_3', 'sg_t2g_4',
]

target = 'total_score_weighted'

# Split your data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=108)

# Initialize your scaler
scaler = StandardScaler()

# Fit your scaler and transform your training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform your test data
X_test_scaled = scaler.transform(X_test)

# Defining the hyperparameters to tune
print("Performing GridSearchCV...")
param_grid = { 
    'n_estimators': [100, 200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['mse', 'mae']
}

# GridSearchCV with RandomForest
CV_rfc = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid, cv= 3, verbose=2, n_jobs=-1)
CV_rfc.fit(X_train_scaled, y_train)

# Printing the best parameters
print(f"Best params: {CV_rfc.best_params_}")
print(CV_rfc.best_params_)

# Create an instance of RandomForestRegressor with the best parameters
rfc_best=RandomForestRegressor(n_estimators=CV_rfc.best_params_['n_estimators'], 
                               max_features=CV_rfc.best_params_['max_features'],
                               max_depth=CV_rfc.best_params_['max_depth'], 
                               criterion=CV_rfc.best_params_['criterion'],
                               n_jobs=-1)

# Fitting the model to our data
rfc_best.fit(X_train_scaled, y_train)

# Add evaluation metrics
y_pred = rfc_best.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae}")

# Running simulations
simulation_count = 20000  # replace with desired number of simulations
results = []
print("Running simulations.....")
for i in range(simulation_count):
    print(f"Going throught the matrix {i} out of 20000 times.")
    # Generate bootstrap sample
    bootstrap_sample_X, bootstrap_sample_y = resample(X_train_scaled, y_train)
    
    # Fit model
    rfc_best.fit(bootstrap_sample_X, bootstrap_sample_y)
    
    # Predict and calculate error for this simulation
    y_pred = rfc_best.predict(X_test_scaled)
    error = mean_absolute_error(y_test, y_pred)
    
    # Append error to results
    results.append(error)

# Print mean of results
print(f"Mean error over {simulation_count} simulations: {np.mean(results)}")

# Save your model with datestamp
model_filename = f"rfc_best_{date.today().strftime('%Y-%m-%d')}.joblib"
model_filepath = os.path.join("/Users/michaelfuscoletti/Desktop/data/processed_data/golf/", model_filename)
absolute_path = os.path.join(root_project_dir, relative_path, model_filename)
joblib.dump(rfc_best, model_filepath)
