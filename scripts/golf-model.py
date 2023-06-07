import pandas as pd
import numpy as np
from joblib import dump
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer

# Load your data
df = pd.read_pickle('/Users/michaelfuscoletti/Desktop/data/processed_data/golf/pkl/df_other.pkl')

# Create pivot tables for all columns that have 1-4 rows of data
# Replace 'columns_to_pivot' with your actual column names
columns_to_pivot = ['round_score', 'sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g', 'sg_total', 'driving_dist', 'driving_acc', 'gir', 'scrambling', 'prox_rgh', 'prox_fw']

# Preserve static columns
static_columns = ['year', 'season', 'event_completed', 'event_name', 'event_id', 'player_name', 'dg_id', 'fin_text']
df_static = df[static_columns].drop_duplicates()

for column in columns_to_pivot:
    pivot = df.pivot_table(index=['event_id', 'dg_id'], columns='round_num', values=column)
    pivot.columns = [f"{column}_{i}" for i in pivot.columns]
    df_static = df_static.join(pivot, on=['event_id', 'dg_id'])

# For total_score and sg_total, check if all 4 rounds exist, if so, calculate the sum
round_columns = [f'round_score_{i}' for i in range(1, 4)]
df_static['total_score'] = df_static[round_columns].sum(axis=1)
df_static.loc[df_static[round_columns].isnull().any(axis=1), 'total_score'] = np.nan

sg_columns = [f'sg_total_{i}' for i in range(1, 4)]
df_static['sg_total_all'] = df_static[sg_columns].sum(axis=1)
df_static.loc[df_static[sg_columns].isnull().any(axis=1), 'sg_total_all'] = np.nan
df = df_static

# Model-based imputation
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

features = [
    'dg_id', 
    'event_id', 
    'sg_putt_1', 'sg_putt_2', 'sg_putt_3', 'sg_putt_4',
    'sg_arg_1', 'sg_arg_2', 'sg_arg_3', 'sg_arg_4',
    'sg_app_1', 'sg_app_2', 'sg_app_3', 'sg_app_4',
    'sg_ott_1', 'sg_ott_2', 'sg_ott_3', 'sg_ott_4',
    'sg_t2g_1', 'sg_t2g_2', 'sg_t2g_3', 'sg_t2g_4',
]  # replace with actual feature names

# Define your targets
target = 'total_score'  # replace with actual target column name

# Split your data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=108)

# Initialize your scaler
scaler = StandardScaler()

# Fit your scaler and transform your training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform your test data
X_test_scaled = scaler.transform(X_test)

# Initialize your models
models = [
    ('Linear Regression', LinearRegression()),
    ('Lasso Regression', Lasso()),
    ('Ridge Regression', Ridge()),
    ('Decision Tree', DecisionTreeRegressor()),
    ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=108)),
    ('Gradient Boosting', GradientBoostingRegressor(random_state=108))
]

# Train and evaluate each model
model_path = '/Users/michaelfuscoletti/Desktop/data/processed_data/golf/models/'
for model_name, model in models:
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} MAE: {mae}")
    print(f"{model_name} MSE: {mse}")
    print(f"{model_name} RMSE: {rmse}")
    print(f"{model_name} MAPE: {mape}%")
    print(f"{model_name} R2 Score: {r2}")

    # Save model after training and evaluating
    dump(model, f"{model_path}{model_name.replace(' ', '_').lower()}.joblib")
    
# TODO: Implement feature importance or coefficients printing depending on the model.
# For example, for linear regression, you can print model.coef_
# For tree-based models, you can print model.feature_importances_
