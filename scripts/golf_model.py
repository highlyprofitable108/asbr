import sqlite3
import pandas as pd
import logging
from tqdm import tqdm
from datetime import date
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from constants import constants
from features import features
from utils.file_utils import save_model
from utils.model_utils import train_and_evaluate_model, split_data
from utils.dataframe_utils import handle_missing_values, perform_knn_imputation

logging.basicConfig(
    filename=f"{constants['model_path']}/model.log", level=logging.INFO
)

# Define parameter grids for different models
param_grid_gb = {
    'n_estimators': [1000, 5000],
    'learning_rate': [0.1],
    'max_depth': [1],
    'min_samples_split': [2],
    'min_samples_leaf': [2]
}

param_grid_dt = {
    'criterion': ['squared_error'],
    'splitter': ['best'],
    'max_depth': [None],
    'min_samples_split': [5],
    'min_samples_leaf': [2]
}

param_grid_lr = {
    'fit_intercept': [True],
    'n_jobs': [-1],
    'positive': [False]
}


param_grid_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
}

param_grid_rf = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

param_grid_ols = {
    'fit_intercept': [True, False],
    'normalize': [True, False],
}


def main():
    print("Establishing connection to the database...")
    db_conn = sqlite3.connect(constants['database_path'])
    cursor = db_conn.cursor()

    conditions = [f"{col} >= -1000 AND {col} <= 1000" for col in features]
    where_clause = " AND ".join(conditions)
    where_clause += " AND " + " AND ".join(
        [f"{col} IS NOT NULL" for col in features])

    print("Executing SQL query...")
    cursor.execute(f"""
        SELECT {', '.join(features + [constants['target_variable']])}
        FROM model_data
        WHERE {where_clause};
    """)

    data = cursor.fetchall()

    if data:
        print("Data fetched successfully. Preprocessing data...")
        column_names = features + [constants['target_variable']]
        df = pd.DataFrame(data, columns=column_names)
        df = perform_knn_imputation(df)
        df = handle_missing_values(df)

        print("Splitting data into train and test datasets...")
        X_train, X_test, y_train, y_test = split_data(
            df,
            [col for col in column_names if col in features],
            constants['target_variable'],
            'regression'
        )

        models = [
            ("Gradient Boosting",
             GradientBoostingRegressor(random_state=0), param_grid_gb),
            ("Decision Tree",
             DecisionTreeRegressor(random_state=0), param_grid_dt),
            ("Linear Regression",
             LinearRegression(), param_grid_lr),
            ("Neural Network",
             MLPRegressor(max_iter=500, random_state=0), param_grid_mlp),
            ("Random Forest",
             RandomForestRegressor(random_state=0), param_grid_rf),
            ("Ordinary Least Squares",
             LinearRegression(), param_grid_ols)
        ]

        for model_name, model, param_grid in tqdm(models,
                                                  desc='Training models',
                                                  unit='model'
                                                  ):
            print(f"\nTraining and evaluating {model_name} model...")
            trained_model, error, _ = train_and_evaluate_model(
                model,
                param_grid,
                X_train,
                X_test,
                y_train,
                y_test,
                'regression'
            )

            print(f"Saving {model_name} model...")
            model_name_formatted = model_name.replace(' ', '_').lower()
            date_str = date.today().strftime()
            model_filename = f"{model_name_formatted}_{date_str}"
            save_model(trained_model,
                       constants['model_path'],
                       model_filename, column_names[:-1])

            print(f"Preparing performance data for {model_name} model...")
            performance_data = {
                'X_test': X_test,
                'y_test': y_test,
                'error': error
            }

            # Log model performance
            logging.info(f"{model_name} error: {error}")

            # Log feature importances for tree-based models
            if isinstance(trained_model, LinearRegression):
                feature_importances = dict(
                    zip(
                        column_names[:-1], trained_model.coef_
                        )
                    )
            else:
                feature_importances = dict(
                    zip(
                        column_names[:-1], trained_model.feature_importances_
                        )
                    )

            logging.info(
                f"{model_name} feature importances: {feature_importances}"
                )

            if model_name == "Neural Network" and not trained_model.converged_:
                logging.warning(
                    f"{model_name} did not converge. "
                    " Consider increasing the maximum number of iterations."
                )

            print(f"Saving performance data for {model_name} model...")
            performance_filename = (
                f"performance_data_{model_name.replace(' ', '_').lower()}"
                f"_{date.today().strftime('%Y-%m-%d')}"
            )

            save_model(
                performance_data,
                constants['model_path'],
                performance_filename,
                column_names[:-1]
            )

    print("\nProcess completed successfully!")
    db_conn.close()


if __name__ == "__main__":
    main()
