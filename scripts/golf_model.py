import sqlite3
import pandas as pd
import logging
from tqdm import tqdm
from datetime import date
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin
import statsmodels.api as sm
from constants import constants
from features import features
from utils.model_utils import train_and_evaluate_model, split_data, save_model
from utils.dataframe_utils import handle_missing_values, perform_knn_imputation

logging.basicConfig(
    filename=f"{constants['model_path']}/model.log", level=logging.INFO
)


class SMWrapper(BaseEstimator, RegressorMixin):
    """ A universal sklearn-style wrapper for statsmodels regressors """
    def __init__(self, model_class, fit_intercept=True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        if self.fit_intercept:
            X = sm.add_constant(X)
        self.model_ = self.model_class(y, X)
        self.results_ = self.model_.fit()
        self.coef_ = (
            self.results_.params if hasattr(self.results_, 'params') else None
        )

    def predict(self, X):
        if self.fit_intercept:
            X = sm.add_constant(X)
        return self.results_.predict(X)


# Define parameter grids for different models
param_grid_lr = {
    'fit_intercept': [True],
    'n_jobs': [-1],
    'positive': [False]
}

param_grid_gb = {
    'n_estimators': [5000],
    'learning_rate': [0.5],
    'max_depth': [3],
    'min_samples_split': [100],
    'min_samples_leaf': [2]
}

param_grid_rf = {
    'n_estimators': [1000],
    'max_depth': [None],
    'min_samples_split': [2],
    'min_samples_leaf': [1]
}

param_grid_dt = {
    'criterion': ['friedman_mse'],
    'splitter': ['random'],
    'max_depth': [None],
    'min_samples_split': [10, 50, 100],
    'min_samples_leaf': [4, 16, 64]
}

param_grid_mlp = {
    'hidden_layer_sizes': [(200, 200)],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [0.01],
    'learning_rate': ['constant'],
}


def main():
    print("Establishing connection to the database...")
    db_conn = sqlite3.connect(constants['database_path'])
    cursor = db_conn.cursor()

    conditions = [f"{col} >= -1000 AND {col} <= 1000" for col in features]
    where_clause = " AND ".join(conditions)
    where_clause += " AND " + " AND ".join(
        [f"{col} IS NOT NULL" for col in features]
    )

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
            (
                "neural_network",
                MLPRegressor(max_iter=500, random_state=108),
                param_grid_mlp
            ),
            (
                "ordinary_least_squares",
                SMWrapper(sm.OLS),
                {}
            ),
            (
                "linear_regression",
                LinearRegression(),
                param_grid_lr
            ),
            (
                "random_forest",
                RandomForestRegressor(random_state=108),
                param_grid_rf
            ),
            (
                "decision_tree",
                DecisionTreeRegressor(random_state=108),
                param_grid_dt
            ),
            (
                "gradient_boosting",
                GradientBoostingRegressor(random_state=108),
                param_grid_gb
            ),
        ]

        for model_name, model, param_grid in tqdm(
            models,
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

            print(f"Preparing performance data for {model_name} model...")
            performance_data = {
                'X_test': X_test,
                'y_test': y_test,
                'error': error
            }

            # Log model performance
            logging.info(f"{model_name} error: {error}")

            # Initialize feature_importances as None
            feature_importances = None

            # Log feature importances for linear models
            if isinstance(trained_model, LinearRegression):
                feature_importances = dict(
                    zip(column_names[:-1], trained_model.coef_)
                )

            # Log feature importances for tree-based models
            elif hasattr(trained_model, 'feature_importances_'):
                feature_importances = dict(
                    zip(column_names[:-1], trained_model.feature_importances_)
                )

            logging.info(
                f"{model_name} feature importances: {feature_importances}"
            )

            if model_name == "Neural Network" and hasattr(
                trained_model, 'converged_'
            ) and not trained_model.converged_:
                print(f"{model_name} did not converge. "
                      "Consider increasing the maximum number of iterations.")

            print(f"Saving for {model_name} model...")
            model_name_formatted = model_name.replace(' ', '_').lower()
            date_str = date.today().strftime('%Y-%m-%d')
            model_filename = f"{model_name_formatted}_{date_str}"
            save_model(
                trained_model,
                constants['model_path'],
                model_filename,
                column_names[:-1],
                performance_data
            )

    print("\nProcess completed successfully!")
    db_conn.close()


if __name__ == "__main__":
    main()
