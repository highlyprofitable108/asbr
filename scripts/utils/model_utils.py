import os
import warnings
import pickle
import logging
import numpy as np
from joblib import dump
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, \
    StratifiedKFold
from sklearn.metrics import mean_absolute_error, accuracy_score
from imblearn.under_sampling import RandomUnderSampler


def load_model(model_directory, filename):
    with open(os.path.join(model_directory, filename), 'rb') as f:
        return pickle.load(f)


def save_model(
    model, model_directory, model_filename, feature_columns, performance_data
):
    # Save the model as a joblib file
    joblib_filename = os.path.join(model_directory, f"{model_filename}.joblib")
    dump(model, joblib_filename)

    # Save the feature columns as a pickle file
    feature_columns_filename = os.path.join(
        model_directory, f"{model_filename}_features.pkl"
    )
    with open(feature_columns_filename, 'wb') as f:
        pickle.dump(feature_columns, f)

    # Save the performance data as a text file
    performance_data_filename = os.path.join(
        model_directory, f"{model_filename}_performance.txt"
    )
    with open(performance_data_filename, 'w') as f:
        f.write(f"X_test:\n{performance_data['X_test']}\n\n")
        f.write(f"y_test:\n{performance_data['y_test']}\n\n")
        f.write(f"error: {performance_data['error']}\n")


def split_data(
    df, feature_columns, target_column, task_type, max_class_count=None
):
    """
    Split dataset into a training set and test set based on task type.

    Parameters:
    df (DataFrame): The dataset.
    feature_columns (list): The feature columns.
    target_column (str): The target column.
    task_type (str): The type of task - 'classification' or 'regression'.
    max_class_count (int): The maximum number of samples per class.

    If specified, the function will balance the dataset.

    Returns:
    tuple: The training set and the test set.
    """

    # Select the feature columns
    X = df[feature_columns]

    # Select the target variable
    y = df[target_column]

    if task_type == 'classification':
        print(
            f"Before preprocessing, we have {len(X)} samples. "
            "The class distribution is:"
        )
        print(y.value_counts())

        # Balance the dataset if necessary
        if max_class_count:
            X, y = balance_data(X, y, max_class_count)

        # Split the dataset into a training set and a test set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=108, stratify=y
        )

    elif task_type == 'regression':
        print(
            f"Before preprocessing, we have {len(X)} samples. "
            "The target variable statistics are:"
        )
        print(y.describe())

        # Split the dataset into a training set and a test set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=108
        )

    else:
        raise ValueError(f"Invalid task_type: {task_type}. "
                         "Expected 'classification' or 'regression'.")

    print(f"After preprocessing, we have {len(X_train)} training samples. "
          f"{len(X_test)} test samples.")

    return X_train, X_test, y_train, y_test


def balance_data(X, y, max_class_count=None):
    """
    Balance the dataset by undersampling the majority class.

    Parameters:
    X (DataFrame): The feature matrix.
    y (Series): The target variable.
    max_class_count (int): The maximum number of samples per class.

    Returns:
    tuple: The resampled feature matrix and target variable.
    """

    if max_class_count is not None:
        rus = RandomUnderSampler(
            sampling_strategy='majority', random_state=108
        )
        X_res, y_res = rus.fit_resample(X, y)

        # Make sure that we do not exceed max_class_count
        classes, counts = np.unique(y_res, return_counts=True)
        for cls, count in zip(classes, counts):
            if count > max_class_count:
                raise ValueError(
                    f"After undersampling, class '{cls}' has {count} samples, "
                    f"which is more than max_class_count ({max_class_count})."
                )
    else:
        X_res, y_res = X, y

    return X_res, y_res


def monte_carlo_simulation(model, X, n):
    """
    Perform Monte Carlo simulations

    Parameters:
    model (estimator): The trained model.
    X (array-like): The feature matrix of the test set.
    n (int): The number of Monte Carlo simulations to perform.

    Returns:
    list: The list of dictionaries containing predictions
        and actual values for each simulation.
    """
    monte_carlo_results = []
    for _ in range(n):
        predictions = model.predict(X)
        actuals = np.asarray(['round_score'] * len(X))
        monte_carlo_results.append(
            {'predictions': predictions, 'actuals': actuals}
        )
    print(monte_carlo_results)    
    return monte_carlo_results


def train_and_evaluate_model(
    estimator,
    param_grid,
    X_train,
    X_test,
    y_train,
    y_test,
    problem_type,
    num_samples_range=[250],
    cv=4
):
    """
    Train and evaluate a model using GridSearchCV and
    perform Monte Carlo simulations.

    Parameters:
    estimator (estimator): The model.
    param_grid (dict): The hyperparameter grid.
    X_train (array-like): The feature matrix of the training set.
    X_test (array-like): The feature matrix of the test set.
    y_train (array-like): The target variable of the training set.
    y_test (array-like): The target variable of the test set.
    problem_type (str): The type of problem - 'classification' or 'regression'.
    num_samples_range (list):
        The range of number of samples for the Monte Carlo simulations.
    cv (int): The number of cross-validation folds.

    Returns:
    tuple: The best model, the best metric score, and the scaler.
    """
    print("Scaling the data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training the model using GridSearchCV...")
    # Determine the cross-validation splitter based on the problem type
    if problem_type == 'classification':
        cv_splitter = StratifiedKFold(
            n_splits=cv, shuffle=True, random_state=0
        )
    else:
        cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=0)

    # Ignore the warnings specified.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        CV_rfc = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=cv_splitter,
            n_jobs=-1,
            verbose=2
        )
        CV_rfc.fit(X_train_scaled, y_train)
        print(f"Best parameters found by GridSearchCV: {CV_rfc.best_params_}")

        print("GridSearchCV results:")
        cv_results = CV_rfc.cv_results_
        for mean_score, std_score, params in zip(
            cv_results["mean_test_score"],
            cv_results["std_test_score"],
            cv_results["params"]
        ):
            print(
                f"Mean score: {mean_score:.4f}, "
                f"Std score: {std_score:.4f}, "
                f"Params: {params}"
            )

    best_num_samples = None
    best_metric = float('inf')

    chunk_size = 100  # Number of simulations per job

    for num_samples in num_samples_range:
        print(
            f"Performing Monte Carlo sim with {num_samples} iterations..."
        )

        # Number of jobs is total simulations divided by chunk size
        num_jobs = num_samples // chunk_size

        def simulate_chunk(i):
            # Each job runs chunk_size simulations
            return monte_carlo_simulation(
                CV_rfc.best_estimator_, X_test_scaled, chunk_size
            )

        # Use list comprehension to create jobs
        monte_carlo_predictions = [simulate_chunk(i) for i in range(num_jobs)]

        # Use np.concatenate to merge all predictions into one array
        monte_carlo_predictions = np.concatenate(monte_carlo_predictions)

        print("Computing metrics over the distribution of predictions...")
        prediction_mean = np.mean(monte_carlo_predictions, axis=0)

        if problem_type == 'classification':
            # Assumes that your model is outputting probabilities
            class_prediction = (prediction_mean > 0.5).astype(int)
            metric = accuracy_score(y_test, class_prediction)
            print(f"Accuracy on the test set: {metric}")
        elif problem_type == 'regression':
            metric = mean_absolute_error(y_test, prediction_mean)
            print(f"Mean absolute error on the test set: {metric}")

        if metric < best_metric:
            best_metric = metric
            best_num_samples = num_samples

        # Log metrics and diagnostics
        logging.info(f"Num samples: {num_samples}")
        logging.info(f"Metric: {metric}")
        logging.info(f"Predicted mean: {prediction_mean}")

    print("Best Monte Carlo iteration number:", best_num_samples)

    return CV_rfc.best_estimator_, best_metric, scaler
