import warnings
import numpy as np
from numba import jit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.metrics import mean_absolute_error, accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from joblib import Parallel, delayed


def split_data(df, feature_columns, target_column, task_type, max_class_count=None):
    """
    Split the dataset into a training set and a test set based on the task type.

    Parameters:
    df (DataFrame): The dataset.
    feature_columns (list): The feature columns.
    target_column (str): The target column.
    task_type (str): The type of task - 'classification' or 'regression'.
    max_class_count (int): The maximum number of samples per class. If specified, the function will balance the dataset.

    Returns:
    tuple: The training set and the test set.
    """

    # Select the feature columns
    X = df[feature_columns]

    # Select the target variable
    y = df[target_column]

    if task_type == 'classification':
        print(f'Before preprocessing, we have {len(X)} samples and the class distribution is:')
        print(y.value_counts())

        # Balance the dataset if necessary
        if max_class_count:
            X, y = balance_data(X, y, max_class_count)

        # Split the dataset into a training set and a test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=108, stratify=y)

    elif task_type == 'regression':
        print(f'Before preprocessing, we have {len(X)} samples and the target distribution is:')
        print(y.describe())

        # Split the dataset into a training set and a test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=108)

    else:
        raise ValueError(f"Invalid task_type: {task_type}. Expected 'classification' or 'regression'.")

    print(f'After preprocessing, we have {len(X_train)} training samples and {len(X_test)} test samples.')

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
        rus = RandomUnderSampler(sampling_strategy='majority', random_state=108)
        X_res, y_res = rus.fit_resample(X, y)
        
        # Make sure that we do not exceed max_class_count
        classes, counts = np.unique(y_res, return_counts=True)
        for cls, count in zip(classes, counts):
            if count > max_class_count:
                raise ValueError(f"After undersampling, class '{cls}' has {count} samples, which is more than the specified max_class_count ({max_class_count}).")
    else:
        X_res, y_res = X, y

    return X_res, y_res


def monte_carlo_simulation(model, X, n, kf_splits):
    # Perform one Monte Carlo simulation
    monte_carlo_predictions = []

    kf = KFold(n_splits=kf_splits, shuffle=True)

    for _ in range(n):
        fold_predictions = []

        for _, test_indices in kf.split(X):
            X_test_fold = X[test_indices]
            y_sample_pred = model.predict(X_test_fold)
            fold_predictions.append(y_sample_pred)

        monte_carlo_predictions.append(np.concatenate(fold_predictions))

    return monte_carlo_predictions


def train_and_evaluate_model(estimator, param_grid, X_train, X_test, y_train, y_test, problem_type, num_samples_range=[100], cv=5):
    """
    Train and evaluate a model using GridSearchCV and perform Monte Carlo simulations.

    Parameters:
    estimator (estimator): The model.
    param_grid (dict): The hyperparameter grid.
    X_train (array-like): The feature matrix of the training set.
    X_test (array-like): The feature matrix of the test set.
    y_train (array-like): The target variable of the training set.
    y_test (array-like): The target variable of the test set.
    problem_type (str): The type of problem - 'classification' or 'regression'.
    num_samples_range (list): The range of number of samples for the Monte Carlo simulations.
    cv (int): The number of cross-validation folds.

    Returns:
    tuple: The best model, the best metric score, and the scaler.
    """

    print("Scaling the data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training the model using GridSearchCV...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)  # Ignore the specific warning
        CV_rfc = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=2)
        CV_rfc.fit(X_train_scaled, y_train)
        print(f"Best parameters found by GridSearchCV: {CV_rfc.best_params_}")

        print("GridSearchCV results:")
        cv_results = CV_rfc.cv_results_
        for mean_score, std_score, params in zip(cv_results["mean_test_score"], cv_results["std_test_score"], cv_results["params"]):
            print(f"Mean score: {mean_score:.4f}, Std score: {std_score:.4f}, Params: {params}")

    best_num_samples = None
    best_metric = float('inf')

    for num_samples in num_samples_range:
        print(f"Performing Monte Carlo simulations with {num_samples} iterations...")
        monte_carlo_predictions = []

        def simulate_iteration(i):
            # Perform the simulation
            predictions = monte_carlo_simulation(CV_rfc.best_estimator_, X_test_scaled, num_samples, cv)
            return predictions

        monte_carlo_predictions = Parallel(n_jobs=2, verbose=2)(
            delayed(
                simulate_iteration)(i) for i in range(num_samples)
        )

        monte_carlo_predictions = np.concatenate(monte_carlo_predictions, axis=0)

        print("Computing metrics over the distribution of predictions...")
        prediction_mean = np.mean(monte_carlo_predictions, axis=0)

        if problem_type == 'classification':
            # Assumes that your model is outputting probabilities
            class_prediction = (prediction_mean > 0.5).astype(int)
            metric = accuracy_score(y_test, class_prediction)
            print(f"Accuracy on the test set: {metric}")
        elif problem_type == 'regression':
            errors = [mean_absolute_error(y_test, pred) for pred in monte_carlo_predictions]
            metric = np.mean(errors)
            print(f"Mean absolute error on the test set: {metric}")

        if metric < best_metric:
            best_metric = metric
            best_num_samples = num_samples

    print("Best Monte Carlo iteration number:", best_num_samples)

    return CV_rfc.best_estimator_, best_metric, scaler