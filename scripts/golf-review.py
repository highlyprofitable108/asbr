from joblib import load
import os
import pandas as pd
import matplotlib.pyplot as plt

# Constants
model_dir = '/Users/michaelfuscoletti/Desktop/asbr/data/golf/models/'

# Get the most recent model file
model_files = [file for file in os.listdir(model_dir) if file.endswith('.joblib')]
model_files.sort(reverse=True)
recent_model_file = model_files[0] if model_files else None

if recent_model_file:
    # Load the most recent model
    model_path = os.path.join(model_dir, recent_model_file)
    model = load(model_path)
else:
    print('No model files found.')
    model = None

# Print the raw data of the model
print("Model data:")
print(model)

# Features used in the model
features = [
    'round_num',
    'sg_ott',
    'sg_putt',
    'sg_arg',
    'sg_app'
]

# Print the model details
print("\nModel parameters:")
print(model.get_params())

# If the model is a sklearn.ensemble.RandomForestRegressor,
# we can access specific attributes:

print("\nNumber of estimators:", model.n_estimators)
print("Criterion:", model.criterion)
print("Max depth:", model.max_depth)
print("Min samples split:", model.min_samples_split)
print("Min samples leaf:", model.min_samples_leaf)
print("Max features:", model.max_features)

# Check if the model has coefficients (applicable for linear models)
if hasattr(model, 'coef_'):
    coefficients = model.coef_
    coefficients_dict = {name: coef for name, coef in zip(features, coefficients)}
    print("\nCoefficients:", coefficients_dict)

# Get and print feature importances (applicable for tree-based models)
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    importances_dict = {name: imp for name, imp in zip(features, importances)}
    print("\nFeature importances:", importances_dict)

# Plot feature importances
if hasattr(model, 'feature_importances_'):
    plt.barh(range(len(importances_dict)), list(importances_dict.values()), align='center')
    plt.yticks(range(len(importances_dict)), list(importances_dict.keys()))
    plt.xlabel('Importance')
    plt.title('Feature importances')
    plt.show()