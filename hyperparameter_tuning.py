"""
Hyperparameter Tuning Script for Regression Models (RandomForest, XGBoost, GradientBoosting)
- Loads training data (X_train.csv and y_train.csv)
- Performs GridSearchCV to find best hyperparameters
- Prints best parameters and cross-validation score
"""

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load cleaned training data
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()  # Ensure it's a flat array

# Define models and their hyperparameter grids
models = {
    "RandomForest": {
        "model": RandomForestRegressor(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
        }
    },
    "XGBoost": {
        "model": XGBRegressor(random_state=42, verbosity=0),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [3, 6, 10],
            "learning_rate": [0.05, 0.1, 0.2]
        }
    },
    "GradientBoosting": {
        "model": GradientBoostingRegressor(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1, 0.2]
        }
    }
}

# Run grid search for each model
for name, m in models.items():
    print(f"\n🔍 Tuning {name}...")
    grid = GridSearchCV(m["model"], m["params"], cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    print(f"✅ Best parameters for {name}: {grid.best_params_}")
    print(f"📉 Best cross-validated RMSE: {-grid.best_score_:.2f}")
