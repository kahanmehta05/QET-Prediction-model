# final_model_training.py

"""
This script retrains the best-performing regression models (Random Forest, XGBoost, Gradient Boosting)
using tuned hyperparameters. It removes known data leakage features and evaluates models on the test set.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import joblib

# -------------------------
# Load Data & Drop Leaky Features
# -------------------------
drop_leaky_features = [
   
    "total_actual_rows",
    "total_actual_loops",
    "total_actual_time_node",
    "output_rows", # This is actual rows from the root node
    "rows_estimate_error", # These are calculated using actual vs estimated, so not pure pre-exec
    "rows_estimate_ratio", # Same as above
    "output_row_estimation_error", # Same as above
    "output_row_estimation_ratio", # Same as above
    "cost_time_ratio", # Uses actual_execution_time
    "sum_node_actual_time_vs_total_actual_time_ratio",
]

X_train = pd.read_csv("X_train.csv").drop(columns=drop_leaky_features, errors='ignore')
y_train = pd.read_csv("y_train.csv").values.ravel()
X_test = pd.read_csv("X_test.csv").drop(columns=drop_leaky_features, errors='ignore')
y_test = pd.read_csv("y_test.csv").values.ravel()

# -------------------------
# Evaluation Function
# -------------------------
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n📊 {name} Evaluation:")
    print(f"RMSE (ms): {rmse:,.2f}")
    print(f"MAE  (ms): {mae:,.2f}")
    print(f"R² Score : {r2:.4f}")
    return r2

# -------------------------
# Train Random Forest
# -------------------------
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_r2 = evaluate_model("Random Forest", rf_model, X_test, y_test)
joblib.dump(rf_model, "best_random_forest.pkl")

# -------------------------
# Train XGBoost
# -------------------------
xgb_model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
xgb_r2 = evaluate_model("XGBoost", xgb_model, X_test, y_test)
joblib.dump(xgb_model, "best_xgboost.pkl")

# -------------------------
# Train Gradient Boosting
# -------------------------
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_r2 = evaluate_model("Gradient Boosting", gb_model, X_test, y_test)
joblib.dump(gb_model, "best_gradient_boosting.pkl")

# -------------------------
# Feature Importance Viewer
# -------------------------
def print_top_features(model, X, model_name):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_names = X.columns
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        print(f"\n🔍 Top 10 Important Features for {model_name}:")
        print(importance_df.head(10))
    else:
        print(f"\n❌ {model_name} does not support feature importance.")

# View top features
print_top_features(rf_model, X_train, "Random Forest")
print_top_features(xgb_model, X_train, "XGBoost")
print_top_features(gb_model, X_train, "Gradient Boosting")

# -------------------------
# Select Best Model
# -------------------------
best_model = max([
    ("Random Forest", rf_model, rf_r2),
    ("XGBoost", xgb_model, xgb_r2),
    ("Gradient Boosting", gb_model, gb_r2)
], key=lambda x: x[2])

print(f"\n🏆 Best Model: {best_model[0]} with R² = {best_model[2]:.4f}")
