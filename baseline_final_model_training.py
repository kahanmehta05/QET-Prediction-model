# baseline_model_training.py

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# -------------------------
# Load Data
# -------------------------
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
X_test = pd.read_csv("X_test.csv")
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
# Train Random Forest (Default)
# -------------------------
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rf_r2 = evaluate_model("Random Forest (Default)", rf_model, X_test, y_test)

# -------------------------
# Train XGBoost (Default)
# -------------------------
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)
xgb_r2 = evaluate_model("XGBoost (Default)", xgb_model, X_test, y_test)

# -------------------------
# Train Gradient Boosting (Default)
# -------------------------
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)
gb_r2 = evaluate_model("Gradient Boosting (Default)", gb_model, X_test, y_test)

# -------------------------
# Compare Models
# -------------------------
best_model = max([
    ("Random Forest", rf_model, rf_r2),
    ("XGBoost", xgb_model, xgb_r2),
    ("Gradient Boosting", gb_model, gb_r2)
], key=lambda x: x[2])

print(f"\n🏆 Best Default Model: {best_model[0]} with R² = {best_model[2]:.4f}")
