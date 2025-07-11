import pandas as pd
import shap
import joblib
from xgboost import XGBRegressor

# Load training data
X_train = pd.read_csv("X_train.csv")

# Drop post-execution features
drop_leaky_features = [
    "total_actual_rows",
    "total_actual_loops",
    "total_actual_time_node",
    "output_rows",
    "rows_estimate_error",
    "rows_estimate_ratio",
    "output_row_estimation_error",
    "output_row_estimation_ratio",
    "cost_time_ratio",
    "sum_node_actual_time_vs_total_actual_time_ratio",
]
X_train = X_train.drop(columns=drop_leaky_features, errors='ignore')

# Load the trained XGBoost model
model = joblib.load("best_xgboost.pkl")

# Use SHAP TreeExplainer
explainer = shap.Explainer(model)
shap_values = explainer(X_train)

# Show top 10 features
shap_df = pd.DataFrame({
    "feature": X_train.columns,
    "mean_abs_shap": shap_values.abs.mean(0).values
}).sort_values(by="mean_abs_shap", ascending=False).head(10)

print("\n🔍 Top 10 SHAP Features:")
print(shap_df)

# Optional: visualize
shap.plots.bar(shap_values, max_display=10)
