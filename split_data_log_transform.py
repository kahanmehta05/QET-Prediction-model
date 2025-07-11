import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# --- Configuration ---
INPUT_CSV = "query_plan_features.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42

TARGET_COLUMN = "actual_execution_time"
ID_COLUMN = "query_id"

# Post-execution features that must be excluded from X_train and X_test
POST_EXECUTION_FEATURES = [
    "total_actual_rows",
    "total_actual_loops",
    "total_actual_time_node",
    "output_rows",  # actual rows from root node
    "rows_estimate_error",
    "rows_estimate_ratio",
    "output_row_estimation_error",
    "output_row_estimation_ratio",
    "cost_time_ratio",
    "sum_node_actual_time_vs_total_actual_time_ratio",
]

# --- Load Data ---
try:
    df = pd.read_csv(INPUT_CSV)
    print(f"✅ Successfully loaded {INPUT_CSV}. Initial shape: {df.shape}")
except FileNotFoundError:
    print(f"❌ Error: {INPUT_CSV} not found.")
    exit()
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

# --- Preprocessing ---

# 1. Log transform skewed features (including post-execution)
print("\n📈 Applying log1p transformation to skewed features...")

log_transform_candidates = [
    col for col in df.columns
    if col != ID_COLUMN and df[col].min() >= 0
]

transformed_features = []
for col in log_transform_candidates:
    if col == TARGET_COLUMN:
        continue
    if df[col].skew() > 1.0 or col in [
        "plan_total_cost", "plan_startup_cost", "plan_rows",
        "total_estimated_rows", "total_estimated_bytes",
        "total_actual_rows", "total_actual_time_node",
    ]:
        df[col] = np.log1p(df[col])
        transformed_features.append(col)
        print(f"  ✅ log1p applied: {col}")
    else:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)

# 2. Apply log1p to target
y = np.log1p(df[TARGET_COLUMN].copy())
print(f"  ✅ log1p applied to target: {TARGET_COLUMN}")

# 3. Drop post-execution features from full dataset BEFORE splitting
print(f"\n🧹 Dropping post-execution features BEFORE splitting...")
df = df.drop(columns=[col for col in POST_EXECUTION_FEATURES if col in df.columns])

# 4. Define X after dropping leaky features
X = df.drop(columns=[ID_COLUMN, TARGET_COLUMN])

# 5. Train-test split
print(f"\n🔀 Splitting data into train/test with test size = {TEST_SIZE * 100:.1f}% ...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# 6. Save outputs
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print(f"\n✅ Data split and saved as:")
print(f"- X_train.csv (Shape: {X_train.shape})")
print(f"- X_test.csv  (Shape: {X_test.shape})")
print(f"- y_train.csv (Shape: {y_train.shape})")
print(f"- y_test.csv  (Shape: {y_test.shape})")

print("\n📌 Sample of X_train (head):")
print(X_train.head())
