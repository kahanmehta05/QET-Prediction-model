import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load your CSV
df = pd.read_csv("query_plan_features.csv")

# Drop non-numeric columns for EDA (like query_id)
numeric_df = df.select_dtypes(include=[np.number]).drop(columns=["actual_execution_time"], errors='ignore')

# 1. Histogram for skewed features
skewed_features = ["actual_execution_time", "plan_total_cost", "plan_rows", "total_estimated_rows"]
for col in skewed_features:
    if col in df.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True, log_scale=(True, False))
        plt.title(f"Distribution of {col} (Log Scale)")
        plt.show()

# 2. Boxplot for outliers
for col in skewed_features:
    if col in df.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.show()

# 3. Correlation heatmap (Pearson)
plt.figure(figsize=(16, 12))
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

# 4. Top correlations with the target
if "actual_execution_time" in df.columns:
    corr_target = corr["actual_execution_time"].drop("actual_execution_time").sort_values(ascending=False)
    print("\nTop correlated features with actual_execution_time:\n", corr_target.head(10))
