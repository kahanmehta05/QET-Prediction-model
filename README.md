# QET Prediction Model

This repository contains the full pipeline for predicting SQL query execution time (QET) using only **pre-execution plan features**, avoiding post-execution data leakage. It was developed as a **minor research project under Prof. Amit Mankodi at DAU (DAIICT)**.

## Project Overview

Modern databases often struggle with predicting query runtimes accurately. Traditional optimizers frequently make mistakes due to cardinality and cost estimation errors. This project builds a **machine learning-based model** that:
- Uses **pre-execution features only**
- Achieves **R² > 0.93**
- Predicts QET accurately before running the query
- Aids in future optimizer feedback and anomaly detection systems

## Why This Research Is Important

- Existing work often includes post-execution features (actual rows, timings), which are unavailable before runtime — causing **data leakage** and unrealistic models.
- We designed a **clean feature set** of 33 purely structural + cost-based features.
- Demonstrates that **pre-execution signals alone** can achieve high accuracy if engineered and modeled well.
  
✅ Final Pre-Execution Feature List  
**Plan-level cost & size**  
`query_planning_time`, `plan_total_cost`, `plan_startup_cost`, `plan_rows`, `plan_width`  

**Structural complexity**  
`max_depth`, `num_operators`  

**Execution characteristics**  
`has_sort`, `has_hash`, `is_parallel_aware`, `max_workers_planned`, `max_workers_launched`  

**Operator/node type counts**  
`node_type_Seq_Scan_count`, `node_type_Hash_Join_count`, `node_type_Merge_Join_count`,  
`node_type_Nested_Loop_count`, `node_type_Aggregate_count`, `node_type_Sort_count`,  
`node_type_Limit_count`, `node_type_Materialize_count`, `node_type_Gather_count`,  
`node_type_Gather_Merge_count`, `node_type_OTHER_count`  

**Estimates**  
`total_estimated_rows`, `total_estimated_bytes`  

**Query-level metadata**  
`query_length_chars`, `has_group_by`, `has_order_by`, `has_limit`,  
`num_aggregate_functions`, `num_joins`, `num_where_conditions`, `num_tables_referenced`  

---

The derived features used in this research are:

- **Aggregate estimates**: `total_estimated_rows`, `total_estimated_bytes` (summed over all operators).  
- **Structural complexity indicators**: `max_depth`, `num_operators` (from plan tree traversal).  
- **Execution characteristics**: `has_sort`, `has_hash`, `is_parallel_aware`, `max_workers_planned`, `max_workers_launched`.  
- **Query-level metadata**: `num_joins`, `num_tables_referenced`, `num_where_conditions`, `has_group_by`, `has_order_by`, `has_limit`, `num_aggregate_functions` (parsed from SQL).  

---

## 📁 Project Structure & File Descriptions

| File Name                            | Description                                                                 |
|-------------------------------------|-----------------------------------------------------------------------------|
| `queryDataset.csv`                  | 10K+ real TPCH-based queries                                                |
| `extract_features.py`               | Parses PostgreSQL JSON plans into features                                  |
| `query_plan_features.csv`           | Final dataset with engineered features                                      |
| `split_data_log_transform.py`       | Applies log1p transforms and splits into train/test CSVs                    |
| `X_train.csv`, `X_test.csv`         | Feature matrices (post-cleaning)                                            |
| `y_train.csv`, `y_test.csv`         | Target runtime values                                                       |
| `final_model_training.py`           | Trains RF, XGBoost, GB models using tuned parameters and pre-exec features  |
| `baseline_final_model_training.py`  | Same as above but uses **default model hyperparameters**                    |
| `baseline_final_model_training_direct_features.py` | Trains models using **only the 5 direct EXPLAIN features**                |
| `hyperparameter_tuning.py`          | Grid search to tune models                                                  |
| `shap_analysis.py`                  | SHAP interpretability plots + top feature attribution                       |
| `feature_correlation_analysis.py`   | Correlation heatmap and top predictors vs QET                               |
| `generate_tpch_queries.py`          | Script to generate diverse TPCH queries                                     |
| `tpch_queries.sql`                  | Sample TPCH-style queries                                                   |
| `result_images/`                    | Folder containing output screenshots and graphs                             |

---

## Full Execution Pipeline

1. **Query Generation**:  
   Use `generate_tpch_queries.py` to generate 10,000+ diverse SQL queries.

2. **Feature Extraction**:  
   Use `extract_features.py` to get features from PostgreSQL's `EXPLAIN (ANALYZE, FORMAT JSON)` output.

3. **Feature Cleaning + Splitting**:  
   Run `split_data_log_transform.py` to apply `log1p()` transform, drop post-exec features, and split into `train/test`.

4. **Model Training & Evaluation**:  
   Use:
   - `final_model_training.py` → trains models using tuned parameters.
   - `baseline_final_model_training.py` → trains with default parameters for comparison.
   - `baseline_final_model_training_direct_features.py` → trains models **only on the 5 direct EXPLAIN features**.

5. **Interpretability & Validation**:  
   Run:
   - `shap_analysis.py` to get top impactful features.
   - `feature_correlation_analysis.py` to inspect skew, correlation, and leakage risk.

---

## 🔎 Training & Evaluation (Direct Features Only)

This experiment isolates the **5 direct EXPLAIN features**:  
`query_planning_time`, `plan_total_cost`, `plan_startup_cost`, `plan_rows`, `plan_width`.

```bash
kahanmehta@kahans-MacBook-Air qet_feature_extractor % python3 baseline_final_model_training_direct_features.py
✅ Loaded data shapes:
 - X_train: (8062, 5)
 - X_test : (2016, 5)
 - y_train: (8062,)
 - y_test : (2016,)

🚀 Training RandomForestRegressor...

📊 Random Forest (Default) Evaluation:
 - RMSE (original units): 95,337.0933
 - MAE  (original units): 10,782.1222
 - R²  (log-space)      : 0.9027

📌 Random Forest Feature importances:
 - plan_total_cost: 0.809890
 - plan_startup_cost: 0.121447
 - plan_rows: 0.035713
 - query_planning_time: 0.017562
 - plan_width: 0.015387

🚀 Training XGBRegressor...

📊 XGBoost (Default) Evaluation:
 - RMSE (original units): 95,666.8800
 - MAE  (original units): 10,851.0421
 - R²  (log-space)      : 0.8940

📌 XGBoost Feature importances:
 - plan_total_cost: 0.800214
 - plan_startup_cost: 0.122681
 - plan_rows: 0.046132
 - plan_width: 0.016204
 - query_planning_time: 0.014769

🚀 Training GradientBoostingRegressor...

📊 Gradient Boosting (Default) Evaluation:
 - RMSE (original units): 95,669.0947
 - MAE  (original units): 10,671.0248
 - R²  (log-space)      : 0.8873

📌 Gradient Boosting Feature importances:
 - plan_total_cost: 0.840665
 - plan_startup_cost: 0.128554
 - plan_rows: 0.024834
 - query_planning_time: 0.003111
 - plan_width: 0.002837
