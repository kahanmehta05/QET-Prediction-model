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

---

## 📁 Project Structure & File Descriptions

| File Name                       | Description                                                                 |
|--------------------------------|-----------------------------------------------------------------------------|
| `queryDataset.csv`             | 10K+ real TPCH-based queries                                                |
| `extract_features.py`          | Parses PostgreSQL JSON plans into features                                  |
| `query_plan_features.csv`      | Final dataset with engineered features                                      |
| `split_data_log_transform.py`  | Applies log1p transforms and splits into train/test CSVs                    |
| `X_train.csv`, `X_test.csv`    | Feature matrices (post-cleaning)                                            |
| `y_train.csv`, `y_test.csv`    | Target runtime values                                                       |
| `final_model_training.py`      | Trains RF, XGBoost, GB models using tuned parameters and pre-exec features  |
| `baseline_final_model_training.py` | Same as above but uses **default model hyperparameters**               |
| `hyperparameter_tuning.py`     | Grid search to tune models                                                  |
| `shap_analysis.py`             | SHAP interpretability plots + top feature attribution                       |
| `feature_correlation_analysis.py` | Correlation heatmap and top predictors vs QET                           |
| `generate_tpch_queries.py`     | Script to generate diverse TPCH queries                                     |
| `tpch_queries.sql`             | Sample TPCH-style queries                                                   |
| `result_images/`               | Folder containing output screenshots and graphs                             |

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

5. **Interpretability & Validation**:  
   Run:
   - `shap_analysis.py` to get top impactful features.
   - `feature_correlation_analysis.py` to inspect skew, correlation, and leakage risk.

---

## Key Achievements

- **R² Score ~ 0.93** using only pre-execution features
- Removed post-execution metrics (e.g., actual rows, loops, time) to prevent data leakage
- Identified top impactful features via SHAP:  
  `plan_total_cost`, `plan_startup_cost`, `estimated_rows`, `query_length`, etc.
- Designed interpretable, portable models using Random Forest, XGBoost, and Gradient Boosting

---

## Future Directions

- Predict **optimizer mistakes** or **plan regressions** using this model
- Integrate into PostgreSQL or external advisor to suggest better join orders
- Extend to **cost estimation correction** or **adaptive query tuning**
- Train on **cross-database workloads** for generalization

---

## 🧾 Citation / Credits

This work was done as part of **Research Project** under the guidance of **Prof. Amit Mankodi**,  
Department of ICT, Dhirubhai Ambani Institute of Information and Communication Technology (DAU / DAIICT), Gujarat, India.

---

##  How to Run

```bash
# Recommended: Use a virtualenv or conda env
pip install -r requirements.txt

# 1. Feature Extraction (if needed)
python extract_features.py

# 2. Clean & Split
python split_data_log_transform.py

# 3. Model Training
python final_model_training.py
