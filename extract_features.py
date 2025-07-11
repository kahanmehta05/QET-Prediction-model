import re
import pandas as pd
import json
from collections import defaultdict

# --- Configuration ---
# Define common node types for consistent columns, even if not present in every query
COMMON_NODE_TYPES = [
    "Seq Scan", "Index Scan", "Bitmap Heap Scan", "Bitmap Index Scan",
    "Hash Join", "Merge Join", "Nested Loop", "Aggregate", "HashAggregate",
    "Sort", "Limit", "Materialize", "Subquery Scan", "Append",
    "Gather", "Gather Merge", "Unique", "Result", "CTE Scan",
    "Foreign Scan", "Custom Scan", "WindowAgg"
]

AGG_FUNCS = ['sum', 'count', 'avg', 'min', 'max']
WINDOW_FUNCS = ['row_number', 'rank', 'dense_rank', 'ntile', 'lead', 'lag']

# --- Helper Functions ---

def safe_divide(numerator, denominator):
    """Safely divides, handling zero denominators."""
    return numerator / denominator if denominator != 0 else 0

def walk_plan_tree(node, depth=0, features=None):
    """
    Recursively walks the plan tree to extract and accumulate features.
    Initializes features as defaultdict(float) for easy accumulation.
    """
    if features is None:
        features = defaultdict(float)
        # Initialize some boolean/max features explicitly if they might not be set
        features["max_depth"] = 0
        features["num_operators"] = 0
        features["has_sort"] = 0
        features["has_hash"] = 0
        features["is_parallel_aware"] = 0
        features["max_workers_planned"] = 0
        features["max_workers_launched"] = 0

        # Initialize counts for all COMMON_NODE_TYPES
        for nt in COMMON_NODE_TYPES:
            features[f"node_type_{nt}_count"] = 0
        features["node_type_OTHER_count"] = 0 # For types not in COMMON_NODE_TYPES

    features["num_operators"] += 1
    features["max_depth"] = max(features["max_depth"], depth)

    node_type = node.get("Node Type", "OTHER")
    if node_type in COMMON_NODE_TYPES:
        features[f"node_type_{node_type}_count"] += 1
    else:
        features["node_type_OTHER_count"] += 1


    # --- Pre-Execution Node-level Accumulations ---
    features["total_estimated_rows"] += node.get("Plan Rows", 0)
    features["total_estimated_bytes"] += node.get("Plan Rows", 0) * node.get("Plan Width", 0)

    if node.get("Parallel Aware") is True:
        features["is_parallel_aware"] = 1
    if node_type == "Sort": # Exact match for Sort
        features["has_sort"] = 1
    if node_type in ["Hash", "Hash Join", "HashAggregate"]: # Specific types for Hash
        features["has_hash"] = 1

    features["max_workers_planned"] = max(features["max_workers_planned"], node.get("Workers Planned", 0))
    features["max_workers_launched"] = max(features["max_workers_launched"], node.get("Workers Launched", 0))

    # --- Post-Execution Node-level Accumulations (excluding Buffer/IO) ---
    features["total_actual_rows"] += node.get("Actual Rows", 0)
    features["total_actual_loops"] += node.get("Actual Loops", 1) # Actual Loops can be 0 or more
    features["total_actual_time_node"] += node.get("Actual Total Time", 0) * node.get("Actual Loops", 1)

    # Recurse into subplans
    for subnode in node.get("Plans", []):
        walk_plan_tree(subnode, depth + 1, features)

    return features

def parse_query_text_features(query_text):
    """
    Extracts features directly from the SQL query text.
    """
    text = query_text.lower()
    features = defaultdict(float) # Use defaultdict for text features too

    features["query_length_chars"] = len(query_text)

    # Keyword presence/counts
    features["has_group_by"] = int("group by" in text)
    features["has_order_by"] = int("order by" in text)
    features["has_distinct"] = int("distinct" in text)
    features["has_limit"] = int("limit" in text)
    features["has_union"] = int("union " in text or "union all" in text) # Space after union
    features["has_subquery_or_cte"] = int(" with " in text or "select (" in text) # Basic check for subquery/CTE

    # Function counts
    features["num_aggregate_functions"] = sum(text.count(f"{func}(") for func in AGG_FUNCS)
    features["num_window_functions"] = sum(text.count(f"{func}(") for func in WINDOW_FUNCS)

    # Basic join/table counts (can be improved with SQL parsing libraries)
    features["num_joins"] = text.count(" join ") # Simple count
    # This is a very rough estimate, consider using a proper SQL parser for accuracy
    features["num_tables_referenced"] = len(re.findall(r'from\s+([a-zA-Z0-9_.\"`]+)', text)) \
                                        + len(re.findall(r'join\s+([a-zA-Z0-9_.\"`]+)', text))

    features["num_where_conditions"] = len(re.findall(r'\b(?:and|or)\b', text)) # Count logical operators

    return features


def extract_features(row):
    """
    Main function to extract all features for a single query row.
    """
    try:
        plan_data = json.loads(row["plan_json"])
        plan = plan_data[0] if isinstance(plan_data, list) else plan_data # Handle list wrapper
        root = plan.get("Plan", plan) # Get the root plan node

        tree_features = walk_plan_tree(root)

        # Top-level features from the plan JSON (mostly pre-execution or target)
        top_features = {
            "query_planning_time": plan.get("Planning Time", 0),
            "plan_total_cost": root.get("Total Cost", 0),
            "plan_startup_cost": root.get("Startup Cost", 0),
            "plan_rows": root.get("Plan Rows", 0), # Root node estimated rows
            "plan_width": root.get("Plan Width", 0),
            "actual_execution_time": plan.get("Execution Time", 0), # THIS IS YOUR TARGET VARIABLE
            "output_rows": root.get("Actual Rows", 0), # Actual rows returned by root node
        }

        # --- Discrepancy Features (calculated during training only, using pre & post) ---
        total_estimated = tree_features["total_estimated_rows"]
        total_actual = tree_features["total_actual_rows"]
        output_rows = top_features["output_rows"] # Actual rows from root
        plan_rows = top_features["plan_rows"] # Estimated rows from root

        discrepancy_features = {
            "rows_estimate_error": total_actual - total_estimated,
            "rows_estimate_ratio": safe_divide(total_actual, total_estimated),
            "output_row_estimation_error": output_rows - plan_rows,
            "output_row_estimation_ratio": safe_divide(output_rows, plan_rows),
            "cost_time_ratio": safe_divide(top_features["actual_execution_time"], top_features["plan_total_cost"]),
            "sum_node_actual_time_vs_total_actual_time_ratio": safe_divide(tree_features["total_actual_time_node"], top_features["actual_execution_time"]),
        }

        # SQL Text Features (Pre-execution)
        sql_features = {}
        # Changed from row.get("query_text") to row.get("sql")
        if pd.notna(row.get("sql")): # Check if 'sql' column exists and is not NaN
            sql_features = parse_query_text_features(row["sql"])

        # Combine all features
        all_feats = {
            "query_id": row.get("query_id", ""), # Ensure query_id is handled
            **top_features,
            **tree_features,
            **discrepancy_features,
            **sql_features
        }
        return all_feats
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON for row {row.get('query_id', '')}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred for row {row.get('query_id', '')}: {e}")
        return None

# --- Main Script Execution ---
if __name__ == "__main__":
    # Load the query dataset
    try:
        df = pd.read_csv("queryDataset.csv")
        # Validate essential columns
        if "plan_json" not in df.columns:
            raise ValueError("queryDataset.csv must contain 'plan_json' column.")
        
        # Changed column check from 'query_text' to 'sql'
        if "sql" not in df.columns:
            print("Warning: 'sql' column not found. SQL text features will be skipped.")
            df["sql"] = None # Add a dummy column to avoid errors

    except FileNotFoundError:
        print("Error: queryDataset.csv not found. Please ensure it's in the same directory.")
        exit()
    except Exception as e:
        print(f"Error loading queryDataset.csv: {e}")
        exit()

    records = []
    for i, row in df.iterrows():
        # Ensure 'plan_json' is a string before passing to json.loads, and handle potential NaN
        plan_json_str = str(row["plan_json"]) if pd.notna(row["plan_json"]) else "{}" # Default to empty JSON if NaN
        
        # Create a temporary row dict to pass to extract_features, including query_id and 'sql'
        temp_row = {
            "plan_json": plan_json_str,
            "query_id": row.get("query_id", i), # Use existing query_id or index
            "sql": row.get("sql") # Pass 'sql' if exists
        }

        features = extract_features(temp_row)
        if features:
            records.append(features)

    # Create DataFrame
    final_df = pd.DataFrame(records)

    # Fill any remaining NaNs with 0 (e.g., if a feature was never encountered)
    final_df.fillna(0, inplace=True)

    # Define your target column name
    target_col = 'actual_execution_time' 

    # Remove all-zero columns (excluding 'query_id' and the target variable)
    zero_cols = [col for col in final_df.columns if col != 'query_id' and col != target_col and final_df[col].sum() == 0]
    if zero_cols:
        print(f"Removed all-zero columns: {zero_cols}")
        final_df.drop(columns=zero_cols, inplace=True)

    # Ensure 'query_id' is the first column and target is the last
    cols = [col for col in final_df.columns if col != 'query_id' and col != target_col]
    final_df = final_df[['query_id'] + cols + [target_col]]

    final_df.to_csv("query_plan_features.csv", index=False)
    print("✅ Final CSV saved with", len(final_df.columns), "columns and", len(final_df), "rows.")

    print("\nGenerated DataFrame Head:")
    print(final_df.head())
    print("\nGenerated DataFrame Info:")
    print(final_df.info())
