import random

# Define TPCH schema: table -> column -> data type
tables = {
    "orders": {
        "o_orderkey": "int", "o_custkey": "int", "o_orderstatus": "str", "o_totalprice": "float", "o_orderdate": "str"
    },
    "lineitem": {
        "l_orderkey": "int", "l_partkey": "int", "l_suppkey": "int", "l_quantity": "float", "l_extendedprice": "float", "l_discount": "float"
    },
    "customer": {
        "c_custkey": "int", "c_name": "str", "c_address": "str", "c_nationkey": "int", "c_acctbal": "float"
    },
    "supplier": {
        "s_suppkey": "int", "s_name": "str", "s_address": "str", "s_nationkey": "int"
    },
    "nation": {
        "n_nationkey": "int", "n_name": "str", "n_regionkey": "int"
    },
    "region": {
        "r_regionkey": "int", "r_name": "str"
    },
    "part": {
        "p_partkey": "int", "p_name": "str", "p_type": "str", "p_retailprice": "float"
    }
}

# Valid joins (table1.col1 = table2.col2)
valid_joins = [
    ("orders", "o_custkey", "customer", "c_custkey"),
    ("orders", "o_orderkey", "lineitem", "l_orderkey"),
    ("lineitem", "l_partkey", "part", "p_partkey"),
    ("customer", "c_nationkey", "nation", "n_nationkey"),
    ("supplier", "s_nationkey", "nation", "n_nationkey"),
    ("nation", "n_regionkey", "region", "r_regionkey")
]

def get_numeric_columns(table):
    return [col for col, dtype in tables[table].items() if dtype in ["int", "float"]]

def get_string_columns(table):
    return [col for col, dtype in tables[table].items() if dtype == "str"]

def generate_query():
    used_tables = set()
    joins = []
    selected_columns = []
    where_clauses = []
    
    has_aggregate_column = False
    non_aggregated_selected_columns = [] # This will hold table.col for non-aggregated selected columns

    base_table = random.choice(list(tables.keys()))
    used_tables.add(base_table)

    # Joins - Reduced complexity
    num_joins = random.randint(1, 3) # Changed from 2-5 to 1-3
    while len(joins) < num_joins:
        possible_joins = [j for j in valid_joins if (j[0] in used_tables and j[2] not in used_tables) or (j[2] in used_tables and j[0] not in used_tables)]
        if not possible_joins:
            break
        join = random.choice(possible_joins)
        if join[0] in used_tables and join[2] not in used_tables:
            joins.append(join)
            used_tables.add(join[2])
        elif join[2] in used_tables and join[0] not in used_tables:
            joins.append((join[2], join[3], join[0], join[1]))
            used_tables.add(join[0])


    # Select columns - Reduced complexity
    for table in used_tables:
        col_list = list(tables[table].keys())
        num_cols_to_select = random.randint(1, min(2, len(col_list))) # Changed from 1-3 to 1-2
        selected_cols_for_table = random.sample(col_list, num_cols_to_select)

        for col in selected_cols_for_table:
            dtype = tables[table][col]
            
            # Decide if this specific column will be aggregated - Reduced probability
            should_aggregate_this_column = random.random() < 0.2 # Changed from 0.5 to 0.2

            if dtype in ["int", "float"] and should_aggregate_this_column:
                agg = random.choice(["SUM", "MAX", "AVG"])
                selected_columns.append(f"{agg}({table}.{col}) AS {agg.lower()}_{col}")
                has_aggregate_column = True
            else:
                selected_columns.append(f"{table}.{col} AS {col}_alias")
                non_aggregated_selected_columns.append(f"{table}.{col}") # Add full column name for GROUP BY

    # Ensure selected_columns is not empty (fallback)
    if not selected_columns:
        default_col_name = random.choice(list(tables[base_table].keys()))
        selected_columns.append(f"{base_table}.{default_col_name} AS default_col")
        non_aggregated_selected_columns.append(f"{base_table}.{default_col_name}")


    query = f"SELECT {', '.join(selected_columns)} FROM {base_table}"

    for j in joins:
        # Ensure join order is from used_table to new_table
        if j[0] in used_tables and j[2] not in used_tables:
            query += f" JOIN {j[2]} ON {j[0]}.{j[1]} = {j[2]}.{j[3]}"
        elif j[2] in used_tables and j[0] not in used_tables:
            query += f" JOIN {j[0]} ON {j[2]}.{j[3]} = {j[0]}.{j[1]}"
        else: # Both already used or neither used (shouldn't happen with current logic)
            query += f" JOIN {j[2]} ON {j[0]}.{j[1]} = {j[2]}.{j[3]}"


    # WHERE clause - Reduced probability
    potential_where_cols = []
    for table in used_tables:
        potential_where_cols.extend([f"{table}.{col}" for col in get_numeric_columns(table)])

    if potential_where_cols and random.random() < 0.4: # Changed from 0.7 to 0.4
        num_where_clauses = random.randint(1, min(3, len(potential_where_cols)))
        selected_where_cols = random.sample(potential_where_cols, num_where_clauses)
        for col_full_name in selected_where_cols:
            val = random.randint(10, 100)
            where_clauses.append(f"{col_full_name} > {val}")

    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    # GROUP BY clause - REVISED LOGIC for correctness
    if has_aggregate_column and non_aggregated_selected_columns:
        # Ensure all non-aggregated columns in SELECT are in GROUP BY
        group_by_clause_cols = sorted(list(set(non_aggregated_selected_columns)))
        query += " GROUP BY " + ", ".join(group_by_clause_cols)


    # ORDER BY + LIMIT - adjusted logic
    order_by_col_for_query = None # Use a new variable for the final ORDER BY column in query
    
    if has_aggregate_column and non_aggregated_selected_columns:
        # If there's a GROUP BY, order by a column from the GROUP BY list.
        order_by_col_for_query = random.choice(non_aggregated_selected_columns)
    elif has_aggregate_column and not non_aggregated_selected_columns:
        # If only aggregates are selected (no non-aggregated columns, so no GROUP BY)
        # We need to pick an aggregate from selected_columns to order by
        aggregate_cols_in_select = [col for col in selected_columns if any(agg in col for agg in ["SUM(", "MAX(", "AVG("])]
        if aggregate_cols_in_select:
            order_by_col_for_query = random.choice(aggregate_cols_in_select)
        else: # Fallback, should theoretically not happen if has_aggregate_column is True
            order_by_col_for_query = f"{base_table}.{random.choice(list(tables[base_table].keys()))}"
    else:
        # If no aggregates were selected at all (no GROUP BY needed)
        # Order by any selected non-aggregated column.
        if non_aggregated_selected_columns:
            order_by_col_for_query = random.choice(non_aggregated_selected_columns)
        else: # Fallback, should not happen if selected_columns is populated.
            order_by_col_for_query = f"{base_table}.{random.choice(list(tables[base_table].keys()))}"

    # Remove alias and aggregate function wrappers for ORDER BY if needed
    if 'AS' in order_by_col_for_query:
        order_by_col_for_query = order_by_col_for_query.split(' AS ')[0]
    for agg_func in ["SUM(", "MAX(", "AVG("]:
        if order_by_col_for_query.startswith(agg_func):
            order_by_col_for_query = order_by_col_for_query[len(agg_func):-1]
            break

    query += f" ORDER BY {order_by_col_for_query} DESC LIMIT {random.randint(10, 100)};"

    return query

# Generate 10,000 safe queries
with open("tpch_queries.sql", "w") as f:
    for _ in range(10000):
        f.write(generate_query() + "\n")

print("✅ Safe and valid TPC-H queries saved to less_complex_tpch_queries.sql")
