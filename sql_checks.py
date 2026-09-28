"""Load the raw tables into SQLite, run the SQL reconciliation queries, and tie the
SQL results out against the pandas issue log. Two independent methods agreeing is the
evidence that the quality checks themselves are right."""
import sqlite3
from pathlib import Path

import pandas as pd

import config

SQL_FILE = Path(__file__).parent / "sql" / "reconciliation_checks.sql"

# SQL query -> pandas check_id it should agree with
TIE_OUT_MAP = {
    "duplicate_txid_in_features": "DQ03",
    "transactions_without_label": "DQ08",
    "labels_without_transaction": "DQ09",
    "orphan_edges": "DQ10",
}


def parse_queries(sql_text: str) -> dict[str, str]:
    queries, name, buf = {}, None, []
    for line in sql_text.splitlines():
        if line.startswith("-- name:"):
            if name:
                queries[name] = "\n".join(buf).strip()
            name, buf = line.split(":", 1)[1].strip(), []
        elif name and not line.startswith("--"):
            buf.append(line)
    if name:
        queries[name] = "\n".join(buf).strip()
    return queries


def run_sql_checks(features_raw, classes_raw, edges_raw, issue_log, db_path=config.DB_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    # Only the key columns are needed for reconciliation; keeps the database small
    features_raw[["txId", "time_step"]].to_sql("features_raw", con, index=False)
    classes_raw.to_sql("classes_raw", con, index=False)
    edges_raw.to_sql("edges_raw", con, index=False)

    queries = parse_queries(SQL_FILE.read_text())
    results = {}
    for name, sql in queries.items():
        if sql.upper().startswith("CREATE"):
            con.execute(sql)
        else:
            results[name] = pd.read_sql_query(sql, con)
    con.close()

    pandas_counts = issue_log.set_index("check_id")["records_affected"]
    rows = []
    for query, check_id in TIE_OUT_MAP.items():
        sql_value = int(results[query].iloc[0, 0])
        pandas_value = int(pandas_counts.get(check_id, -1))
        rows.append({
            "check": query,
            "pandas_check_id": check_id,
            "sql_result": sql_value,
            "pandas_result": pandas_value,
            "tie_out": "MATCH" if sql_value == pandas_value else "MISMATCH - investigate",
        })
    return results, pd.DataFrame(rows)
