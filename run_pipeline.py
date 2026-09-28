"""Run the full pipeline.

    python run_pipeline.py                 # Elliptic data only (JSON step uses any saved files)
    python run_pipeline.py --fetch-json    # also pull fresh transactions from the public API
    python run_pipeline.py --raw-dir path  # point at a different folder of Elliptic files
    python run_pipeline.py --raw-dir data/synthetic/raw --json-dir data/synthetic/json
                                           # smoke test on synthetic data (see scripts/)
"""
import argparse
from pathlib import Path

import config
from src import anomaly, features, json_flatten, load, quality_checks, report, sql_checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument("--json-dir", type=Path, default=config.JSON_DIR)
    parser.add_argument("--fetch-json", action="store_true",
                        help="Download recent transactions from the Blockstream Esplora API")
    args = parser.parse_args()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("1/6 Loading Elliptic files...")
    feats_raw, classes_raw, edges_raw = load.load_raw(args.raw_dir)
    print(f"    {len(feats_raw):,} transactions | {len(classes_raw):,} labels | "
          f"{len(edges_raw):,} edges")

    print("2/6 Running data quality and reconciliation checks...")
    feats, labels, edges, issue_log, recon = quality_checks.run_checks(
        feats_raw, classes_raw, edges_raw)
    issues = issue_log[issue_log["status"] == "ISSUE"]
    print(f"    {len(issue_log)} checks run, {len(issues)} with issues")

    print("3/6 Tying out pandas checks with independent SQL queries...")
    sql_results, tie_out = sql_checks.run_sql_checks(
        feats_raw, classes_raw, edges_raw, issue_log)
    print(f"    {(tie_out['tie_out'] == 'MATCH').sum()} of {len(tie_out)} tie-outs match")

    print("4/6 Scoring anomalies and evaluating against known labels...")
    df = features.add_degree_features(feats, edges).merge(
        labels[["txId", "label"]], on="txId", how="left")
    df, selected = anomaly.score(df)
    overall, by_step = anomaly.evaluate(df)
    queue = anomaly.review_queue(df)
    flagged = anomaly.flagged_labeled(df)
    if selected.empty:
        print("    WARNING: no feature met the selection rules in config.py; nothing flagged")
    m = dict(zip(overall["metric"], overall["holdout (35-49)"]))
    print(f"    Features chosen on time steps 1-34: "
          + ", ".join(f"{r.feature} ({r.direction})" for r in selected.itertuples()))
    print(f"    Holdout (35-49): precision {m['Precision']:.1%} vs {m['Baseline illicit rate']:.1%} "
          f"baseline | recall {m['Recall']:.1%} | {len(queue):,} unlabeled flags escalated")

    print("5/6 Processing raw JSON transactions...")
    json_tables = None
    if args.fetch_json:
        try:
            path, _ = json_flatten.fetch_latest_block()
            print(f"    Saved {path}")
        except Exception as exc:  # network problems should not stop the rest of the report
            print(f"    Could not fetch JSON ({exc}). Continuing with any saved files.")
    txs = json_flatten.load_json_files(args.json_dir)
    if txs:
        tx_df, in_df, out_df = json_flatten.flatten(txs)
        fee_rec = json_flatten.reconcile_fees(tx_df, in_df, out_df)
        json_summary = json_flatten.json_quality_summary(tx_df, in_df, out_df, fee_rec)
        json_tables = {"JSON Summary": json_summary, "Fee Reconciliation": fee_rec}
        tx_df.to_csv(config.OUTPUT_DIR / "json_transactions.csv", index=False)
        in_df.to_csv(config.OUTPUT_DIR / "json_inputs.csv", index=False)
        out_df.to_csv(config.OUTPUT_DIR / "json_outputs.csv", index=False)
        breaks = (fee_rec["result"] == "Break").sum()
        print(f"    {len(tx_df):,} transactions flattened | {breaks} fee breaks")
    else:
        print("    No JSON files found in data/json (run with --fetch-json). Skipping.")

    print("6/6 Writing Excel exception report and Tableau extracts...")
    sheets = {
        "Data Quality Log": issue_log,
        "Record Reconciliation": recon,
        "SQL Tie-Out": tie_out,
        "Model Evaluation": overall,
        "Selected Features": selected,
        "By Time Step": by_step,
        "Review Queue": queue,
        "Flagged Labeled": flagged,
        "Class Distribution (SQL)": sql_results["class_distribution"],
    }
    xlsx = report.write_report(sheets, json_tables)
    csv_dir = report.export_csvs(df, by_step, issue_log)
    print(f"    Report: {xlsx}\n    Tableau CSVs: {csv_dir}")


if __name__ == "__main__":
    main()
