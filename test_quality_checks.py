"""Checks the checkers: every planted problem in the synthetic data must be found, with the
exact count, and the SQL tie-out must agree with pandas.

    python scripts/make_synthetic_data.py
    python -m pytest tests/
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import config  # noqa: E402
from make_synthetic_data import PLANTED  # noqa: E402
from src import json_flatten, load, quality_checks, sql_checks  # noqa: E402

RAW = ROOT / "data" / "synthetic" / "raw"
EXPECTED = {
    "DQ02": PLANTED["missing_time_step"],
    "DQ03": PLANTED["duplicate_txid_in_features"],
    "DQ04": PLANTED["time_step_out_of_range"],
    "DQ05": PLANTED["rows_with_null_features"],
    "DQ06": PLANTED["duplicate_class_rows"],
    "DQ07": PLANTED["invalid_class_labels"],
    "DQ08": PLANTED["transactions_without_label"],
    "DQ09": PLANTED["labels_without_transaction"],
    "DQ10": PLANTED["orphan_edges"],
    "DQ11": PLANTED["self_loops"],
    "DQ12": PLANTED["duplicate_edges"],
}


def _run():
    raw = load.load_raw(RAW)
    return raw, quality_checks.run_checks(*raw)


def test_planted_issues_are_found_exactly():
    _, (_, _, _, log, _) = _run()
    found = log.set_index("check_id")["records_affected"].to_dict()
    for check_id, expected in EXPECTED.items():
        assert found[check_id] == expected, f"{check_id}: expected {expected}, got {found[check_id]}"


def test_sql_ties_out_with_pandas(tmp_path):
    raw, (_, _, _, log, _) = _run()
    _, tie_out = sql_checks.run_sql_checks(*raw, log, db_path=tmp_path / "t.db")
    assert (tie_out["tie_out"] == "MATCH").all(), tie_out


def test_json_fee_breaks_found():
    txs = json_flatten.load_json_files(ROOT / "data" / "synthetic" / "json")
    tx, ins, outs = json_flatten.flatten(txs)
    rec = json_flatten.reconcile_fees(tx, ins, outs)
    assert (rec["result"] == "Break").sum() == PLANTED["json_fee_breaks"]
