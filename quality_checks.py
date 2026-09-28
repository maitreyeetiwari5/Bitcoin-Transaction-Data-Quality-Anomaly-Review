"""Data quality and reconciliation checks.

Every check writes one row to an issue log, even when it passes, so the log doubles
as audit evidence that the check was run. Fixes are applied here, in one place, and
each fix is recorded in the 'action_taken' column.
"""
import pandas as pd

import config


class IssueLog:
    def __init__(self):
        self.rows = []

    def add(self, check_id, check, table, records_affected, severity, detail, action):
        self.rows.append({
            "check_id": check_id,
            "check": check,
            "table": table,
            "status": "PASS" if records_affected == 0 else "ISSUE",
            "records_affected": int(records_affected),
            "severity": severity if records_affected else "-",
            "detail": detail,
            "action_taken": action if records_affected else "None needed",
        })

    def to_frame(self):
        return pd.DataFrame(self.rows)


def run_checks(features, classes, edges):
    log = IssueLog()
    raw_counts = {"features": len(features), "classes": len(classes), "edges": len(edges)}

    # --- Structure -------------------------------------------------------------------
    n_cols = features.shape[1]
    log.add("DQ01", "Feature file column count", "features",
            0 if n_cols == config.EXPECTED_FEATURE_COLS else 1, "High",
            f"Expected {config.EXPECTED_FEATURE_COLS} columns, found {n_cols}",
            "Flagged for review; column names assigned by position")

    # --- Features table --------------------------------------------------------------
    missing_keys = features["txId"].isna() | features["time_step"].isna()
    log.add("DQ02", "Missing txId or time_step", "features", missing_keys.sum(), "High",
            "Rows cannot be joined or placed in time without these keys",
            "Rows excluded")
    features = features[~missing_keys].copy()
    features["txId"] = features["txId"].astype("int64")
    features["time_step"] = features["time_step"].astype("int64")

    dup_tx = features["txId"].duplicated(keep="first")
    log.add("DQ03", "Duplicate txId in features", "features", dup_tx.sum(), "High",
            "Same transaction appears more than once",
            "Kept first occurrence, dropped the rest")
    features = features[~dup_tx]

    lo, hi = config.EXPECTED_TIME_STEPS
    bad_step = ~features["time_step"].between(lo, hi)
    log.add("DQ04", "time_step outside expected range", "features", bad_step.sum(), "Medium",
            f"Expected time steps {lo}-{hi}", "Rows excluded")
    features = features[~bad_step]

    feat_cols = [c for c in features.columns if c not in ("txId", "time_step")]
    rows_with_nulls = features[feat_cols].isna().any(axis=1)
    log.add("DQ05", "Missing feature values", "features", rows_with_nulls.sum(), "Low",
            f"{int(features[feat_cols].isna().sum().sum())} null cells across "
            f"{int(rows_with_nulls.sum())} rows",
            "Kept rows; nulls ignored when computing anomaly scores")

    # --- Classes table ---------------------------------------------------------------
    dup_cls = classes["txId"].duplicated(keep="first")
    log.add("DQ06", "Duplicate txId in classes", "classes", dup_cls.sum(), "High",
            "Same transaction labeled more than once", "Kept first label, dropped the rest")
    classes = classes[~dup_cls].copy()

    # fillna first: blank labels must survive as text so they can be reported, not vanish
    raw_class = classes["class"].fillna("<blank>").astype(str)
    normalized = raw_class.str.strip().str.lower()
    invalid = ~normalized.isin(config.VALID_CLASSES)
    log.add("DQ07", "Invalid class label", "classes", invalid.sum(), "Medium",
            "Labels outside {'1','2','unknown'}: "
            + ", ".join(sorted(raw_class[invalid].unique())[:5]),
            "Treated as 'unknown' so they are excluded from evaluation, not guessed")
    classes["class"] = normalized.where(~invalid, "unknown")

    # --- Reconciliation between tables -----------------------------------------------
    feat_ids, cls_ids = set(features["txId"]), set(classes["txId"])
    no_label = feat_ids - cls_ids
    no_features = cls_ids - feat_ids
    log.add("DQ08", "Reconciliation: transactions without a label", "features vs classes",
            len(no_label), "Medium", "In features file but missing from classes file",
            "Assigned 'unknown' label")
    log.add("DQ09", "Reconciliation: labels without a transaction", "classes vs features",
            len(no_features), "Medium", "In classes file but missing from features file",
            "Labels excluded (nothing to score)")
    classes = classes[classes["txId"].isin(feat_ids)]

    # --- Edges table -----------------------------------------------------------------
    orphan = ~edges["txId1"].isin(feat_ids) | ~edges["txId2"].isin(feat_ids)
    log.add("DQ10", "Orphan edges", "edges", orphan.sum(), "High",
            "Edge points to a transaction that is not in the features file",
            "Edges excluded from degree calculations")
    edges = edges[~orphan]

    self_loop = edges["txId1"] == edges["txId2"]
    log.add("DQ11", "Self-loop edges", "edges", self_loop.sum(), "Low",
            "Transaction linked to itself", "Edges excluded")
    edges = edges[~self_loop]

    dup_edge = edges.duplicated(subset=["txId1", "txId2"], keep="first")
    log.add("DQ12", "Duplicate edges", "edges", dup_edge.sum(), "Low",
            "Same payment flow recorded more than once", "Kept first, dropped the rest")
    edges = edges[~dup_edge]

    step = features.set_index("txId")["time_step"]
    cross = step.reindex(edges["txId1"]).values != step.reindex(edges["txId2"]).values
    log.add("DQ13", "Edges spanning two time steps", "edges", int(cross.sum()), "Info",
            "Elliptic documents edges as within a single time step",
            "Kept; flagged for analyst awareness")

    # --- Final reconciliation of record counts ---------------------------------------
    clean_counts = {"features": len(features), "classes": len(classes), "edges": len(edges)}
    recon = pd.DataFrame({
        "table": list(raw_counts),
        "raw_rows": list(raw_counts.values()),
        "clean_rows": [clean_counts[t] for t in raw_counts],
    })
    recon["rows_removed"] = recon["raw_rows"] - recon["clean_rows"]

    labels = features[["txId", "time_step"]].merge(classes, on="txId", how="left")
    labels["class"] = labels["class"].fillna("unknown")
    labels["label"] = labels["class"].map(config.CLASS_LABELS)

    return features, labels[["txId", "time_step", "label"]], edges, log.to_frame(), recon
