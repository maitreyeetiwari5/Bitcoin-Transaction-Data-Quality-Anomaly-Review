"""Robust anomaly flags with a time-based split, honest evaluation, and escalation.

Method
  1. Robust z-score for every feature within each time step:
         z = 0.6745 * (x - median) / MAD
     The median and MAD keep a few extreme transactions from distorting the baseline.
  2. Feature selection uses ONLY the training period (time steps 1-34). Each feature is
     tested in both directions (unusually high, unusually low), and the most precise
     ones are kept, skipping near-duplicates.
  3. A transaction is flagged when enough selected features breach the threshold.
  4. Results are reported on the holdout period (time steps 35-49), which played no
     part in choosing the features. Training-period results are shown for comparison.

Design rules
  * Flags are scored against known labels only. Unknown transactions are never counted
    as hits or misses, because nobody knows what they are.
  * Flagged unknown transactions go to a review queue. The pipeline never labels them.
"""
import numpy as np
import pandas as pd

import config


def robust_z(values: pd.Series) -> pd.Series:
    median = values.median()
    mad = (values - median).abs().median()
    if mad > 0:
        return 0.6745 * (values - median) / mad
    # MAD is zero when over half the values are identical (common for degree counts).
    # Fall back to the mean absolute deviation, scaled to be comparable (factor 1.2533).
    mean_ad = (values - median).abs().mean()
    if mean_ad > 0:
        return (values - median) / (1.2533 * mean_ad)
    return pd.Series(0.0, index=values.index)


def candidate_features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c.startswith(("local_feat_", "agg_feat_")) or c in ("in_degree", "out_degree")]


def select_features(df: pd.DataFrame, z: pd.DataFrame):
    """Pick features and directions using labeled transactions in the training period."""
    train = (df["time_step"] <= config.TRAIN_MAX_TIME_STEP) & df["label"].isin(["illicit", "licit"])
    illicit = df["label"] == "illicit"
    n_illicit = int((train & illicit).sum())

    rows = []
    for feat in z.columns:
        for direction, sign in (("high", 1), ("low", -1)):
            hit = train & (sign * z[feat] > config.MAD_THRESHOLD)
            flagged = int(hit.sum())
            if flagged < config.MIN_TRAIN_FLAGS or n_illicit == 0:
                continue
            tp = int((hit & illicit).sum())
            rows.append((feat, direction, flagged, tp, tp / flagged, tp / n_illicit))
    ranked = pd.DataFrame(rows, columns=["feature", "direction", "train_flagged",
                                         "train_true_positives", "train_precision",
                                         "train_recall"])
    ranked = ranked[ranked["train_recall"] >= config.MIN_TRAIN_RECALL]
    ranked = ranked.sort_values("train_precision", ascending=False)

    chosen = []
    for _, row in ranked.iterrows():
        duplicate = any(
            abs(np.corrcoef(z.loc[train, row["feature"]], z.loc[train, c["feature"]])[0, 1])
            > config.MAX_CORRELATION for c in chosen)
        if not duplicate and row["feature"] not in [c["feature"] for c in chosen]:
            chosen.append(row.to_dict())
        if len(chosen) == config.N_SELECTED_FEATURES:
            break

    selected = pd.DataFrame(chosen)
    if len(selected):
        selected[["train_precision", "train_recall"]] = (
            selected[["train_precision", "train_recall"]].round(4))
    return selected


def score(df: pd.DataFrame):
    feats = candidate_features(df)
    z = df.groupby("time_step")[feats].transform(robust_z)
    selected = select_features(df, z)

    breached = pd.DataFrame(index=df.index)
    for _, row in selected.iterrows():
        sign = 1 if row["direction"] == "high" else -1
        breached[row["feature"]] = sign * z[row["feature"]]
    df["features_breached"] = (breached > config.MAD_THRESHOLD).sum(axis=1)
    df["anomaly_score"] = breached.max(axis=1).round(2) if len(selected) else 0.0
    df["top_driver"] = breached.idxmax(axis=1) if len(selected) else ""
    df["flagged"] = df["features_breached"] >= config.MIN_FEATURES_BREACHED
    df["period"] = np.where(df["time_step"] <= config.TRAIN_MAX_TIME_STEP,
                            "train (1-34)", "holdout (35-49)")
    return df, selected


def _metrics(part: pd.DataFrame):
    is_illicit = part["label"] == "illicit"
    tp = int((part["flagged"] & is_illicit).sum())
    fp = int((part["flagged"] & ~is_illicit).sum())
    fn = int((~part["flagged"] & is_illicit).sum())
    baseline = is_illicit.mean() if len(part) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    lift = precision / baseline if baseline and not np.isnan(precision) else np.nan
    return len(part), int(is_illicit.sum()), tp + fp, tp, fp, fn, baseline, precision, recall, lift


def evaluate(df: pd.DataFrame):
    labeled = df[df["label"].isin(["illicit", "licit"])]
    names = ["Labeled transactions evaluated", "Illicit transactions (known)",
             "Flagged among labeled", "True positives (flagged and illicit)",
             "False positives (flagged but licit)", "Missed illicit (not flagged)",
             "Baseline illicit rate", "Precision", "Recall", "Lift over baseline"]
    out = {"metric": names}
    for period in ("holdout (35-49)", "train (1-34)"):
        vals = _metrics(labeled[labeled["period"] == period])
        out[period] = [round(v, 4) if isinstance(v, float) else v for v in vals]
    overall = pd.DataFrame({k: pd.Series(v, dtype=object) for k, v in out.items()})

    by_step = (labeled.assign(illicit=labeled["label"] == "illicit")
               .groupby(["time_step", "period"])
               .agg(labeled=("label", "size"), illicit=("illicit", "sum"),
                    flagged=("flagged", "sum"),
                    true_positives=("illicit", lambda s: int((s & labeled.loc[s.index, "flagged"]).sum())))
               .reset_index())
    by_step["precision"] = (by_step["true_positives"] / by_step["flagged"].replace(0, np.nan)).round(4)
    by_step["recall"] = (by_step["true_positives"] / by_step["illicit"].replace(0, np.nan)).round(4)
    return overall, by_step


def review_queue(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["txId", "time_step", "anomaly_score", "top_driver", "features_breached",
            "in_degree", "out_degree"]
    queue = df[df["flagged"] & (df["label"] == "unknown")][cols]
    queue = queue.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
    queue["status"] = "Pending analyst review"
    return queue


def flagged_labeled(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["txId", "time_step", "period", "label", "anomaly_score", "top_driver",
            "features_breached", "in_degree", "out_degree"]
    out = df[df["flagged"] & df["label"].isin(["illicit", "licit"])][cols]
    return out.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
