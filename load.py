"""Load the three Elliptic files into DataFrames with readable column names.

Loading is deliberately permissive: nothing is dropped or fixed here. Every problem
is left in place so the quality checks can find it, log it, and fix it explicitly.
"""
import pandas as pd

import config


def feature_column_names(n_cols: int) -> list[str]:
    """Elliptic's features file has no header. Column 0 is txId, column 1 is the time step,
    the next 93 are local features, and the remaining 72 are aggregated neighbor features."""
    names = ["txId", "time_step"]
    n_features = n_cols - 2
    n_local = min(93, n_features)
    names += [f"local_feat_{i}" for i in range(1, n_local + 1)]
    names += [f"agg_feat_{i}" for i in range(1, n_features - n_local + 1)]
    return names


def load_raw(raw_dir=config.RAW_DIR):
    features = pd.read_csv(raw_dir / config.FEATURES_FILE, header=None)
    features.columns = feature_column_names(features.shape[1])

    # Read classes as strings so unexpected values ('3', 'Unknown', blanks) survive for checking
    classes = pd.read_csv(raw_dir / config.CLASSES_FILE, dtype={"class": str})
    edges = pd.read_csv(raw_dir / config.EDGES_FILE)
    return features, classes, edges
