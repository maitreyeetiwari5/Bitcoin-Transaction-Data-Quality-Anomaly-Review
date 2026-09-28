"""Build interpretable graph features from the edge list.

Most Elliptic features are anonymized, so their meaning is unknown. Degree features
are different: in_degree is how many earlier transactions paid into this one, and
out_degree is how many later transactions it paid out to. They can be explained to
an investigator in one sentence, which is why they lead the anomaly features.
"""
import pandas as pd


def add_degree_features(features: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    out_deg = edges.groupby("txId1").size().rename("out_degree")
    in_deg = edges.groupby("txId2").size().rename("in_degree")
    df = features.merge(in_deg, left_on="txId", right_index=True, how="left")
    df = df.merge(out_deg, left_on="txId", right_index=True, how="left")
    df[["in_degree", "out_degree"]] = df[["in_degree", "out_degree"]].fillna(0).astype(int)
    return df
