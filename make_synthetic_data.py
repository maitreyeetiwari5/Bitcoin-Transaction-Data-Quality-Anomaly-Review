"""Generate small synthetic files in the exact Elliptic and Esplora formats, with known
data problems planted in them. Used only to smoke-test the pipeline: if the checks find
exactly the problems planted here, they work. NOT for any reported results.

    python scripts/make_synthetic_data.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_RAW = ROOT / "data" / "synthetic" / "raw"
OUT_JSON = ROOT / "data" / "synthetic" / "json"

rng = np.random.default_rng(42)
N_TX, N_STEPS, N_FEATURES = 6000, 49, 165

# Planted issues (the pipeline should report these exact counts)
PLANTED = {
    "duplicate_txid_in_features": 5,
    "missing_time_step": 3,
    "time_step_out_of_range": 2,
    "rows_with_null_features": 4,
    "duplicate_class_rows": 3,
    "invalid_class_labels": 4,
    "transactions_without_label": 6,
    "labels_without_transaction": 7,
    "orphan_edges": 8,
    "self_loops": 3,
    "duplicate_edges": 5,
    "json_fee_breaks": 2,
}


def make_elliptic():
    tx_ids = rng.choice(np.arange(1_000_000, 9_999_999), size=N_TX, replace=False)
    steps = np.sort(rng.integers(1, N_STEPS + 1, size=N_TX))
    labels = rng.choice(["1", "2", "unknown"], size=N_TX, p=[0.03, 0.22, 0.75])

    feats = rng.normal(0, 1, size=(N_TX, N_FEATURES))
    illicit = labels == "1"
    feats[illicit, :3] += rng.normal(5.0, 1.0, size=(illicit.sum(), 3))  # separable-ish signal
    features = pd.DataFrame(feats)
    features.insert(0, "time_step", steps)
    features.insert(0, "txId", tx_ids)

    # Edges within each time step; illicit transactions fan out more
    edges = []
    for step in range(1, N_STEPS + 1):
        idx = np.where(steps == step)[0]
        for i in idx:
            k = rng.poisson(4 if illicit[i] else 1)
            if k and len(idx) > 1:
                targets = rng.choice(idx[idx != i], size=min(k, len(idx) - 1), replace=False)
                edges += [(tx_ids[i], tx_ids[t]) for t in targets]
    edges = pd.DataFrame(edges, columns=["txId1", "txId2"]).drop_duplicates()
    classes = pd.DataFrame({"txId": tx_ids, "class": labels})

    # --- plant issues -------------------------------------------------------------
    # Transactions without a label: remove their class rows
    unlabeled = rng.choice(tx_ids, PLANTED["transactions_without_label"], replace=False)
    classes = classes[~classes["txId"].isin(unlabeled)]
    # Labels without a transaction
    ghost_ids = np.arange(100, 100 + PLANTED["labels_without_transaction"])
    classes = pd.concat([classes, pd.DataFrame({"txId": ghost_ids, "class": "2"})])
    # Duplicate class rows (from labeled, non-ghost transactions)
    dup_cls = classes[classes["txId"] >= 1_000_000].sample(
        PLANTED["duplicate_class_rows"], random_state=1)
    classes = pd.concat([classes, dup_cls])
    # Invalid class labels
    bad = classes[(classes["txId"] >= 1_000_000)
                  & ~classes["txId"].isin(dup_cls["txId"])].sample(
        PLANTED["invalid_class_labels"], random_state=2).index
    classes.loc[bad, "class"] = ["3", "Illicit", "", "?"]

    # Features: null values, then extra rows with bad keys / duplicates
    null_rows = rng.choice(len(features), PLANTED["rows_with_null_features"], replace=False)
    for r in null_rows:
        features.iat[r, 5] = np.nan
    extra = []
    dup_rows = features.iloc[rng.choice(len(features), PLANTED["duplicate_txid_in_features"],
                                        replace=False)]
    extra.append(dup_rows)
    missing = features.iloc[:PLANTED["missing_time_step"]].copy()
    missing["txId"] = np.arange(200, 200 + len(missing))
    missing["time_step"] = np.nan
    extra.append(missing)
    out_range = features.iloc[:PLANTED["time_step_out_of_range"]].copy()
    out_range["txId"] = np.arange(300, 300 + len(out_range))
    out_range["time_step"] = 99
    extra.append(out_range)
    features = pd.concat([features] + extra, ignore_index=True)

    # Edges: orphans, self-loops, duplicates
    orphan = pd.DataFrame({"txId1": rng.choice(tx_ids, PLANTED["orphan_edges"]),
                           "txId2": np.arange(400, 400 + PLANTED["orphan_edges"])})
    loops_src = rng.choice(tx_ids, PLANTED["self_loops"], replace=False)
    loops = pd.DataFrame({"txId1": loops_src, "txId2": loops_src})
    dups = edges.sample(PLANTED["duplicate_edges"], random_state=3)
    edges = pd.concat([edges, orphan, loops, dups], ignore_index=True)

    OUT_RAW.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUT_RAW / "elliptic_txs_features.csv", header=False, index=False)
    classes.to_csv(OUT_RAW / "elliptic_txs_classes.csv", index=False)
    edges.to_csv(OUT_RAW / "elliptic_txs_edgelist.csv", index=False)


def make_esplora_json(n=60):
    """Transactions shaped like Blockstream Esplora's /block/:hash/txs response."""
    txs = []
    for i in range(n):
        coinbase = i == 0
        n_in, n_out = (1, 2) if coinbase else (int(rng.integers(1, 4)), int(rng.integers(1, 4)))
        vin = []
        for _ in range(n_in):
            if coinbase:
                vin.append({"txid": "0" * 64, "vout": 4294967295, "prevout": None,
                            "is_coinbase": True, "sequence": 4294967295})
            else:
                vin.append({"txid": f"{rng.integers(1e15):064x}", "vout": int(rng.integers(0, 3)),
                            "prevout": {"scriptpubkey_address": f"bc1q{rng.integers(1e12):x}",
                                        "scriptpubkey_type": "v0_p2wpkh",
                                        "value": int(rng.integers(10_000, 5_000_000))},
                            "is_coinbase": False, "sequence": 4294967293})
        total_in = sum(v["prevout"]["value"] for v in vin if v["prevout"]) if not coinbase else 0
        fee = 0 if coinbase else int(rng.integers(200, 5_000))
        spend = total_in - fee if not coinbase else 312_500_000
        splits = np.diff(np.sort(np.r_[0, rng.integers(1, spend, n_out - 1), spend]))
        vout = [{"scriptpubkey_address": f"bc1q{rng.integers(1e12):x}",
                 "scriptpubkey_type": "v0_p2wpkh", "value": int(v)} for v in splits]
        if i == n - 1:  # an OP_RETURN data output with no address and zero value
            vout.append({"scriptpubkey_type": "op_return", "value": 0})
        txs.append({"txid": f"{rng.integers(1e15):064x}", "version": 2, "locktime": 0,
                    "vin": vin, "vout": vout, "size": int(rng.integers(150, 600)),
                    "weight": int(rng.integers(500, 2000)), "fee": fee,
                    "status": {"confirmed": True, "block_height": 900_000,
                               "block_time": 1_758_000_000}})
    # Plant fee breaks: reported fee disagrees with inputs - outputs
    for t in txs[1:1 + PLANTED["json_fee_breaks"]]:
        t["fee"] += 111
    OUT_JSON.mkdir(parents=True, exist_ok=True)
    (OUT_JSON / "synthetic_block.json").write_text(json.dumps(txs, indent=2))


if __name__ == "__main__":
    make_elliptic()
    make_esplora_json()
    print(f"Synthetic data written to {OUT_RAW.parent}")
    print("Planted issues:", json.dumps(PLANTED, indent=2))
    sys.exit(0)
