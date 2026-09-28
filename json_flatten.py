"""Semi-structured data step: pull raw Bitcoin transactions as JSON from the public
Blockstream Esplora API, flatten the nested structure into three tables, and reconcile.

Each transaction is nested: a list of inputs (vin), each with a 'prevout' object, and a
list of outputs (vout). Flattening produces:
    transactions : one row per transaction
    inputs       : one row per input
    outputs      : one row per output

Reconciliation check: for every non-coinbase transaction,
    sum(input values) - sum(output values) should equal the reported fee.
A coinbase transaction (the miner reward) has no real inputs, so it is excluded.
"""
import json
import time

import pandas as pd

import config


def fetch_latest_block(pages: int = 4, pause: float = 0.5) -> tuple[str, list]:
    """Download up to 25 * pages transactions from the most recent block."""
    import requests

    base = config.ESPLORA_BASE_URL
    tip = requests.get(f"{base}/blocks/tip/hash", timeout=30)
    tip.raise_for_status()  # stop with a clear error instead of saving a bad file
    block_hash = tip.text.strip()
    if len(block_hash) != 64:
        raise ValueError(f"Unexpected block hash from API: {block_hash[:80]!r}")
    txs = []
    for page in range(pages):
        resp = requests.get(f"{base}/block/{block_hash}/txs/{page * 25}", timeout=30)
        if resp.status_code != 200:
            break
        batch = resp.json()
        if not batch:
            break
        txs.extend(batch)
        time.sleep(pause)  # be polite to a free public API

    if not txs:
        raise ValueError("API returned no transactions; nothing saved")
    config.JSON_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.JSON_DIR / f"block_{block_hash[:16]}.json"
    out_path.write_text(json.dumps(txs, indent=2))
    return str(out_path), txs


def load_json_files(json_dir=config.JSON_DIR) -> list:
    txs = []
    for path in sorted(json_dir.glob("*.json")):
        txs.extend(json.loads(path.read_text()))
    return txs


def flatten(txs: list):
    tx_rows, in_rows, out_rows = [], [], []
    for tx in txs:
        status = tx.get("status") or {}
        vin = tx.get("vin") or []
        vout = tx.get("vout") or []
        is_coinbase = any(i.get("is_coinbase") for i in vin)
        tx_rows.append({
            "txid": tx.get("txid"),
            "block_height": status.get("block_height"),
            "block_time": pd.to_datetime(status.get("block_time"), unit="s", utc=True)
                          if status.get("block_time") else pd.NaT,
            "n_inputs": len(vin),
            "n_outputs": len(vout),
            "fee_sats": tx.get("fee"),
            "size_bytes": tx.get("size"),
            "weight": tx.get("weight"),
            "is_coinbase": is_coinbase,
        })
        for idx, i in enumerate(vin):
            prev = i.get("prevout") or {}
            in_rows.append({
                "txid": tx.get("txid"),
                "input_index": idx,
                "prev_txid": i.get("txid"),
                "prev_vout": i.get("vout"),
                "address": prev.get("scriptpubkey_address"),
                "value_sats": prev.get("value"),
                "is_coinbase": bool(i.get("is_coinbase")),
            })
        for idx, o in enumerate(vout):
            out_rows.append({
                "txid": tx.get("txid"),
                "output_index": idx,
                "address": o.get("scriptpubkey_address"),
                "script_type": o.get("scriptpubkey_type"),
                "value_sats": o.get("value"),
            })
    return pd.DataFrame(tx_rows), pd.DataFrame(in_rows), pd.DataFrame(out_rows)


def reconcile_fees(tx_df, in_df, out_df) -> pd.DataFrame:
    ins = in_df.groupby("txid")["value_sats"].sum(min_count=1).rename("inputs_sats")
    outs = out_df.groupby("txid")["value_sats"].sum(min_count=1).rename("outputs_sats")
    rec = tx_df[["txid", "is_coinbase", "fee_sats"]].merge(
        ins, left_on="txid", right_index=True, how="left").merge(
        outs, left_on="txid", right_index=True, how="left")
    rec = rec[~rec["is_coinbase"]].copy()
    rec["implied_fee_sats"] = rec["inputs_sats"] - rec["outputs_sats"]
    rec["difference_sats"] = rec["implied_fee_sats"] - rec["fee_sats"]
    rec["result"] = rec["difference_sats"].map(
        lambda d: "Missing data" if pd.isna(d) else ("Ties out" if d == 0 else "Break"))
    return rec.drop(columns="is_coinbase")


def json_quality_summary(tx_df, in_df, out_df, rec) -> pd.DataFrame:
    rows = [
        ("Transactions parsed", len(tx_df)),
        ("Input rows", len(in_df)),
        ("Output rows", len(out_df)),
        ("Coinbase transactions (excluded from fee check)", int(tx_df["is_coinbase"].sum())),
        ("Duplicate txids", int(tx_df["txid"].duplicated().sum())),
        ("Outputs with no address (e.g. OP_RETURN data outputs)",
         int(out_df["address"].isna().sum()) if len(out_df) else 0),
        ("Fee reconciliation: ties out", int((rec["result"] == "Ties out").sum())),
        ("Fee reconciliation: breaks", int((rec["result"] == "Break").sum())),
        ("Fee reconciliation: missing data", int((rec["result"] == "Missing data").sum())),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])
