# Bitcoin Transaction Data Quality & Anomaly Review

A data preparation and review pipeline for blockchain transaction data, built the way an
investigations team would need it: every record reconciled, every data problem logged
with the fix applied, every anomaly flag tested against known outcomes, and anything
unexplained escalated to an analyst rather than guessed at.

It uses the public **Elliptic dataset** (about 200,000 Bitcoin transactions, a small share
labeled illicit, a larger share labeled licit, and most unlabeled) and a sample of **raw
Bitcoin transactions in JSON** pulled from the public Blockstream Esplora API.

## What the pipeline does

| Step | What happens | Output |
|---|---|---|
| 1. Load | Reads the three Elliptic files without fixing anything, so every problem stays visible to the checks | — |
| 2. Data quality checks | 13 checks across structure, keys, labels, cross-file reconciliation, and graph edges. Each check is logged whether it passes or fails | `Data Quality Log`, `Record Reconciliation` |
| 3. SQL tie-out | Loads the raw tables into SQLite and recomputes the key checks with independent SQL queries. The two methods must agree | `SQL Tie-Out`, `Class Distribution (SQL)` |
| 4. Anomaly flags | Robust z-scores (median and MAD) per time step. Features are chosen on time steps 1–34 and results reported on 35–49, which the selection never saw | `Selected Features`, `Model Evaluation`, `By Time Step`, `Flagged Labeled` |
| 5. Escalation | Flagged transactions with no label go to a review queue. The pipeline never labels them illicit | `Review Queue` |
| 6. Raw JSON | Flattens nested transaction JSON into transactions, inputs, and outputs tables, then reconciles fees | `JSON Summary`, `Fee Reconciliation` |
| 7. Reporting | One Excel exception report (summary built from live formulas) plus flat CSVs for Tableau | `outputs/` |

## How to run

```bash
pip install -r requirements.txt

# 1. Download the Elliptic dataset from Kaggle (search "Elliptic Data Set") and put
#    these three files in data/raw/:
#      elliptic_txs_features.csv   elliptic_txs_classes.csv   elliptic_txs_edgelist.csv

# 2. Run the pipeline, pulling ~100 fresh transactions from the public API
python run_pipeline.py --fetch-json

# Later runs reuse the saved JSON in data/json/
python run_pipeline.py
```

**Test the checks themselves first (optional).** `scripts/make_synthetic_data.py` builds
small files in the exact Elliptic and Esplora formats with known problems planted in them.
The tests confirm every planted problem is found with the exact count, and that SQL and
pandas agree:

```bash
python scripts/make_synthetic_data.py
python -m pytest tests/
python run_pipeline.py --raw-dir data/synthetic/raw --json-dir data/synthetic/json
```

Synthetic data is for testing the code only. None of it is used in the results below.

## Data quality check catalog

| ID | Check | Fix applied |
|---|---|---|
| DQ01 | Features file has 167 columns | Flagged for review |
| DQ02 | Missing txId or time_step | Row excluded |
| DQ03 | Duplicate txId in features | First kept |
| DQ04 | time_step outside 1–49 | Row excluded |
| DQ05 | Missing feature values | Kept; nulls ignored in scoring |
| DQ06 | Duplicate txId in classes | First label kept |
| DQ07 | Class label not in {1, 2, unknown} | Treated as unknown (excluded from evaluation, not guessed) |
| DQ08 | Transaction with no label | Assigned unknown |
| DQ09 | Label with no transaction | Label excluded |
| DQ10 | Edge pointing to a missing transaction | Edge excluded |
| DQ11 | Self-loop edge | Edge excluded |
| DQ12 | Duplicate edge | First kept |
| DQ13 | Edge spanning two time steps | Kept, flagged for awareness |

`Record Reconciliation` shows raw rows, clean rows, and rows removed for each table, so
every excluded record is accounted for.

## Anomaly method

**Step 1: robust z-scores.** For every feature, within each time step:

```
robust z = 0.6745 × (value - median) / MAD
```

The median and MAD are used instead of the mean and standard deviation because a handful
of extreme transactions would otherwise inflate the baseline they are measured against.
When more than half the values in a time step are identical (common for degree counts),
MAD is zero, so the method falls back to the mean absolute deviation, scaled by 1.2533.

**Step 2: choose features on the past, test on the future.** The data is split by time,
as in the original Elliptic paper: time steps 1–34 are the training period and 35–49 are
the holdout period. Every feature is tested in both directions (unusually high, unusually
low) on the training period only, and the most precise are kept (at least 10% recall,
near-duplicates skipped). The chosen features and their training scores are listed in the
`Selected Features` sheet.

**Step 3: flag.** A transaction is flagged when at least 2 of the 3 selected features
breach the threshold (|z| > 3.5). All settings are in `config.py`.

**Step 4: report on the holdout period.** The headline precision and recall come from time
steps 35–49, which played no part in choosing the features. Training-period results are
shown alongside for comparison; a large gap between the two would signal overfitting.

**Why not just flag the most extreme transactions?** That was the first version, and it
failed: precision fell below the baseline rate. In this dataset, transactions with
unusually many links are mostly licit (large, busy services), so raw extremeness points
the wrong way. The fix was to let the training period show which features, and which
direction, separate illicit activity, then prove it on unseen later data.

**Evaluation rules.** Precision and recall are calculated on labeled transactions only.
Unknown transactions are never counted as hits or misses. Precision is always shown next
to the baseline illicit rate, because a precision number means little on its own.

## JSON reconciliation

Each raw transaction nests a list of inputs (each pointing to the earlier output it spends)
and a list of outputs. After flattening, the pipeline checks for every non-coinbase
transaction that:

```
sum(input values) - sum(output values) = reported fee
```

Any difference is logged as a break. Coinbase transactions (the miner reward) have no real
inputs and are excluded. Outputs with no address, such as OP_RETURN data outputs, are
counted and kept.

## Output data dictionary

**Review Queue / Flagged Labeled**

| Column | Meaning |
|---|---|
| txId | Elliptic transaction ID (anonymized) |
| time_step | Elliptic time step, 1–49, roughly two weeks each |
| label | illicit, licit, or unknown (Flagged Labeled only) |
| period | train (1-34) or holdout (35-49) (Flagged Labeled only) |
| anomaly_score | Largest robust z-score across the selected features, in their flagged direction |
| top_driver | Selected feature with the largest robust z-score |
| features_breached | Number of features with \|z\| above the threshold |
| in_degree / out_degree | Edge counts after data quality exclusions |
| status | Review status (Review Queue only) |

**Fee Reconciliation**

| Column | Meaning |
|---|---|
| inputs_sats / outputs_sats | Summed input and output values, in satoshis |
| implied_fee_sats | inputs - outputs |
| fee_sats | Fee reported by the API |
| difference_sats | implied - reported; 0 means it ties out |

## Assumptions

- Elliptic's features file has no header. Columns are named by position: txId, time_step,
  93 local features, then 72 aggregated features.
- Class `1` = illicit, `2` = licit, `unknown` = unlabeled, as documented by Elliptic.
- Edges are directed from txId1 to txId2, following the flow of funds.
- Where a record is duplicated, the first occurrence is treated as the record of truth.

## Limitations

- **This is not a fraud classifier.** The flags point to transactions that look unusual
  within their time period. Whether any of them is illicit is an investigator's call.
- **Most Elliptic features are anonymized**, so a flag can say which feature drove it
  (`top_driver`) but not what that feature means in business terms.
- **Labels are sparse and imbalanced.** Most transactions are unlabeled, so recall can
  only be measured on the small labeled subset.
- **Behavior shifts over time.** The original Elliptic paper (Weber et al., 2019) reports
  that models trained on earlier time steps stop working after a dark market shutdown
  around time step 43. This pipeline shows the same break: see `By Time Step`. Rules
  learned from the past need regular re-validation.
- **Feature selection uses labels.** The thresholds are unsupervised, but the choice of
  which features to watch is learned from labeled history. That is why results are only
  claimed on the holdout period.
- **The JSON sample is not linked to Elliptic.** Elliptic's transaction IDs are
  anonymized, so the raw JSON step demonstrates semi-structured data handling and
  reconciliation on separate, live data.

## Results

From a full run on the Elliptic dataset plus a live pull of 100 transactions from the
Blockstream Esplora API:

- **Data quality:** 13 checks run on 203,769 transactions and 234,355 edges; all passed,
  and all 4 SQL tie-outs match the pandas results.
- **Anomaly flags (holdout, time steps 35–49):** 35.4% precision vs. a 6.5% baseline
  illicit rate (5.4x lift), 50.4% recall. Training-period precision was 46.7%, so the
  rules held up reasonably well on data they had never seen.
- **Time steps 43+:** precision falls from 33–66% to near zero, matching the dark market
  shutdown reported by Weber et al. (2019). Rules learned from the past need regular
  re-validation.
- **Escalation:** 12,636 unlabeled transactions routed to the review queue, ranked by
  anomaly score. None were labeled by the pipeline.
- **JSON:** 100 live transactions flattened into 353 input rows and 258 output rows; fees
  reconciled on all 99 non-coinbase transactions with zero breaks. 15 outputs carried no
  address (data outputs) and were kept and counted.

## Project structure

```
config.py                   paths, thresholds, feature choices
run_pipeline.py             runs all steps end to end
src/load.py                 reads the Elliptic files
src/quality_checks.py       data quality checks and issue log
src/sql/reconciliation_checks.sql
src/sql_checks.py           SQL tie-out against the pandas checks
src/features.py             in/out degree from the edge list
src/anomaly.py              robust z-scores, evaluation, review queue
src/json_flatten.py         Esplora API pull, JSON flattening, fee reconciliation
src/report.py               Excel exception report and Tableau CSVs
scripts/make_synthetic_data.py   test data with planted issues
tests/test_quality_checks.py     confirms the checks find what was planted
```

## Data sources

- Elliptic Data Set: Weber et al., "Anti-Money Laundering in Bitcoin: Experimenting with
  Graph Convolutional Networks for Financial Forensics," KDD Workshop, 2019. Available on
  Kaggle.
- Blockstream Esplora API: public Bitcoin block explorer API (blockstream.info/api).
