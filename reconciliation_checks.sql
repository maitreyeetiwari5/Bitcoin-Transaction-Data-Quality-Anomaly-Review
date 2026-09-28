-- Independent SQL tie-out of the pandas quality checks.
-- Runs against the RAW tables, so the counts should match the pandas issue log exactly.
-- Each query is preceded by a "-- name:" line that the runner uses as its label.
-- features_valid applies the same key and time-step rules as the pandas checks
-- (non-null txId and time_step, time_step 1-49), so the two can be compared like for like.

-- name: create_features_valid
CREATE VIEW IF NOT EXISTS features_valid AS
SELECT DISTINCT txId, time_step
FROM features_raw
WHERE txId IS NOT NULL AND time_step IS NOT NULL AND time_step BETWEEN 1 AND 49;

-- name: raw_row_counts
SELECT 'features' AS table_name, COUNT(*) AS row_count FROM features_raw
UNION ALL SELECT 'classes', COUNT(*) FROM classes_raw
UNION ALL SELECT 'edges', COUNT(*) FROM edges_raw;

-- name: duplicate_txid_in_features
SELECT COUNT(*) - COUNT(DISTINCT txId) AS duplicate_rows
FROM features_raw
WHERE txId IS NOT NULL AND time_step IS NOT NULL;

-- name: transactions_without_label
SELECT COUNT(DISTINCT f.txId) AS unlabeled_transactions
FROM features_valid f
LEFT JOIN classes_raw c ON f.txId = c.txId
WHERE c.txId IS NULL;

-- name: labels_without_transaction
SELECT COUNT(DISTINCT c.txId) AS orphan_labels
FROM classes_raw c
LEFT JOIN features_valid f ON c.txId = f.txId
WHERE f.txId IS NULL;

-- name: orphan_edges
SELECT COUNT(*) AS orphan_edges
FROM edges_raw e
WHERE e.txId1 NOT IN (SELECT txId FROM features_valid)
   OR e.txId2 NOT IN (SELECT txId FROM features_valid);

-- name: class_distribution
SELECT LOWER(TRIM(class)) AS class_value, COUNT(*) AS transactions
FROM classes_raw
GROUP BY LOWER(TRIM(class))
ORDER BY transactions DESC;

-- name: illicit_share_by_time_step
SELECT f.time_step,
       SUM(CASE WHEN TRIM(c.class) = '1' THEN 1 ELSE 0 END) AS illicit,
       SUM(CASE WHEN TRIM(c.class) = '2' THEN 1 ELSE 0 END) AS licit,
       ROUND(1.0 * SUM(CASE WHEN TRIM(c.class) = '1' THEN 1 ELSE 0 END)
             / NULLIF(SUM(CASE WHEN TRIM(c.class) IN ('1', '2') THEN 1 ELSE 0 END), 0), 4)
         AS illicit_share_of_labeled
FROM features_valid f
JOIN classes_raw c ON f.txId = c.txId
GROUP BY f.time_step
ORDER BY f.time_step;
