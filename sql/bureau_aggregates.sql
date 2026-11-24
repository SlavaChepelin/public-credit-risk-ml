-- Агрегаты bureau «как в проде» для Home Credit.
-- Ключ: SK_ID_CURR. Используется DuckDB / scripts/build_features_duckdb.py.

SELECT
    SK_ID_CURR,
    COUNT(*) AS bureau_count,
    SUM(AMT_CREDIT_SUM) AS bureau_amt_credit_sum,
    AVG(AMT_CREDIT_SUM) AS bureau_amt_credit_mean,
    SUM(AMT_CREDIT_SUM_DEBT) AS bureau_amt_debt_sum,
    AVG(AMT_CREDIT_SUM_DEBT) AS bureau_amt_debt_mean,
    SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1 ELSE 0 END) AS bureau_active_count,
    AVG(DAYS_CREDIT) AS bureau_days_credit_mean,
    MAX(DAYS_CREDIT) AS bureau_days_credit_max,
    MAX(CREDIT_DAY_OVERDUE) AS bureau_overdue_max
FROM bureau
GROUP BY SK_ID_CURR;
