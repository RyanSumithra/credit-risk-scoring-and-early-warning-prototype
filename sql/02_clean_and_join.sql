-- Relative application day is zero. Only completed monthly snapshots are used.
-- Aggregate bureau_balance first to prevent join multiplication.
DROP TABLE IF EXISTS bureau_monthly;
CREATE TABLE bureau_monthly AS
SELECT SK_ID_BUREAU, COUNT(*) AS months_observed,
 AVG(CASE WHEN STATUS IN ('1','2','3','4','5') THEN 1.0 ELSE 0 END) AS delinquent_month_share,
 MAX(CASE WHEN STATUS IN ('1','2','3','4','5') THEN CAST(STATUS AS INTEGER) ELSE 0 END) AS worst_status
FROM bureau_balance WHERE MONTHS_BALANCE <= -1 GROUP BY SK_ID_BUREAU;
CREATE UNIQUE INDEX ix_bm_id ON bureau_monthly(SK_ID_BUREAU);
DROP TABLE IF EXISTS bureau_clean;
CREATE TABLE bureau_clean AS
SELECT b.*, m.months_observed, m.delinquent_month_share, m.worst_status
FROM bureau b LEFT JOIN bureau_monthly m USING(SK_ID_BUREAU)
WHERE b.DAYS_CREDIT <= 0 AND (b.DAYS_CREDIT_UPDATE <= 0 OR b.DAYS_CREDIT_UPDATE IS NULL);

-- Partial payments are collapsed; future payments contribute zero.
DROP TABLE IF EXISTS installment_clean;
CREATE TABLE installment_clean AS
SELECT SK_ID_CURR, SK_ID_PREV, NUM_INSTALMENT_VERSION, NUM_INSTALMENT_NUMBER,
 MIN(DAYS_INSTALMENT) AS due_day, MAX(AMT_INSTALMENT) AS amount_due,
 SUM(CASE WHEN DAYS_ENTRY_PAYMENT <= 0 THEN AMT_PAYMENT ELSE 0 END) AS amount_paid,
 MAX(CASE WHEN DAYS_ENTRY_PAYMENT <= 0 THEN DAYS_ENTRY_PAYMENT END) AS last_payment_day
FROM installments_payments WHERE DAYS_INSTALMENT <= 0
GROUP BY SK_ID_CURR, SK_ID_PREV, NUM_INSTALMENT_VERSION, NUM_INSTALMENT_NUMBER;
