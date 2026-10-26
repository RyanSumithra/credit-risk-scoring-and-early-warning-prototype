DROP TABLE IF EXISTS bureau_features;
CREATE TABLE bureau_features AS
SELECT SK_ID_CURR, COUNT(*) AS bureau_count,
 SUM(CASE WHEN CREDIT_ACTIVE='Active' THEN 1 ELSE 0 END) AS bureau_active_count,
 AVG(DAYS_CREDIT) AS bureau_avg_credit_day, MIN(DAYS_CREDIT) AS bureau_oldest_credit_day,
 AVG(CREDIT_DAY_OVERDUE) AS bureau_avg_overdue_days, MAX(CREDIT_DAY_OVERDUE) AS bureau_max_overdue_days,
 SUM(AMT_CREDIT_SUM) AS bureau_credit_sum, SUM(AMT_CREDIT_SUM_DEBT) AS bureau_debt_sum,
 SUM(AMT_CREDIT_SUM_OVERDUE) AS bureau_overdue_sum,
 SUM(AMT_CREDIT_SUM_DEBT)*1.0/NULLIF(SUM(AMT_CREDIT_SUM),0) AS bureau_debt_credit_ratio,
 AVG(delinquent_month_share) AS bureau_delinquent_month_share,
 MAX(worst_status) AS bureau_worst_month_status, SUM(months_observed) AS bureau_months_observed
FROM bureau_clean GROUP BY SK_ID_CURR;
CREATE UNIQUE INDEX ix_bureau_feature_id ON bureau_features(SK_ID_CURR);
DROP TABLE IF EXISTS previous_features;
CREATE TABLE previous_features AS
SELECT SK_ID_CURR, COUNT(*) AS previous_count,
 AVG(CASE WHEN NAME_CONTRACT_STATUS='Refused' THEN 1.0 ELSE 0 END) AS previous_refusal_share,
 AVG(CASE WHEN NAME_CONTRACT_STATUS='Approved' THEN 1.0 ELSE 0 END) AS previous_approval_share,
 AVG(AMT_CREDIT) AS previous_avg_credit, AVG(AMT_ANNUITY) AS previous_avg_annuity,
 MAX(DAYS_DECISION) AS previous_latest_decision_day,
 AVG(AMT_CREDIT/NULLIF(AMT_APPLICATION,0)) AS previous_credit_application_ratio,
 AVG(CNT_PAYMENT) AS previous_avg_term
FROM previous_application WHERE DAYS_DECISION <= 0 GROUP BY SK_ID_CURR;
CREATE UNIQUE INDEX ix_previous_feature_id ON previous_features(SK_ID_CURR);
DROP TABLE IF EXISTS installment_features;
CREATE TABLE installment_features AS
SELECT SK_ID_CURR, COUNT(*) AS installment_count,
 AVG(CASE WHEN COALESCE(last_payment_day,0)>due_day OR amount_paid<amount_due THEN 1.0 ELSE 0 END) AS installment_late_share,
 AVG(MAX(0,CASE WHEN amount_paid<amount_due THEN -due_day ELSE COALESCE(last_payment_day,0)-due_day END)) AS installment_avg_days_late,
 MAX(MAX(0,CASE WHEN amount_paid<amount_due THEN -due_day ELSE COALESCE(last_payment_day,0)-due_day END)) AS installment_max_days_late,
 SUM(amount_paid)*1.0/NULLIF(SUM(amount_due),0) AS installment_payment_ratio,
 SUM(MAX(0,amount_due-amount_paid)) AS installment_unpaid_amount,
 AVG(CASE WHEN due_day>=-365 AND (COALESCE(last_payment_day,0)>due_day OR amount_paid<amount_due) THEN 1.0 WHEN due_day>=-365 THEN 0.0 END) AS installment_recent_late_share
FROM installment_clean GROUP BY SK_ID_CURR;
CREATE UNIQUE INDEX ix_installment_feature_id ON installment_features(SK_ID_CURR);
DROP TABLE IF EXISTS pos_features;
CREATE TABLE pos_features AS
SELECT SK_ID_CURR, COUNT(*) AS pos_month_count, COUNT(DISTINCT SK_ID_PREV) AS pos_loan_count,
 MAX(SK_DPD) AS pos_max_dpd, AVG(SK_DPD) AS pos_avg_dpd,
 AVG(CASE WHEN SK_DPD>0 THEN 1.0 ELSE 0 END) AS pos_delinquent_share,
 AVG(CNT_INSTALMENT_FUTURE) AS pos_avg_remaining_installments
FROM POS_CASH_balance WHERE MONTHS_BALANCE <= -1 GROUP BY SK_ID_CURR;
CREATE UNIQUE INDEX ix_pos_feature_id ON pos_features(SK_ID_CURR);
DROP TABLE IF EXISTS card_features;
CREATE TABLE card_features AS
SELECT SK_ID_CURR, COUNT(*) AS card_month_count, COUNT(DISTINCT SK_ID_PREV) AS card_loan_count,
 AVG(AMT_BALANCE/NULLIF(AMT_CREDIT_LIMIT_ACTUAL,0)) AS card_avg_utilization,
 MAX(AMT_BALANCE/NULLIF(AMT_CREDIT_LIMIT_ACTUAL,0)) AS card_max_utilization,
 AVG(SK_DPD) AS card_avg_dpd, MAX(SK_DPD) AS card_max_dpd,
 AVG(CASE WHEN SK_DPD>0 THEN 1.0 ELSE 0 END) AS card_delinquent_share,
 SUM(AMT_PAYMENT_TOTAL_CURRENT)*1.0/NULLIF(SUM(AMT_INST_MIN_REGULARITY),0) AS card_payment_min_ratio
FROM credit_card_balance WHERE MONTHS_BALANCE <= -1 GROUP BY SK_ID_CURR;
CREATE UNIQUE INDEX ix_card_feature_id ON card_features(SK_ID_CURR);
