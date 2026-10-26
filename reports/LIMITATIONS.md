# Limitations and assumptions

- This is Home Credit consumer lending data, not NeoGrowth data, and not verified
  Indian MSME lending data. Relevance to merchant underwriting is methodological.
- TARGET follows the competition's repayment-difficulty label. The official field
  description intentionally withholds the exact overdue and timing thresholds;
  do not describe it as a known regulatory default definition or fixed horizon.
- The holdout is stratified random, not out-of-time or pseudo-OOT. IDs are not
  treated as chronological. Random-split PSI cannot establish time stability.
- Borrower IDs do not repeat across partitions. Linked household/network relations
  cannot be ruled out because no comprehensive household/entity map is available.
- Only applicants with observed labels are modeled. Acceptance/selection bias,
  reject inference, survival effects and label maturity are not resolved.
- Relative dates and snapshots are used conservatively (prior completed months,
  prior decisions, due installments, payments observed by day zero, bureau credit
  opened and last updated by day zero). Real production availability still needs
  event timestamps, feature-store lineage and point-in-time reconstruction.
- Future installment payments are excluded; installment records are collapsed by
  loan/version/number. Revisions and missing obligations may still affect features.
- Missing histories remain missing, rather than being asserted to mean zero risk.
  Sparse monthly histories and incomplete reporting affect behavior aggregates.
- DAYS_EMPLOYED's positive sentinel is replaced with missing. Numeric outliers are
  audited, not silently removed. Binning thresholds are learned on training only.
- The descriptive full-portfolio EDA is not used to choose features or model settings.
  IV, bin merging, correlation/VIF checks and model fitting use training labels only.
- Numeric monotonic bins are empirical training-data pools; special/missing bins
  are separate and categorical bins need not be ordered. Smoothed WoE uses an
  additive half-count; unseen categories receive neutral WoE if no pooled bin exists.
- The scorecard is a compact, regularized linear log-odds approximation. IV is a
  univariate screening heuristic, not proof of causal relevance. Correlation and
  VIF filters do not eliminate all redundant signals or nonlinear dependencies.
- The fitted age WoE term has a positive coefficient despite the good/bad WoE
  convention. This conditional direction reversal is flagged in scorecard_design.json
  and requires review before a governed scorecard could be used in production.
- The boosting search is a bounded CPU-friendly candidate grid using stratified
  cross-validation on training only. It is not exhaustive hyperparameter optimization.
- Raw model probabilities are assessed for calibration; no calibration model is
  fit. A partition named calibration is reserved for policy/cut-off selection.
- Training discrimination is optimistic in-sample. Test and holdout are not used
  for transformations, tuning or selection of the recommended cut-off.
- Scores are unrounded for prediction and policy; rounding table points can change
  scores near a boundary. Higher points mean lower modeled risk.
- SHAP describes model associations, not causes or actionable advice. Correlated
  features can share attribution. Global/local plots use a fixed sample of holdout
  borrowers and tree-path-dependent log-odds explanations.
- Score PSI uses training-defined quantile boundaries, explicit missing buckets
  and half-count smoothing. Key-feature PSI is on train-fitted WoE representations.
  Thresholds are heuristic alerts, not statistical hypothesis tests or guarantees.
- Bootstrap intervals capture applicant resampling uncertainty in the fixed
  holdout; they omit model-selection uncertainty and distribution drift.
- Cut-off economics assume editable annual application count, identical loan size,
  LGD, and net margin on good loans. They scale the observed sample's rejection
  rates to the assumed application count. They are ILLUSTRATIVE ESTIMATES.
- Gross avoided losses and net benefit differ: net benefit subtracts foregone
  good-borrower margin. Neither is audited profit. Funding, collections, capital,
  acquisition costs, recovery timing, taxes and policy feedback are not modeled.
- Observed TARGET is used as a simplified bad-loan proxy for economics. No actual
  Home Credit losses are inferred; INR assumptions are not currency conversion.
- Bad-rate reductions are retrospective cohort-selection estimates, not causal
  improvements or real deployed outcomes. Rejected good share uses all observed
  good applicants as its denominator. The fixed policy may miss its minimum
  approval constraint in another partition or in deployment.
- The dashboard permits descriptive post-hoc threshold exploration on holdout;
  this does not change the preselected recommendation. Do not promote a new
  dashboard-selected policy without a fresh validation cohort.
- Sensitive/proxy attributes require fair-lending and legal review. Gender is
  excluded from predictors, but age, geography, occupation and other proxies remain.
  No fairness certification, causal debiasing or regulatory approval is claimed.
- SIMULATED merchant sales, outcomes and dates are artificial; only IDs are sampled
  for linkage. The data-generating process builds in pre-event decline and cannot
  validate operational alert precision, recall, lead time or incremental collections.
- Reproducibility depends on software versions and hardware. Raw-file hashes,
  software lock file, fixed seed, SQL, model artifacts and logs are provided.
- Source data and the large SQLite database are omitted from the distributable;
  users must obtain the official dataset under its applicable terms and rerun SQL.
- No model is ready for lending decisions without independent validation, actual
  Indian MSME data, temporal backtesting, monitoring, controls and business sign-off.
