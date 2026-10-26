# Interview prep

**Business framing**

I used 307,511 Home Credit applications as a public consumer-credit proxy. It demonstrates MSME-relevant methods, but does not establish performance on Indian merchants.

**Why SQL aggregation?**

I collapsed each child history to applicant level before joining, preventing one-to-many row multiplication. The final table has 168 candidate features; IDs and TARGET are excluded from that count.

**Why random holdout?**

There is no clean application date and IDs are not proven chronological. I used disjoint stratified random train, policy, test and holdout partitions; I do not claim out-of-time validation.

**Why WoE and logistic?**

WoE turns bins into smoothed log good/bad ratios. I fit bins, IV screening and correlation/VIF checks on training only, retaining 20 features for an additive, auditable logistic scorecard.

**What do AUC and Gini mean?**

AUC is the chance a random bad applicant ranks riskier than a random good. Gini = 2 x AUC - 1; holdout scorecard Gini is 0.5039, versus 0.5626 for LightGBM.

**What is KS?**

KS is the maximum difference between cumulative bad and good distributions across risk thresholds. Holdout scorecard KS is 37.64%; it measures separation, not probability calibration.

**How do points work?**

The configured base is 600 points at 50:1 good/bad odds. Each 20 points doubles those odds; I verified exact unrounded point additivity.

**Calibration and PSI**

Calibration compares predicted and observed risk; I report Brier scores and reliability curves without fitting a calibrator. Holdout score PSI is 0.00016; <0.1 stable, 0.1-0.25 moderate, >0.25 shifted are heuristic alerts.

**Why boosting and SHAP?**

LightGBM captures nonlinear interactions. Candidate settings use training-only cross-validation; tree-path-dependent SHAP explains contributions to log odds, with a numerical additivity check. Explanations are associative.

**How was the cut-off chosen?**

I optimized assumed net benefit on the policy partition with a minimum approval constraint, then froze score 545. Holdout estimated bad-rate reduction is 38.59%; good rejection share is 18.08%.

**How were rupee estimates made?**

With assumed 100,000 annual applications, Rs 500,000 loans, 60.00% LGD and 8.00% good margin, gross avoided loss is Rs 124.34 crore; net benefit is Rs 57.86 crore. These are illustrative, not realized.

**Limitations and SIMULATED extension**

No temporal validation, verified MSME population, causal policy effect or fairness certification is claimed. SIMULATED sales recall is 100.00%, with 16 SIMULATED days median lead time; built-in decline drives that artificial result.