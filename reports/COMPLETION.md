# Execution completion and evidence

All requested phases were executed on the official downloaded Home Credit CSVs.
The merchant-sales extension alone is SIMULATED. No production lending use is claimed.

| Step | Completion evidence |
|---|---|
| Setup | Requested folders, source modules, SQL files and notebooks exist |
| SQL | sql_execution.txt, ingestion.json, sql_layer.json, sql_integrity_checks.json |
| Audit / EDA | audit_execution.txt, audit.json, missing_values.csv, numeric_audit.csv, segment_risk.csv, figures |
| Split | audit.json and saved disjoint split assignments; stratified random holdout, not OOT |
| Scorecard | scorecard_design.json, woe_bins.csv, iv_ranking.csv, vif.csv, scorecard_points.csv, saved model |
| Boosting / SHAP | boosting_cv.csv, boosting_design.json, shap_global.csv, shap_local.csv, figures, saved model |
| Validation | metrics.json, gains/calibration CSVs, score_psi.json, key_feature_psi.json, bootstrap intervals |
| Cut-off | cutoff_strategy.json, cutoff CSVs, fixed-policy holdout counts and assumed economics |
| SIMULATED extension | SIMULATED_early_warning.json, SIMULATED daily-sales/outcome files and documented process |
| Dashboard | Offline HTML, Power BI exports, dashboard_screenshot.png, dashboard_verification.json |
| Memo | output/pdf/business_memo.pdf and visually inspected page PNGs |
| Final documentation | README.md, resume_bullets.md, interview_prep.pdf, executed notebook series |

## Verified invariants

- SQL modeling row count equals the labeled application count, with unique applicant IDs.
- Bureau and previous-loan parent keys are unique; future bureau updates are excluded.
- Every WoE bin count is based only on the model-training partition.
- Train, policy/calibration, test and holdout partitions do not overlap.
- All saved validation metrics agree with a fresh calculation from scored outputs.
- Reloaded scorecard and boosting models reproduce saved predictions on sampled holdout borrowers.
- Saved policy counts and economics agree with a fresh holdout calculation.
- JavaScript slider calculations agree with Python; charts render and policy reset works.
- The memo and interview sheet have the requested page counts and were visually inspected.
- Notebook cells executed without recorded errors.

## Execution issues and resolution

- Initial package download was blocked by sandbox DNS restrictions. The approved
  retry installed the dependencies; requirements.txt records that environment.
- A SQL launch before installation finished stopped at a missing pandas import.
  It processed no data; ingestion was then executed successfully.
- The supplied data dictionary uses Windows-1252, rather than UTF-8. It was read
  using its encoding, with no replacement or substitute dataset.
- The bureau date audit found future update records. The SQL filter was tightened
  and the complete SQL feature layer rebuilt before model fitting.
- The sandboxed Jupyter run stalled before executing cells. It was stopped; the
  approved run used local runtime directories and executed all notebooks successfully.
- Poppler emitted display-font warnings for unused fallback fonts. All rendered
  PDF pages were visually checked and showed no missing text or layout defects.
- LightGBM reports negative categorical values as missing. This is the intentional
  encoding for absent or unseen categories; model fitting and verification completed.

The source metric authority is reports/metrics.json. Read reports/LIMITATIONS.md
before discussing performance, policy impacts or the SIMULATED extension.
