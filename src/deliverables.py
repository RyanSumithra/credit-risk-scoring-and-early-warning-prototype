"""Generate stakeholder outputs exclusively from executed metric artifacts."""
import json
from pathlib import Path
from html import escape
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from src.common import ROOT,REPORTS,dump

def pct(value): return f'{value:.2%}'

def documents(m):
    output=ROOT/'output/pdf'; output.mkdir(parents=True,exist_ok=True)
    sc=m['validation']['scorecard']['holdout']; gb=m['validation']['lightgbm']['holdout']
    cut=m['cutoff']['evaluation']['holdout']; econ=m['cutoff']['assumptions']; sim=m['SIMULATED_early_warning']
    psi=m['score_psi']['scorecard']['holdout']['psi']
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='MemoTitle',fontName='Helvetica-Bold',fontSize=23,leading=27,textColor=colors.HexColor('#102e43'),spaceAfter=14))
    styles.add(ParagraphStyle(name='MemoBody',fontName='Helvetica',fontSize=10.2,leading=14,spaceAfter=9))
    styles.add(ParagraphStyle(name='MemoHead',fontName='Helvetica-Bold',fontSize=12,leading=15,textColor=colors.HexColor('#147d92'),spaceBefore=8,spaceAfter=6))
    styles.add(ParagraphStyle(name='SmallNote',fontName='Helvetica',fontSize=8.4,leading=11,textColor=colors.HexColor('#506575'),spaceAfter=6))
    def p(text,style='MemoBody'): return Paragraph(text,styles[style])
    def table(rows,widths):
        t=Table([[p(str(c),'SmallNote') for c in row] for row in rows],colWidths=widths,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e4f0f2')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#147d92')),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7)]))
        return t
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#c9d8df')); canvas.line(42,40,A4[0]-42,40)
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#506575'))
        canvas.drawString(42,27,'HOME CREDIT PORTFOLIO DEMONSTRATION | NOT FOR LENDING DECISIONS')
        canvas.drawRightString(A4[0]-42,27,str(doc.page))
    rows=[['Untouched random holdout','Scorecard','LightGBM'],
          ['AUC',f"{sc['auc']:.4f}",f"{gb['auc']:.4f}"],
          ['Gini',f"{sc['gini']:.4f}",f"{gb['gini']:.4f}"],
          ['KS',pct(sc['ks']),pct(gb['ks'])],
          ['Brier score (lower is better)',f"{sc['brier']:.5f}",f"{gb['brier']:.5f}"],
          ['Observed / predicted TARGET rate',pct(sc['bad_rate'])+' / '+pct(sc['mean_predicted_pd']),pct(gb['bad_rate'])+' / '+pct(gb['mean_predicted_pd'])]]
    story=[p('Credit risk: ranking to policy','MemoTitle'),p('To: DVP - Analytics | Date: 26 October 2026','SmallNote'),
           p('Decision requested','MemoHead'),p('Sponsor a controlled validation study on actual Indian MSME data. Retain the interpretable scorecard as a governance reference and assess the boosting challenger before any lending deployment.'),
           p('Problem and evidence','MemoHead'),
           p(f"The project ranks repayment difficulty on {m['dataset']['rows']:,} labeled Home Credit applications. SQL joins aggregate bureau, prior applications, installments, POS and card history into {m['dataset']['candidate_features']} candidate features. The observed TARGET rate is {pct(m['dataset']['bad_rate'])}. This is consumer lending proxy data, not NeoGrowth or verified Indian MSME data."),
           p(f"The scorecard uses {m['scorecard_design']['selected_feature_count']} training-selected WoE features. A LightGBM challenger is tuned using training-only cross-validation. A separate policy partition sets the cut-off; neither test nor holdout labels set preprocessing, model parameters or the recommended policy. The holdout contains {sc['rows']:,} applications."),
           table(rows,[240,135,135]),Spacer(1,8),
           p('What the findings support','MemoHead'),
           p(f"Both models show measurable ranking ability on the random holdout. LightGBM's holdout AUC difference from the scorecard is {gb['auc']-sc['auc']:.4f}; the paired bootstrap interval is [{m['holdout_auc_bootstrap_95pct']['boosting_minus_scorecard_auc']['low']:.4f}, {m['holdout_auc_bootstrap_95pct']['boosting_minus_scorecard_auc']['high']:.4f}]. Scorecard score PSI is {psi:.5f} versus training. A random split cannot establish time stability."),
           p('Source: reports/metrics.json, SQL artifacts and fixed-seed executed code. TARGET denotes repayment difficulty; exact overdue/timing thresholds are withheld by the dataset provider. SHAP explanations show associations, not causes.','SmallNote'),
           PageBreak(),p('Policy economics and controls','MemoTitle'),
           p('Illustrative cut-off policy','MemoHead'),
           p(f"Approve at score >= {m['cutoff']['selected_cutoff']:.0f}. This cut-off maximizes assumed net benefit on the policy partition subject to its minimum approval constraint. On untouched holdout, approval is {pct(cut['approval_rate'])}; accepted TARGET rate is {pct(cut['accepted_bad_rate'])}, versus {pct(cut['baseline_bad_rate'])} baseline. That is an estimated retrospective relative reduction of {pct(cut['relative_bad_rate_reduction'])}, while rejecting {pct(cut['good_customers_rejected_share'])} of observed good applicants."),
           table([['Editable assumption / estimate','Value'],['Annual application count',f"{econ['annual_applications']:,}"],['Average loan / LGD / good-loan net margin',f"Rs {econ['average_loan_inr']:,} / {pct(econ['lgd'])} / {pct(econ['good_loan_margin'])}"],['Annual gross loss avoided (ILLUSTRATIVE)',f"Rs {cut['estimated_annual_gross_loss_avoided_crore']:.2f} crore"],['Annual good-loan margin foregone (ILLUSTRATIVE)',f"Rs {cut['estimated_annual_good_margin_foregone_crore']:.2f} crore"],['Annual net benefit (ILLUSTRATIVE)',f"Rs {cut['estimated_annual_net_benefit_crore']:.2f} crore"]],[340,170]),
           Spacer(1,7),p('Gross avoided loss = rejected bad share x annual applications x loan size x LGD. Net benefit subtracts rejected good share x annual applications x loan size x margin. Estimates assume unchanged applicant mix and identical exposure; they are not realized savings or a currency conversion.','SmallNote'),
           p('Risks and implementation gates','MemoHead'),
           p('There is no verified application chronology: this is random holdout validation. Selection bias, reject inference and label maturity remain unresolved. Calibration is measured, not corrected. Age, geography and occupation may encode sensitive proxies despite excluding gender. The age WoE coefficient reverses direction after adjustment and needs governance review.'),
           p('Before piloting: reconstruct point-in-time MSME features and a documented target horizon; run temporal and segment validation; assess calibration and fairness; validate economic assumptions and approval capacity; establish independent review, score/feature drift monitoring, override logging and rollback controls.'),
           p('SIMULATED early-warning extension','MemoHead'),
           p(f"A separate artificial merchant-sales process produces SIMULATED precision {pct(sim['precision'])}, recall {pct(sim['recall'])} and median lead time {sim['median_lead_days']:.0f} SIMULATED days among detected artificial events. The process deliberately builds in declining sales. These figures demonstrate alert logic and cannot establish real merchant performance."),
           p('Interactive dashboard and CSV export accompany this memo. Full assumptions and limitations: reports/LIMITATIONS.md. All quantitative claims are generated from reports/metrics.json.','SmallNote')]
    memo=output/'business_memo.pdf'
    SimpleDocTemplate(str(memo),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=38,bottomMargin=52).build(story,onFirstPage=footer,onLaterPages=footer)
    entries=[
        ('Business framing',f"I used {m['dataset']['rows']:,} Home Credit applications as a public consumer-credit proxy. It demonstrates MSME-relevant methods, but does not establish performance on Indian merchants."),
        ('Why SQL aggregation?',f"I collapsed each child history to applicant level before joining, preventing one-to-many row multiplication. The final table has {m['dataset']['candidate_features']} candidate features; IDs and TARGET are excluded from that count."),
        ('Why random holdout?',"There is no clean application date and IDs are not proven chronological. I used disjoint stratified random train, policy, test and holdout partitions; I do not claim out-of-time validation."),
        ('Why WoE and logistic?',f"WoE turns bins into smoothed log good/bad ratios. I fit bins, IV screening and correlation/VIF checks on training only, retaining {m['scorecard_design']['selected_feature_count']} features for an additive, auditable logistic scorecard."),
        ('What do AUC and Gini mean?',f"AUC is the chance a random bad applicant ranks riskier than a random good. Gini = 2 x AUC - 1; holdout scorecard Gini is {sc['gini']:.4f}, versus {gb['gini']:.4f} for LightGBM."),
        ('What is KS?',f"KS is the maximum difference between cumulative bad and good distributions across risk thresholds. Holdout scorecard KS is {pct(sc['ks'])}; it measures separation, not probability calibration."),
        ('How do points work?',f"The configured base is {m['scorecard_design']['base_score']} points at {m['scorecard_design']['base_good_bad_odds']}:1 good/bad odds. Each {m['scorecard_design']['pdo']} points doubles those odds; I verified exact unrounded point additivity."),
        ('Calibration and PSI',f"Calibration compares predicted and observed risk; I report Brier scores and reliability curves without fitting a calibrator. Holdout score PSI is {psi:.5f}; <0.1 stable, 0.1-0.25 moderate, >0.25 shifted are heuristic alerts."),
        ('Why boosting and SHAP?',"LightGBM captures nonlinear interactions. Candidate settings use training-only cross-validation; tree-path-dependent SHAP explains contributions to log odds, with a numerical additivity check. Explanations are associative."),
        ('How was the cut-off chosen?',f"I optimized assumed net benefit on the policy partition with a minimum approval constraint, then froze score {m['cutoff']['selected_cutoff']:.0f}. Holdout estimated bad-rate reduction is {pct(cut['relative_bad_rate_reduction'])}; good rejection share is {pct(cut['good_customers_rejected_share'])}."),
        ('How were rupee estimates made?',f"With assumed {econ['annual_applications']:,} annual applications, Rs {econ['average_loan_inr']:,} loans, {pct(econ['lgd'])} LGD and {pct(econ['good_loan_margin'])} good margin, gross avoided loss is Rs {cut['estimated_annual_gross_loss_avoided_crore']:.2f} crore; net benefit is Rs {cut['estimated_annual_net_benefit_crore']:.2f} crore. These are illustrative, not realized."),
        ('Limitations and SIMULATED extension',f"No temporal validation, verified MSME population, causal policy effect or fairness certification is claimed. SIMULATED sales recall is {pct(sim['recall'])}, with {sim['median_lead_days']:.0f} SIMULATED days median lead time; built-in decline drives that artificial result."),
    ]
    interview=[p('Interview notes','MemoTitle'),p('Speak to evidence, assumptions and limits | Executed portfolio project','SmallNote')]
    styles.add(ParagraphStyle(name='Interview',fontName='Helvetica',fontSize=9.1,leading=12,spaceAfter=9))
    for heading,text in entries: interview.append(p('<b>'+escape(heading)+'</b><br/>'+escape(text),'Interview'))
    prep=output/'interview_prep.pdf'
    SimpleDocTemplate(str(prep),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=32,bottomMargin=48).build(interview,onFirstPage=footer,onLaterPages=footer)
    (REPORTS/'interview_prep.md').write_text('# Interview prep\n\n'+'\n\n'.join('**'+a+'**\n\n'+b for a,b in entries),encoding='utf-8')
    bullets=[
        f"Built an end-to-end credit risk pipeline on {m['dataset']['rows']:,} Home Credit loan applications, using SQL to aggregate bureau, prior-loan, installment, POS and card history into {m['dataset']['candidate_features']} candidate features for repayment-difficulty prediction.",
        f"Developed a {m['scorecard_design']['selected_feature_count']}-feature WoE/IV logistic scorecard (random-holdout Gini {sc['gini']:.4f}, KS {pct(sc['ks'])}) and benchmarked LightGBM (Gini {gb['gini']:.4f}), using SHAP to explain model risk drivers.",
        f"Validated models using stratified random holdout testing and PSI monitoring, with scorecard score PSI {psi:.5f} against training; no out-of-time validation claimed.",
        f"Designed a score cut-off with an estimated retrospective bad-rate reduction of {pct(cut['relative_bad_rate_reduction'])} while rejecting {pct(cut['good_customers_rejected_share'])} of good applicants; illustrative annual gross loss avoided of Rs {cut['estimated_annual_gross_loss_avoided_crore']:.2f} crore and net benefit of Rs {cut['estimated_annual_net_benefit_crore']:.2f} crore under stated assumptions.",
        'Created an interactive HTML/Plotly dashboard tracking segment risk, score distribution and cut-off economics, with a Power BI CSV export and a business memo for the DVP - Analytics.',
        f"SIMULATED merchant daily-sales data to demonstrate an early-warning flag identifying {pct(sim['recall'])} of artificial event borrowers at {pct(sim['precision'])} SIMULATED precision, with {sim['median_lead_days']:.0f} SIMULATED days median lead time among detected events.",
    ]
    (REPORTS/'resume_bullets.md').write_text('# Resume bullets (generated from metrics.json)\n\n'+'\n\n'.join('- '+b for b in bullets)+'\n',encoding='utf-8')
    return memo,prep

def readme(m):
    sc=m['validation']['scorecard']['holdout'];gb=m['validation']['lightgbm']['holdout'];cut=m['cutoff']['evaluation']['holdout']
    income=m['audit']['income_segments']
    insight='\n'.join(f"- {s['segment']}: {pct(s['bad_rate'])} TARGET rate, {pct(s['volume_share'])} of full labeled volume, {s['relative_risk']:.2f}x portfolio risk." for s in income[:4])
    text=f'''# MSME Credit Risk Scorecard and Early-Warning System

Help a lending analytics team rank repayment difficulty, compare an auditable scorecard
with a nonlinear challenger, and quantify approval trade-offs before a controlled pilot.
This executed portfolio uses **Home Credit consumer lending data as a methodological
proxy**, not NeoGrowth data or verified Indian MSME data. It is not a production model.

## Headline results: untouched stratified random holdout

| Measure | Scorecard | LightGBM |
|---|---:|---:|
| AUC | {sc['auc']:.4f} | {gb['auc']:.4f} |
| Gini | {sc['gini']:.4f} | {gb['gini']:.4f} |
| KS | {pct(sc['ks'])} | {pct(gb['ks'])} |
| Brier | {sc['brier']:.5f} | {gb['brier']:.5f} |

- SQL modeling table: **{m['dataset']['rows']:,} rows**, **{m['dataset']['candidate_features']} candidate predictors** (excludes identifier and TARGET).
- Scorecard retains **{m['scorecard_design']['selected_feature_count']}** training-selected predictors; LightGBM uses **{m['scorecard_design']['eligible_features']}** eligible predictors.
- Scorecard holdout PSI: **{m['score_psi']['scorecard']['holdout']['psi']:.5f}** versus train; this random split cannot establish temporal stability.
- Fixed policy score >= **{m['cutoff']['selected_cutoff']:.0f}**, selected on the separate policy partition: holdout approval **{pct(cut['approval_rate'])}**, accepted bad rate **{pct(cut['accepted_bad_rate'])}** versus **{pct(cut['baseline_bad_rate'])}** baseline.
- Estimated retrospective bad-rate reduction **{pct(cut['relative_bad_rate_reduction'])}**, rejecting **{pct(cut['good_customers_rejected_share'])}** of observed good applicants.
- **ILLUSTRATIVE** annual gross loss avoided **Rs {cut['estimated_annual_gross_loss_avoided_crore']:.2f} crore**; net benefit after foregone good margin **Rs {cut['estimated_annual_net_benefit_crore']:.2f} crore**. See editable assumptions below.

Every quantitative result above is generated from `reports/metrics.json`. Actual code
output is saved in `reports/*_execution.txt` and `reports/headline_metrics.txt`.

## Approach and validation design

SQLite loads all required source tables. SQL collapses monthly bureau records before
joining and partial installment payments before aggregating. Day-zero cutoffs and
prior completed monthly snapshots exclude future observations. Application count
and unique IDs are asserted after joining. Raw source SHA-256 hashes are retained.

Stratified random partitions separate model training, policy/calibration, test and
holdout. There is no verified application date: **no out-of-time or pseudo-OOT claim**.
The policy partition is reserved for cut-off selection, not probability calibration.
Numeric WoE bins are trained by quantiles and adjacent monotonic pooling; missing
values get a special bin. Train-only IV, correlation and VIF checks select the
scorecard. LightGBM candidate settings use training-only cross-validation. Raw
probability calibration is evaluated without fitting a calibrator.

The points scale uses configured base score {m['scorecard_design']['base_score']},
good/bad odds {m['scorecard_design']['base_good_bad_odds']}:1 and PDO
{m['scorecard_design']['pdo']}. Points = offset - factor x log odds of TARGET=1.
Exact unrounded point additivity is checked. SHAP plots explain log-odds contributions;
they do not establish causes. Gains tables, paired AUC bootstrap intervals and
score/key-feature PSI are included.

## Business insights (descriptive full labeled portfolio)

{insight}

These observed segments are descriptive; small segments need caution. The figures do
not establish causal effects and do not select features using holdout labels.

![Dashboard](reports/figures/dashboard_screenshot.png)

Open `dashboard/index.html` in a browser: all charts work offline. The cut-off simulator
recomputes exact holdout outcomes and illustrative economics as inputs change.
Post-hoc exploration must not replace the preselected policy without fresh validation.
`dashboard/powerbi_holdout.csv` is the Power BI import export.

## Run end to end (PowerShell, Python 3.11)

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
# Obtain the official data first: see DATA_DOWNLOAD.md.
.\\.venv\\Scripts\\python.exe scripts/download_data.py --check-only
.\\.venv\\Scripts\\python.exe scripts/run_pipeline.py --stage all
.\\.venv\\Scripts\\python.exe scripts/finalize_project.py
```

Individual stages are `sql`, `audit`, `model`, `simulation`, `deliverables`. SQL ingestion
is restartable using source hashes; feature aggregation and modeling are deterministic
rebuilds. Finalization executes notebooks, captures and checks the dashboard in installed
Chrome, renders PDFs using Poppler (`pdftoppm` must be on PATH), checks saved outputs,
and builds the ZIP. It uses headless Chrome directly, without a driver download.
The notebook series is an executed walkthrough of the artifacts; reusable full
implementation lives in `src/`, `sql/`, `early_warning/` and `dashboard/`.

## Economic assumptions (not measured outcomes)

Edit `config/project.json`: annual applications, average INR loan, LGD and margin on
good loans, plus minimum approval share. The chosen threshold maximizes illustrative
net benefit on the policy partition subject to the configured approval constraint.
Gross loss avoided = rejected bad share x assumed applications x loan size x LGD.
Net benefit subtracts rejected good share x assumed applications x loan size x good
margin. No Home Credit currency conversion, actual recovery data or realized saving
is implied. Recompute the model stage after changing assumptions to reselect policy.

## SIMULATED early warning

`early_warning/README.md` documents a seeded sales process with artificial events,
weekly seasonality, noise, declines and temporary shocks. Only applicant IDs link
to Home Credit; sales and outcomes are **SIMULATED**, independent of actual TARGET.
SIMULATED precision is **{pct(m['SIMULATED_early_warning']['precision'])}**, recall
**{pct(m['SIMULATED_early_warning']['recall'])}**, median lead time
**{m['SIMULATED_early_warning']['median_lead_days']:.0f} SIMULATED days** among detected events.
Built-in decline makes these method-demonstration results unsuitable as real findings.

## Deliverables and evidence

- `output/pdf/business_memo.pdf`: memo to DVP - Analytics.
- `output/pdf/interview_prep.pdf`: one-page interview sheet.
- `reports/resume_bullets.md`: measured, qualified resume claims.
- `reports/metrics.json`: machine-readable metric source of truth.
- `reports/scorecard_points.csv`, `woe_bins.csv`, `vif.csv`, `boosting_cv.csv`.
- `reports/gains_*.csv`, `calibration_*.csv`, `score_psi.json`, `key_feature_psi.json`.
- `models/`: fitted scorecard and LightGBM pipelines.
- `dist/msme_credit_risk_project.zip`: code, models, reports, dashboard and simulation;
  source data, database, environment and large processed intermediates are omitted.

## Limitations and next steps

Read **[all limitations and assumptions](reports/LIMITATIONS.md)**. Main limits are
population mismatch, withheld target horizon, random rather than temporal validation,
selection bias, unproven production feature availability, uncorrected calibration,
sensitive proxies, heuristic PSI thresholds and assumed economics. No causal policy
effect, fair-lending certification, or production readiness is claimed. Next steps
require actual MSME data, point-in-time features, temporal testing, business calibration
and independent model-risk review.

## Sources

- [Official competition](https://www.kaggle.com/competitions/home-credit-default-risk)
- Supplied `HomeCredit_columns_description.csv` (read with Windows-1252 encoding).
- [LightGBM parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html)
- [SHAP TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)
- [Kaggle CLI](https://github.com/Kaggle/kaggle-cli)
'''
    (ROOT/'README.md').write_text(text,encoding='utf-8')

def run():
    from dashboard.build import build
    build()
    m=json.loads((REPORTS/'metrics.json').read_text())
    memo,prep=documents(m)
    readme(m)
    (REPORTS/'headline_metrics.txt').write_text(json.dumps({
        'dataset':m['dataset'],'holdout_scorecard':m['validation']['scorecard']['holdout'],
        'holdout_lightgbm':m['validation']['lightgbm']['holdout'],
        'holdout_scorecard_psi':m['score_psi']['scorecard']['holdout'],
        'holdout_fixed_policy':m['cutoff']['evaluation']['holdout'],
        'economic_assumptions':m['cutoff']['assumptions'],'SIMULATED_early_warning':m['SIMULATED_early_warning']},indent=2),encoding='utf-8')
    print('STEP 11 COMPLETE: business memo generated from measured outputs: '+str(memo),flush=True)
    print('STEP 12 DRAFT COMPLETE: README, resume bullets and interview sheet generated; rendering and verification pending.',flush=True)
