"""Build and execute educational notebooks over the actual pipeline outputs."""
from pathlib import Path
import json
import os
import sys
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
os.environ['JUPYTER_RUNTIME_DIR']=str(ROOT/'.runtime/jupyter')
os.environ['IPYTHONDIR']=str(ROOT/'.runtime/ipython')
os.environ['JUPYTER_CONFIG_DIR']=str(ROOT/'.runtime/jupyter-config')
for env in ['JUPYTER_RUNTIME_DIR','IPYTHONDIR','JUPYTER_CONFIG_DIR']:
    Path(os.environ[env]).mkdir(parents=True,exist_ok=True)
KERNELS=ROOT/'.runtime/kernels'
kernel=KERNELS/'project';kernel.mkdir(parents=True,exist_ok=True)
(kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
                                             'display_name':'Project Python','language':'python'}),encoding='utf-8')
intro='''These cells were executed against the real project outputs. The complete training
and SQL implementation is in the linked source modules. This walkthrough reads and
verifies executed artifacts rather than silently retraining the expensive models.
Home Credit is a consumer-lending proxy; validation is random holdout, never true OOT.
'''
setup="""from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image
ROOT=Path.cwd()
if not (ROOT/'reports/metrics.json').exists():
    ROOT=ROOT.parent
metrics=json.loads((ROOT/'reports/metrics.json').read_text())
"""
notebooks=[
    ('01_data_audit','Data audit and label integrity',[
        ('Executed audit',"print(json.dumps(metrics['audit'],indent=2))\ndisplay(pd.read_csv(ROOT/'reports/missing_values.csv').head(20))\ndisplay(pd.read_csv(ROOT/'reports/numeric_audit.csv').head(12))"),
        ('Interpretation','''TARGET records repayment difficulty with provider-withheld overdue/timing thresholds.
Missing histories remain missing. Positive employment-day sentinel values were replaced
with null in SQL; outliers were measured, not silently deleted. See src/audit.py.''')]),
    ('02_eda_business_insights','Portfolio risk and business insights',[
        ('Segment outcomes',"segments=pd.read_csv(ROOT/'reports/segment_risk.csv')\ndisplay(segments)\ndisplay(Image(filename=str(ROOT/'reports/figures/audit_and_segments.png')))"),
        ('Interpretation','''Relative risk = segment TARGET rate / overall TARGET rate. Volume share uses the
full labeled application count. These are descriptive associations, not causal effects.
Small segment estimates are uncertain. This EDA does not choose features on holdout.''')]),
    ('03_woe_iv_binning','Train-only WoE, IV and monotonic bins',[
        ('WoE and IV',"display(pd.read_csv(ROOT/'reports/iv_ranking.csv').head(25))\nbins=pd.read_csv(ROOT/'reports/woe_bins.csv')\ndisplay(bins.head(25))\nassert bins.groupby('feature')['count'].sum().eq(metrics['validation']['scorecard']['train']['rows']).all()\nprint('All bin counts match model-training rows.')"),
        ('Binning method','''See src/woe_binning.py. Numeric training quantiles are merged for monotonic event
rates; missing has its own bin. Rare categories are pooled. WoE = ln(smoothed good share /
smoothed bad share); IV sums (good share - bad share) x WoE. Bin edges and selection never
use test/holdout labels. Missing/unseen categories and smoothing are documented.''')]),
    ('04_scorecard_model','Logistic regression to additive points',[
        ('Model and points',"print(json.dumps(metrics['scorecard_design'],indent=2))\ndisplay(pd.read_csv(ROOT/'reports/vif.csv'))\ndisplay(pd.read_csv(ROOT/'reports/scorecard_points.csv').head(30))"),
        ('Points formula','''For factor = PDO / ln(2), offset = base score - factor x ln(base good/bad odds),
score = offset - factor x logistic log odds of TARGET=1. Baseline constants are shared
across selected variables in the points table. Higher score means lower risk. Production
rounding needs separate boundary tests. Model additivity is numerically verified.''')]),
    ('05_gradient_boosting_shap','Cross-validation and SHAP explanations',[
        ('Training-only tuning',"cv=pd.read_csv(ROOT/'reports/boosting_cv.csv')\ndisplay(cv)\ndisplay(cv.groupby('candidate').auc.agg(['mean','std']))\ndisplay(pd.read_csv(ROOT/'reports/shap_global.csv').head(20))\ndisplay(Image(filename=str(ROOT/'reports/figures/shap_global.png')))\ndisplay(Image(filename=str(ROOT/'reports/figures/shap_local.png')))"),
        ('Interpretation','''See src/models.py. Category mappings are fitted separately inside each CV fold.
Tree-path-dependent SHAP explains log odds; positive local contributions increase modeled
risk. Attributions can be shared among correlated features and do not imply causality.
No holdout performance selected the candidate settings.''')]),
    ('06_validation','Discrimination, calibration and stability',[
        ('Actual validation tables',"for model,splits in metrics['validation'].items():\n    print(model)\n    display(pd.DataFrame(splits).T)\ndisplay(pd.read_csv(ROOT/'reports/gains_scorecard_holdout.csv'))\nprint(json.dumps(metrics['score_psi'],indent=2))\nprint(json.dumps(metrics['key_feature_psi'],indent=2))\nprint(json.dumps(metrics['holdout_auc_bootstrap_95pct'],indent=2))\ndisplay(Image(filename=str(ROOT/'reports/figures/validation.png')))"),
        ('Interpretation','''Gini = 2 x AUC - 1. KS is computed over all ROC thresholds; decile KS may be lower.
Calibration curves and Brier scores assess probability quality without fitting a
calibrator. PSI compares training-defined buckets with missing and half-count smoothing.
Heuristics: <0.1 stable, 0.1-0.25 moderate, >0.25 shifted. Random-split stability is not
time stability. Bootstrap intervals condition on the fitted models.''')]),
    ('07_cutoff_strategy','Fixed policy, estimated economics and SIMULATED alerts',[
        ('Policy selection and outcomes',"print(json.dumps(metrics['cutoff'],indent=2))\ndisplay(pd.read_csv(ROOT/'reports/cutoff_calibration.csv').head(20))\ndisplay(Image(filename=str(ROOT/'reports/figures/cutoff_and_distribution.png')))\nprint('SIMULATED EARLY WARNING - METHOD DEMONSTRATION ONLY')\nprint(json.dumps(metrics['SIMULATED_early_warning'],indent=2))\ndisplay(Image(filename=str(ROOT/'reports/figures/SIMULATED_early_warning.png')))"),
        ('Economics and governance','''Policy selection uses the reserved policy partition, maximizing illustrative
net benefit with a minimum approval constraint. Holdout metrics are evaluation only.
Bad-rate reduction is relative to the cohort baseline; good rejection share divides
rejected good applicants by all good applicants. Gross loss avoided and net benefit are
different. Edit config/project.json and rerun model stage to change the recommended policy.
All merchant outcomes, dates and sales in the extension are SIMULATED; built-in decline
creates predictability and cannot demonstrate real operational value.''')]),
]

def main():
    for name,title,sections in notebooks:
        print('STARTING '+name,flush=True)
        cells=[nbf.v4.new_markdown_cell('# '+title+'\n\n'+intro),nbf.v4.new_code_cell(setup)]
        for heading,body in sections:
            cells.append(nbf.v4.new_markdown_cell('## '+heading))
            cells.append(nbf.v4.new_code_cell(body) if heading not in ['Interpretation','Binning method','Points formula','Economics and governance'] else nbf.v4.new_markdown_cell(body))
        nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
        manager=KernelManager(kernel_name='project',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(KERNELS)]))
        try:
            NotebookClient(nb,timeout=120,startup_timeout=30,km=manager,resources={'metadata':{'path':str(ROOT)}}).execute()
        finally:
            if manager.has_kernel: manager.shutdown_kernel(now=True)
        nbf.write(nb,ROOT/'notebooks'/f'{name}.ipynb')
        print('EXECUTED '+name,flush=True)

if __name__=='__main__': main()
