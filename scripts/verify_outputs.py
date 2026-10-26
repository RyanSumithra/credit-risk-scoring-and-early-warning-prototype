"""Meaningful artifact checks against actual saved borrowers and fitted models."""
import json
import os
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from pypdf import PdfReader
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
from src.common import data,partitions,SEED,dump
from src.cutoff import strategy
from src.validation import evaluate

def main():
    m=json.loads((ROOT/'reports/metrics.json').read_text())
    df=data(); parts=partitions(df)
    scored=pd.read_csv(ROOT/'data/processed/scored_portfolio.csv').set_index('SK_ID_CURR')
    assert len(scored)==len(df) and scored.index.is_unique
    assert np.isfinite(scored[['score','scorecard_pd','lightgbm_pd']]).all().all()
    for split,idx in parts.items():
        subset=scored.loc[df.loc[idx,'SK_ID_CURR']]
        assert subset.split.eq(split).all()
        for model,col in [('scorecard','scorecard_pd'),('lightgbm','lightgbm_pd')]:
            computed=evaluate(subset.TARGET,subset[col])
            for key,value in computed.items():
                assert np.isclose(value,m['validation'][model][split][key]),(model,split,key)
    bins=pd.read_csv(ROOT/'reports/woe_bins.csv')
    assert bins.groupby('feature')['count'].sum().eq(len(parts['train'])).all()
    sample=df.loc[parts['holdout']].sample(100,random_state=SEED)
    saved=scored.loc[sample.SK_ID_CURR]
    bundle=joblib.load(ROOT/'models/scorecard.joblib')
    z=bundle['binner'].transform(sample,bundle['selected'])
    np.testing.assert_allclose(bundle['model'].predict_proba(z)[:,1],saved.scorecard_pd,atol=1e-12)
    np.testing.assert_allclose(bundle['offset']-bundle['factor']*bundle['model'].decision_function(z),saved.score,atol=1e-9)
    bundle=joblib.load(ROOT/'models/lightgbm.joblib')
    np.testing.assert_allclose(bundle['model'].predict_proba(bundle['mapper'].transform(sample))[:,1],saved.lightgbm_pd,atol=1e-12)
    held=scored[scored.split=='holdout']
    policy=strategy(held.TARGET,held.score,m['cutoff']['selected_cutoff'],m['cutoff']['assumptions'])
    for key,value in policy.items():
        assert np.isclose(value,m['cutoff']['evaluation']['holdout'][key]),key
    pages={name:len(PdfReader(ROOT/'output/pdf'/name).pages) for name in ['business_memo.pdf','interview_prep.pdf']}
    assert pages=={'business_memo.pdf':2,'interview_prep.pdf':1},pages
    image=Image.open(ROOT/'reports/figures/dashboard_screenshot.png')
    assert image.width>=1200 and image.height>=900
    for name in ['business_memo','interview_prep']:
        assert (ROOT/'reports/figures'/f'{name}-1.png').exists()
    notebooks=list((ROOT/'notebooks').glob('*.ipynb'))
    for path in notebooks:
        nb=json.loads(path.read_text())
        assert all(c.get('execution_count') is not None for c in nb['cells'] if c['cell_type']=='code')
        assert not any(o.get('output_type')=='error' for c in nb['cells'] for o in c.get('outputs',[]))
    report={'status':'passed','partition_and_saved_metric_checks':True,'bin_counts_training_only':True,
            'reloaded_model_predictions_match_saved_scores':True,'holdout_policy_recalculation':True,
            'pdf_pages':pages,'dashboard_screenshot_size':list(image.size),'executed_notebooks':len(notebooks)}
    dump(ROOT/'reports/verification.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
