"""Self-contained offline dashboard with exact holdout threshold calculations."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs
from src.common import ROOT,REPORTS,dump

def build():
    metrics=json.loads((REPORTS/'metrics.json').read_text())
    portfolio=pd.read_csv(ROOT/'data/processed/scored_portfolio.csv')
    holdout=portfolio[portfolio.split=='holdout'].copy()
    holdout.to_csv(ROOT/'dashboard/powerbi_holdout.csv',index=False)
    all_segments=holdout.groupby('NAME_INCOME_TYPE').TARGET.agg(['size','mean']).sort_values('size',ascending=False).reset_index()
    all_segments.to_csv(ROOT/'dashboard/powerbi_segment_kpis.csv',index=False)
    # Avoid letting tiny groups with extreme empirical rates dominate the chart.
    segment=all_segments[all_segments['size']>=100]
    counts=[]
    edges=np.linspace(holdout.score.min(),holdout.score.max(),41)
    centers=((edges[:-1]+edges[1:])/2).tolist()
    for label in [0,1]: counts.append(np.histogram(holdout.loc[holdout.TARGET==label,'score'],bins=edges)[0].tolist())
    payload={
        'scores':holdout.score.tolist(),'target':holdout.TARGET.tolist(),
        'segments':segment.NAME_INCOME_TYPE.tolist(),'segmentRate':segment['mean'].tolist(),'segmentVolume':segment['size'].tolist(),
        'centers':centers,'goods':counts[0],'bads':counts[1],
        'recommended':metrics['cutoff']['selected_cutoff'],'assumptions':metrics['cutoff']['assumptions'],
        'auc':metrics['validation']['scorecard']['holdout']['auc'],'boostAuc':metrics['validation']['lightgbm']['holdout']['auc'],
        'psi':metrics['score_psi']['scorecard']['holdout']['psi'],
    }
    template=(ROOT/'dashboard/template.html').read_text(encoding='utf-8')
    html=template.replace('__PLOTLY__',get_plotlyjs()).replace('__DATA__',json.dumps(payload))
    path=ROOT/'dashboard/index.html'; path.write_text(html,encoding='utf-8')
    info={'tool':'Self-contained interactive HTML / Plotly / JavaScript','holdout_rows':len(holdout),
          'html_bytes':path.stat().st_size,'recommended_cutoff':payload['recommended'],
          'segment_chart_minimum_applications':100,'segment_chart_included_groups':len(segment),
          'segment_chart_excluded_groups':len(all_segments)-len(segment)}
    metrics['dashboard']=info; dump(REPORTS/'metrics.json',metrics)
    print('STEP 10 DASHBOARD BUILT: '+json.dumps(info),flush=True)
    return info
