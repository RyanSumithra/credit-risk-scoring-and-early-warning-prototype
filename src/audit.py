"""Descriptive audit; findings never drive test-set feature selection."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.common import ROOT, REPORTS, FIGURES, data, partitions, dump

def run():
    df = data()
    FIGURES.mkdir(exist_ok=True)
    missing = df.isna().mean().sort_values(ascending=False).rename('missing_share')
    missing.to_csv(REPORTS/'missing_values.csv')
    numeric = df.select_dtypes(include='number')
    desc = numeric.describe(percentiles=[.01,.25,.5,.75,.99]).T
    q1,q3 = numeric.quantile(.25),numeric.quantile(.75)
    desc['iqr_outlier_share'] = ((numeric.lt(q1-1.5*(q3-q1))) | numeric.gt(q3+1.5*(q3-q1))).mean()
    desc.to_csv(REPORTS/'numeric_audit.csv')
    raw_employment = pd.read_csv(ROOT/'data/raw/application_train.csv', usecols=['DAYS_EMPLOYED'])
    segments=[]
    for feature in ['NAME_INCOME_TYPE','NAME_CONTRACT_TYPE','NAME_HOUSING_TYPE','CODE_GENDER']:
        grouped = df.groupby(feature, dropna=False).TARGET.agg(['size','sum','mean']).reset_index()
        grouped.columns=['segment','applications','bads','bad_rate']
        grouped['feature']=feature
        grouped['volume_share']=grouped.applications/len(df)
        grouped['relative_risk']=grouped.bad_rate/df.TARGET.mean()
        segments.append(grouped)
    segment = pd.concat(segments, ignore_index=True)
    segment.to_csv(REPORTS/'segment_risk.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    missing.head(18).sort_values().plot.barh(ax=axes[0], color='#147d92')
    axes[0].set_title('Largest missing-value shares'); axes[0].set_xlabel('Share of applications')
    income=segment[segment.feature=='NAME_INCOME_TYPE'].sort_values('applications',ascending=False).head(6)
    axes[1].bar(income.segment,income.bad_rate,color='#e79b47')
    axes[1].axhline(df.TARGET.mean(),color='#102d43',linestyle='--',label='Portfolio')
    axes[1].tick_params(axis='x',rotation=35); axes[1].set_title('Observed TARGET rate by income type'); axes[1].legend()
    fig.tight_layout(); fig.savefig(FIGURES/'audit_and_segments.png',dpi=160); plt.close(fig)
    parts=partitions(df)
    split_rows=[]
    for name,idx in parts.items():
        split_rows.append({'split':name,'rows':len(idx),'bad_count':int(df.loc[idx,'TARGET'].sum()),'bad_rate':float(df.loc[idx,'TARGET'].mean())})
    assignment=pd.Series(index=df.index,dtype='str')
    for name,idx in parts.items(): assignment.loc[idx]=name
    pd.DataFrame({'SK_ID_CURR':df.SK_ID_CURR,'split':assignment}).to_csv(ROOT/'data/processed/split_assignment.csv',index=False)
    report={'rows':len(df),'features':len(df.columns)-2,'bad_count':int(df.TARGET.sum()),
            'bad_rate':float(df.TARGET.mean()),'duplicates':int(df.SK_ID_CURR.duplicated().sum()),
            'employment_sentinels_replaced':int(raw_employment.DAYS_EMPLOYED.eq(365243).sum()),
            'split_design':'Stratified random holdout; no chronology claim. Calibration partition carved from training pool.',
            'splits':split_rows,'income_segments':income.to_dict(orient='records')}
    dump(REPORTS/'audit.json',report)
    print('STEP 3 COMPLETE: AUDIT AND EDA\n'+json.dumps(report,indent=2),flush=True)
    print('STEP 4 COMPLETE: Disjoint stratified random train/calibration/test/holdout partitions saved. No pseudo-OOT claim.',flush=True)
