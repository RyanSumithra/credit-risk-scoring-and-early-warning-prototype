"""SIMULATED merchant sales. All outcomes/dates here are artificial."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.common import ROOT,REPORTS,FIGURES,SEED,dump

def run():
    cfg=json.loads((ROOT/'config/project.json').read_text())['simulation']
    rng=np.random.default_rng(SEED)
    ids=pd.read_parquet(ROOT/'data/processed/modeling.parquet',columns=['SK_ID_CURR']).sample(cfg['borrowers'],random_state=SEED).SK_ID_CURR.to_numpy()
    days=np.arange(cfg['days']); daily=[]; outcomes=[]
    for borrower in ids:
        event=bool(rng.random()<cfg['event_probability'])
        event_day=int(rng.integers(110,171)) if event else -1
        baseline=rng.lognormal(np.log(15000),.65)
        weekly=1+.15*np.sin(2*np.pi*days/7+rng.uniform(0,2*np.pi))
        organic=np.exp(rng.normal(0,.0007)*days)
        noise=np.exp(rng.normal(-.5*.18**2,.18,len(days)))
        stress=np.ones(len(days))
        if event:
            stress=1-.65*np.clip((days-(event_day-55))/55,0,1)
        elif rng.random()<.25:
            shock_start=int(rng.integers(75,141)); duration=int(rng.integers(20,46))
            stress[(days>=shock_start)&(days<shock_start+duration)]=rng.uniform(.5,.7)
        sales=baseline*weekly*organic*noise*stress
        current=pd.Series(sales).rolling(30).mean()
        previous=current.shift(30)
        decline=1-current/previous
        condition=decline>cfg['trend_threshold']
        persistent=condition.rolling(cfg['persistence_days']).sum().eq(cfg['persistence_days'])
        eligible=persistent.to_numpy() & (days>=59)
        if event: eligible &= days<event_day
        alert_day=int(days[eligible][0]) if eligible.any() else -1
        outcomes.append({'SK_ID_CURR':int(borrower),'SIMULATED_event':int(event),'SIMULATED_event_day':event_day,
                         'SIMULATED_first_alert_day':alert_day,'SIMULATED_alert':int(alert_day>=0),
                         'SIMULATED_lead_days':event_day-alert_day if event and alert_day>=0 else None})
        daily.append(pd.DataFrame({'SK_ID_CURR':borrower,'SIMULATED_day':days,'SIMULATED_daily_sales_inr':sales,
                                   'SIMULATED_30day_decline':decline,'SIMULATED_flag':persistent.astype(int)}))
    daily=pd.concat(daily,ignore_index=True); outcome=pd.DataFrame(outcomes)
    daily.to_csv(ROOT/'early_warning/SIMULATED_daily_sales.csv.gz',index=False,compression='gzip')
    outcome.to_csv(ROOT/'early_warning/SIMULATED_outcomes.csv',index=False)
    tp=int(((outcome.SIMULATED_event==1)&(outcome.SIMULATED_alert==1)).sum())
    fp=int(((outcome.SIMULATED_event==0)&(outcome.SIMULATED_alert==1)).sum())
    fn=int(((outcome.SIMULATED_event==1)&(outcome.SIMULATED_alert==0)).sum())
    report={'label':'SIMULATED - METHOD DEMONSTRATION, NOT REAL MERCHANT FINDINGS',
            'borrowers':len(outcome),'daily_rows':len(daily),'simulated_events':int(outcome.SIMULATED_event.sum()),
            'true_positives':tp,'false_positives':fp,'false_negatives':fn,
            'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.,
            'median_lead_days':float(outcome.SIMULATED_lead_days.median()),
            'threshold':cfg['trend_threshold'],'persistence_days':cfg['persistence_days'],
            'outcome_definition':'Independent artificial events; real Home Credit TARGET is not used in this simulation.'}
    dump(REPORTS/'SIMULATED_early_warning.json',report)
    metrics=json.loads((REPORTS/'metrics.json').read_text()); metrics['SIMULATED_early_warning']=report
    dump(REPORTS/'metrics.json',metrics)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    detected=outcome[(outcome.SIMULATED_event==1)&(outcome.SIMULATED_alert==1)].iloc[0]
    series=daily[daily.SK_ID_CURR==detected.SK_ID_CURR]
    axes[0].plot(series.SIMULATED_day,series.SIMULATED_daily_sales_inr,color='#147d92',alpha=.5)
    axes[0].axvline(detected.SIMULATED_first_alert_day,color='#e79b47',label='First alert')
    axes[0].axvline(detected.SIMULATED_event_day,color='#a32638',label='Artificial event')
    axes[0].set_title('SIMULATED | example merchant sales'); axes[0].set_xlabel('SIMULATED day'); axes[0].legend()
    axes[1].hist(outcome.SIMULATED_lead_days.dropna(),bins=15,color='#147d92')
    axes[1].set_title('SIMULATED | lead time among detected events'); axes[1].set_xlabel('SIMULATED days before event')
    fig.tight_layout(); fig.savefig(FIGURES/'SIMULATED_early_warning.png',dpi=160); plt.close(fig)
    (ROOT/'early_warning/README.md').write_text('''# SIMULATED early-warning demonstration

Every event, event date and sales value in this directory is SIMULATED. Real applicant
IDs are sampled only to demonstrate linkage; real TARGET does not generate these outcomes.

The seeded process assigns independent artificial events with configurable probability.
Baseline daily sales are lognormal (median INR 15,000; log standard deviation 0.65),
with 15% weekly seasonality, small random organic slopes, and lognormal daily noise
(log standard deviation 0.18, mean corrected). Event cases decline linearly from
55 days before their artificial event to 35% of baseline at the event. A quarter of
non-event cases receive temporary 30-50% sales shocks lasting 20-45 days. This
deliberately generates overlap and false positives rather than perfect separation.

A fixed flag compares the most recent 30-day average with the preceding 30 days;
a decline above the configured threshold must persist for the configured days.
Only past/current sales enter a flag. Evaluation uses first alerts before artificial
events; non-event borrowers are observed over the whole artificial horizon. Precision
is event borrowers / all flagged borrowers; recall is detected / all event borrowers.
Median lead time is event day minus first alert among detected event borrowers.

The decline mechanism builds predictability into the simulation. These results cannot
establish real-world accuracy, actual default lead time, or operational value. The
threshold is fixed in config before generation; no tuning or external validation occurs.
''',encoding='utf-8')
    print('STEP 9 COMPLETE: '+json.dumps(report,indent=2),flush=True)
