"""Discrimination, calibration, gains and train-reference stability."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, log_loss
from sklearn.calibration import calibration_curve

def evaluate(y,p):
    auc=roc_auc_score(y,p)
    fpr,tpr,_=roc_curve(y,p)
    return {'rows':len(y),'bad_count':int(np.sum(y)),'bad_rate':float(np.mean(y)),
            'auc':float(auc),'gini':float(2*auc-1),'ks':float(max(tpr-fpr)),
            'brier':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,p)),
            'mean_predicted_pd':float(np.mean(p))}

def gains(y,p):
    frame=pd.DataFrame({'y':np.asarray(y),'p':p}).sort_values('p',ascending=False)
    frame['decile']=pd.qcut(np.arange(len(frame)),10,labels=False)+1
    table=frame.groupby('decile').agg(rows=('y','size'),bads=('y','sum'),min_pd=('p','min'),max_pd=('p','max'))
    table['goods']=table.rows-table.bads
    table['bad_rate']=table.bads/table.rows
    table['cumulative_bad_share']=table.bads.cumsum()/table.bads.sum()
    table['cumulative_good_share']=table.goods.cumsum()/table.goods.sum()
    table['ks_at_decile']=table.cumulative_bad_share-table.cumulative_good_share
    table['cumulative_lift']=table.cumulative_bad_share/(table.rows.cumsum()/table.rows.sum())
    return table.reset_index()

def psi(reference,current,n_bins=10):
    reference=np.asarray(reference,dtype=float); current=np.asarray(current,dtype=float)
    finite=reference[np.isfinite(reference)]
    edges=np.unique(np.quantile(finite,np.linspace(0,1,n_bins+1)))[1:-1] if len(finite) else np.array([])
    def distribution(values):
        labels=np.searchsorted(edges,values,side='left')
        labels[~np.isfinite(values)]=len(edges)+1
        counts=np.bincount(labels,minlength=len(edges)+2).astype(float)
        return (counts+.5)/(len(values)+.5*len(counts))
    a,b=distribution(reference),distribution(current)
    value=float(np.sum((b-a)*np.log(b/a)))
    label='stable' if value<.1 else 'moderate' if value<=.25 else 'shifted'
    return {'psi':value,'interpretation':label,'reference_rows':len(reference),'current_rows':len(current),
            'reference_bin_edges':edges.tolist(),'reference_share':a.tolist(),'current_share':b.tolist()}

def bootstrap_intervals(y,scorecard,boosting,seed,repeats=200):
    rng=np.random.default_rng(seed); y=np.asarray(y)
    # Paired bootstrap: identical resampled borrowers for both models.
    values=[]
    for _ in range(repeats):
        idx=rng.integers(0,len(y),len(y))
        a=roc_auc_score(y[idx],scorecard[idx]); b=roc_auc_score(y[idx],boosting[idx])
        values.append((a,b,b-a))
    bounds=np.quantile(values,[.025,.975],axis=0)
    return {name:{'low':float(bounds[0,i]),'high':float(bounds[1,i])} for i,name in enumerate(['scorecard_auc','boosting_auc','boosting_minus_scorecard_auc'])}
