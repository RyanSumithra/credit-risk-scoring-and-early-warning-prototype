"""Observed cohort selection and explicitly illustrative portfolio economics."""
import numpy as np
import pandas as pd

def strategy(y,scores,cutoff,assumptions):
    y=np.asarray(y); scores=np.asarray(scores); accepted=scores>=cutoff
    rejected=~accepted; n=len(y); total_bad=y.sum(); total_good=n-total_bad
    accepted_bad=y[accepted].sum(); rejected_bad=y[rejected].sum(); rejected_good=rejected.sum()-rejected_bad
    baseline=total_bad/n
    rate=accepted_bad/accepted.sum() if accepted.sum() else None
    scale=assumptions['annual_applications']/n
    gross=rejected_bad*scale*assumptions['average_loan_inr']*assumptions['lgd']
    foregone=rejected_good*scale*assumptions['average_loan_inr']*assumptions['good_loan_margin']
    return {'cutoff':float(cutoff),'applications':n,'accepted':int(accepted.sum()),
            'accepted_bads':int(accepted_bad),'rejected_bads':int(rejected_bad),'rejected_goods':int(rejected_good),
            'approval_rate':float(accepted.mean()),'baseline_bad_rate':float(baseline),
            'accepted_bad_rate':None if rate is None else float(rate),
            'relative_bad_rate_reduction':None if rate is None else float(1-rate/baseline),
            'good_customers_rejected_share':float(rejected_good/total_good),
            'estimated_annual_gross_loss_avoided_crore':float(gross/1e7),
            'estimated_annual_good_margin_foregone_crore':float(foregone/1e7),
            'estimated_annual_net_benefit_crore':float((gross-foregone)/1e7)}

def tradeoff(y,scores,assumptions,cutoffs):
    return pd.DataFrame([strategy(y,scores,c,assumptions) for c in cutoffs])
