"""Train-only scorecard development and cross-validated LightGBM benchmark."""
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
from lightgbm import LGBMClassifier
import shap
from src.common import ROOT, REPORTS, FIGURES, SEED, data, partitions, dump
from src.woe_binning import WoEBinner
from src.validation import evaluate, gains, psi, bootstrap_intervals
from src.cutoff import strategy, tradeoff

class CategoryMapper:
    """Native categorical codes; mappings learned on the current training fold."""
    def fit(self,X):
        self.columns=list(X.columns)
        self.categories={c:{v:i for i,v in enumerate(sorted(X[c].dropna().astype(str).unique()))}
                         for c in X.select_dtypes(exclude='number').columns}
        return self
    def transform(self,X):
        result=X[self.columns].copy()
        for c,mapping in self.categories.items():
            result[c]=result[c].astype('string').map(mapping).fillna(-1).astype('int32')
        return result.astype('float64').replace([np.inf,-np.inf],np.nan)

def collinearity(z,ranked,config):
    selected=[]; rejected=[]
    for c in ranked:
        if len(selected)>=config['max_features']:
            rejected.append({'feature':c,'reason':'maximum feature count'}); continue
        corr=z[selected].corrwith(z[c]).abs() if selected else pd.Series(dtype=float)
        if len(corr) and corr.max()>config['correlation_limit']:
            rejected.append({'feature':c,'reason':'correlation','max_abs_correlation':float(corr.max())})
        else: selected.append(c)
    while len(selected)>1:
        correlation=z[selected].corr().to_numpy()
        if np.linalg.matrix_rank(correlation)<len(selected):
            rejected.append({'feature':selected.pop(),'reason':'singular correlation matrix'}); continue
        vifs=np.diag(np.linalg.inv(correlation))
        if vifs.max()<=config['vif_limit']: break
        worst=int(np.argmax(vifs))
        rejected.append({'feature':selected.pop(worst),'reason':'VIF','vif':float(vifs[worst])})
    vifs=np.diag(np.linalg.inv(z[selected].corr().to_numpy())) if len(selected)>1 else [1.]
    return selected,pd.DataFrame({'feature':selected,'vif':vifs}),rejected

def run():
    started=time.time()
    cfg=json.loads((ROOT/'config/project.json').read_text())
    df=data(); parts=partitions(df)
    X=df.drop(columns=['SK_ID_CURR','TARGET','CODE_GENDER'])
    y=df.TARGET.astype(int)
    # Apply eligibility rules using model-training data only.
    train=parts['train']
    eligible=[c for c in X if X.loc[train,c].isna().mean()<.95 and X.loc[train,c].nunique(dropna=True)>1]
    X=X[eligible].replace([np.inf,-np.inf],np.nan)
    binner=WoEBinner().fit(X.loc[train],y.loc[train])
    iv=pd.Series(binner.iv_,name='iv').sort_values(ascending=False)
    iv.to_csv(REPORTS/'iv_ranking.csv')
    binner.table_.to_csv(REPORTS/'woe_bins.csv',index=False)
    ranked=iv[(iv>=cfg['scorecard']['iv_min'])&(iv<=cfg['scorecard']['iv_max'])].index.tolist()
    z=binner.transform(X,ranked)
    selected,vif,rejected=collinearity(z.loc[train],ranked,cfg['scorecard'])
    assert selected, 'No eligible scorecard features; cannot continue honestly.'
    vif.to_csv(REPORTS/'vif.csv',index=False)
    dump(REPORTS/'feature_selection_rejections.json',rejected)
    z[selected].loc[train].corr().to_csv(REPORTS/'scorecard_feature_correlations.csv')
    scorecard=LogisticRegression(C=1.0,solver='lbfgs',max_iter=1000)
    scorecard.fit(z.loc[train,selected],y.loc[train])
    assert scorecard.n_iter_.max()<1000, 'Scorecard did not converge.'
    factor=cfg['scorecard']['pdo']/np.log(2)
    offset=cfg['scorecard']['base_score']-factor*np.log(cfg['scorecard']['base_good_bad_odds'])
    points=binner.table_[binner.table_.feature.isin(selected)].copy()
    coefs=dict(zip(selected,scorecard.coef_[0]))
    points['coefficient']=points.feature.map(coefs)
    points['points']=((offset-factor*scorecard.intercept_[0])/len(selected))-factor*points.coefficient*points.woe
    points.to_csv(REPORTS/'scorecard_points.csv',index=False)
    # Exact unrounded additivity check on actual held-out applicants.
    sample=parts['holdout'][:100]
    direct=offset-factor*scorecard.decision_function(z.loc[sample,selected])
    reconstructed=np.zeros(len(sample))
    for c in selected:
        reconstructed+=(offset-factor*scorecard.intercept_[0])/len(selected)-factor*coefs[c]*z.loc[sample,c].to_numpy()
    error=float(np.max(np.abs(direct-reconstructed)))
    assert error<1e-8
    scorecard_info={'selected_features':selected,'selected_feature_count':len(selected),'eligible_features':len(eligible),
                    'factor':float(factor),'offset':float(offset),'base_score':cfg['scorecard']['base_score'],
                    'pdo':cfg['scorecard']['pdo'],'base_good_bad_odds':cfg['scorecard']['base_good_bad_odds'],
                    'intercept':float(scorecard.intercept_[0]),'coefficient':coefs,'max_additivity_error':error,
                    'positive_woe_coefficient_features':[c for c in selected if coefs[c]>0]}
    dump(REPORTS/'scorecard_design.json',scorecard_info)
    joblib.dump({'binner':binner,'selected':selected,'model':scorecard,'factor':factor,'offset':offset},ROOT/'models/scorecard.joblib')
    print('STEP 5 COMPLETE: '+json.dumps(scorecard_info),flush=True)

    candidates=[
        {'num_leaves':15,'min_child_samples':150,'reg_lambda':2.,'n_estimators':300},
        {'num_leaves':31,'min_child_samples':150,'reg_lambda':2.,'n_estimators':300},
        {'num_leaves':15,'min_child_samples':300,'reg_lambda':5.,'n_estimators':500},
        {'num_leaves':31,'min_child_samples':300,'reg_lambda':5.,'n_estimators':500},
    ]
    base={'objective':'binary','learning_rate':.04,'colsample_bytree':.85,'subsample':.85,'subsample_freq':1,
          'random_state':SEED,'n_jobs':4,'verbosity':-1,'deterministic':True,'force_col_wise':True}
    fold=StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED)
    cv_rows=[]
    trainX=X.loc[train]; trainy=y.loc[train]
    for candidate_id,params in enumerate(candidates):
        for fold_id,(a,b) in enumerate(fold.split(trainX,trainy)):
            mapper=CategoryMapper().fit(trainX.iloc[a])
            classifier=LGBMClassifier(**base,**params)
            classifier.fit(mapper.transform(trainX.iloc[a]),trainy.iloc[a],categorical_feature=list(mapper.categories))
            auc=roc_auc_score(trainy.iloc[b],classifier.predict_proba(mapper.transform(trainX.iloc[b]))[:,1])
            cv_rows.append({'candidate':candidate_id,'fold':fold_id,'auc':float(auc),**params})
            print('CV '+json.dumps(cv_rows[-1]),flush=True)
    cv=pd.DataFrame(cv_rows); cv.to_csv(REPORTS/'boosting_cv.csv',index=False)
    winner=int(cv.groupby('candidate').auc.mean().idxmax())
    mapper=CategoryMapper().fit(trainX)
    encoded=mapper.transform(X)
    boosting=LGBMClassifier(**base,**candidates[winner])
    boosting.fit(encoded.loc[train],y.loc[train],categorical_feature=list(mapper.categories))
    joblib.dump({'mapper':mapper,'model':boosting},ROOT/'models/lightgbm.joblib')
    dump(REPORTS/'boosting_design.json',{'model':'LightGBM','winning_candidate':winner,'parameters':{**base,**candidates[winner]},
                                       'mean_cv_auc':cv.groupby('candidate').auc.mean().to_dict()})
    shap_sample=X.loc[parts['holdout']].sample(n=min(1200,len(parts['holdout'])),random_state=SEED)
    shap_input=mapper.transform(shap_sample)
    explainer=shap.TreeExplainer(boosting,feature_perturbation='tree_path_dependent',model_output='raw')
    explanation=explainer(shap_input)
    contributions=np.asarray(explanation.values)
    if contributions.ndim==3: contributions=contributions[:,:,1]
    expected=np.asarray(explanation.base_values)
    if expected.ndim>1: expected=expected[:,1]
    shap_error=float(np.max(np.abs(expit(expected+contributions.sum(axis=1))-boosting.predict_proba(shap_input)[:,1])))
    assert shap_error<1e-6
    global_shap=pd.Series(np.abs(contributions).mean(axis=0),index=X.columns,name='mean_abs_shap_log_odds').sort_values(ascending=False)
    global_shap.to_csv(REPORTS/'shap_global.csv')
    fig,ax=plt.subplots(figsize=(9,6)); global_shap.head(18).sort_values().plot.barh(ax=ax,color='#147d92')
    ax.set_title('LightGBM | global Tree SHAP'); ax.set_xlabel('Mean absolute contribution to log odds of TARGET=1')
    fig.tight_layout(); fig.savefig(FIGURES/'shap_global.png',dpi=160); plt.close(fig)
    local_row=int(np.argmax(boosting.predict_proba(shap_input)[:,1]))
    local=pd.DataFrame({'feature':X.columns,'value':shap_input.iloc[local_row].to_numpy(),'shap_log_odds':contributions[local_row]})
    local['SK_ID_CURR']=int(df.loc[shap_sample.index[local_row],'SK_ID_CURR'])
    local.to_csv(REPORTS/'shap_local.csv',index=False)
    top=local.reindex(local.shap_log_odds.abs().sort_values(ascending=False).head(12).index).sort_values('shap_log_odds')
    fig,ax=plt.subplots(figsize=(9,5)); ax.barh(top.feature,top.shap_log_odds,color=['#e79b47' if v>0 else '#147d92' for v in top.shap_log_odds])
    ax.set_title('Local SHAP | highest-risk applicant in fixed holdout sample'); ax.set_xlabel('Contribution to log odds; positive increases modeled risk')
    fig.tight_layout(); fig.savefig(FIGURES/'shap_local.png',dpi=160); plt.close(fig)
    print('STEP 6 COMPLETE: LightGBM CV and global/local SHAP; additivity error='+str(shap_error),flush=True)

    all_p={'scorecard':scorecard.predict_proba(z[selected])[:,1],'lightgbm':boosting.predict_proba(encoded)[:,1]}
    scores=offset-factor*scorecard.decision_function(z[selected])
    results={}; stability={}; output=[]
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for model_name,p in all_p.items():
        results[model_name]={}; stability[model_name]={}
        for split,idx in parts.items():
            results[model_name][split]=evaluate(y.loc[idx],p[idx])
            gains(y.loc[idx],p[idx]).to_csv(REPORTS/f'gains_{model_name}_{split}.csv',index=False)
            obs,pred=calibration_curve(y.loc[idx],p[idx],n_bins=10,strategy='quantile')
            pd.DataFrame({'mean_predicted_pd':pred,'observed_bad_rate':obs}).to_csv(REPORTS/f'calibration_{model_name}_{split}.csv',index=False)
            stability[model_name][split]=psi(scores[train] if model_name=='scorecard' else all_p[model_name][train],scores[idx] if model_name=='scorecard' else p[idx])
        idx=parts['holdout']
        fpr,tpr,_=roc_curve(y.loc[idx],p[idx]); axes[0].plot(fpr,tpr,label=f'{model_name}: AUC {results[model_name]["holdout"]["auc"]:.3f}')
        obs,pred=calibration_curve(y.loc[idx],p[idx],n_bins=10,strategy='quantile'); axes[1].plot(pred,obs,'o-',label=model_name)
    axes[0].plot([0,1],[0,1],'k--',alpha=.4); axes[0].set(xlabel='False positive rate',ylabel='True positive rate',title='Untouched random holdout | ROC'); axes[0].legend()
    axes[1].plot([0,.35],[0,.35],'k--',alpha=.4); axes[1].set(xlabel='Mean predicted PD',ylabel='Observed TARGET rate',title='Holdout | raw-probability calibration'); axes[1].legend()
    fig.tight_layout(); fig.savefig(FIGURES/'validation.png',dpi=160); plt.close(fig)
    key_psi={c:psi(z.loc[train,c],z.loc[parts['holdout'],c]) for c in selected[:10]}
    dump(REPORTS/'key_feature_psi.json',{'representation':'Train-fitted WoE values; neutral fallback for unseen levels','features':key_psi})
    dump(REPORTS/'score_psi.json',stability)
    confidence=bootstrap_intervals(y.loc[parts['holdout']],all_p['scorecard'][parts['holdout']],all_p['lightgbm'][parts['holdout']],SEED)
    dump(REPORTS/'holdout_auc_bootstrap.json',confidence)
    print('STEP 7 COMPLETE: '+json.dumps({'validation':results,'holdout_score_psi':stability['scorecard']['holdout'],'paired_bootstrap':confidence}),flush=True)

    grid=cfg['cutoff_grid']
    cutoffs=np.arange(grid['start'],grid['stop'],grid['step'])
    policy_idx=parts['calibration']
    frontier=tradeoff(y.loc[policy_idx],scores[policy_idx],cfg['economics'],cutoffs)
    frontier.to_csv(REPORTS/'cutoff_calibration.csv',index=False)
    feasible=frontier[frontier.approval_rate>=cfg['economics']['minimum_approval_rate']]
    chosen=feasible.sort_values(['estimated_annual_net_benefit_crore','approval_rate'],ascending=[False,False]).iloc[0]
    chosen_cutoff=float(chosen.cutoff)
    policy={'selected_using':'Calibration/policy partition only; maximize assumed net benefit subject to minimum approval',
            'assumptions':cfg['economics'],'selected_cutoff':chosen_cutoff,'evaluation':{}}
    for split,idx in parts.items():
        policy['evaluation'][split]=strategy(y.loc[idx],scores[idx],chosen_cutoff,cfg['economics'])
        tradeoff(y.loc[idx],scores[idx],cfg['economics'],cutoffs).to_csv(REPORTS/f'cutoff_{split}.csv',index=False)
        temp=df.loc[idx,['SK_ID_CURR','TARGET','NAME_INCOME_TYPE','NAME_CONTRACT_TYPE','age_years']].copy()
        temp['split']=split; temp['score']=scores[idx]; temp['scorecard_pd']=all_p['scorecard'][idx]; temp['lightgbm_pd']=all_p['lightgbm'][idx]
        temp['accepted_at_fixed_cutoff']=scores[idx]>=chosen_cutoff; output.append(temp)
    pd.concat(output).to_csv(ROOT/'data/processed/scored_portfolio.csv',index=False)
    dump(REPORTS/'cutoff_strategy.json',policy)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    idx=parts['holdout']; axes[0].hist([scores[idx][y.loc[idx].to_numpy()==0],scores[idx][y.loc[idx].to_numpy()==1]],bins=35,density=True,label=['TARGET=0','TARGET=1'],color=['#147d92','#e79b47'],alpha=.7)
    axes[0].axvline(chosen_cutoff,color='#102d43',linestyle='--'); axes[0].set_title('Holdout score distribution'); axes[0].legend()
    holdout_frontier=tradeoff(y.loc[idx],scores[idx],cfg['economics'],cutoffs)
    axes[1].plot(holdout_frontier.approval_rate,holdout_frontier.accepted_bad_rate,color='#147d92'); axes[1].set(xlabel='Approval share',ylabel='Accepted TARGET rate',title='Holdout | fixed-grid descriptive trade-off')
    fig.tight_layout(); fig.savefig(FIGURES/'cutoff_and_distribution.png',dpi=160); plt.close(fig)
    print('STEP 8 COMPLETE: '+json.dumps(policy),flush=True)
    metrics={'status':'computed','dataset':json.loads((REPORTS/'sql_layer.json').read_text()),
             'ingestion':json.loads((REPORTS/'ingestion.json').read_text()),'audit':json.loads((REPORTS/'audit.json').read_text()),
             'scorecard_design':scorecard_info,'benchmark':'LightGBM','boosting_cv':cv_rows,
             'validation':results,'score_psi':stability,'key_feature_psi':key_psi,'holdout_auc_bootstrap_95pct':confidence,
             'shap':{'sample_rows':len(shap_sample),'max_probability_additivity_error':shap_error,'top_features':global_shap.head(10).to_dict()},
             'cutoff':policy,'model_stage_seconds':time.time()-started}
    dump(REPORTS/'metrics.json',metrics)
    print('MODEL STAGE COMPLETE: reports/metrics.json saved.',flush=True)
