"""Train-only supervised monotonic binning and smoothed good/bad WoE."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

class WoEBinner:
    def __init__(self,max_bins=10,min_category_share=.01):
        self.max_bins=max_bins
        self.min_category_share=min_category_share

    def fit(self,X,y):
        self.specs_={}; self.tables_=[]; self.iv_={}
        y=np.asarray(y)
        for name in X.columns:
            s=X[name]
            if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True)>12:
                values=pd.to_numeric(s,errors='coerce').to_numpy(dtype=float)
                edges=np.unique(np.nanquantile(values,np.linspace(0,1,self.max_bins+1)))[1:-1]
                # Pool adjacent violations of the empirical event-rate order.
                while True:
                    labels=np.searchsorted(edges,values,side='left')
                    valid=np.isfinite(values)
                    rates=pd.DataFrame({'bin':labels[valid],'y':y[valid]}).groupby('bin').y.mean()
                    if len(rates)<3: break
                    direction=1 if spearmanr(rates.index,rates.values).statistic>=0 else -1
                    violation=np.where(direction*np.diff(rates.values)<-1e-12)[0]
                    if not len(violation): break
                    edges=np.delete(edges,int(rates.index[violation[0]]))
                spec={'kind':'numeric','edges':edges.tolist()}
            else:
                strings=s.astype('string').fillna('__MISSING__')
                counts=strings.value_counts()
                keep=counts[counts>=len(s)*self.min_category_share].head(30).index.tolist()
                spec={'kind':'categorical','keep':keep}
            labels=self._labels(s,spec)
            table=pd.DataFrame({'bin':labels,'bad':y}).groupby('bin',dropna=False).bad.agg(['size','sum']).reset_index()
            table.columns=['bin','count','bad']
            table['good']=table['count']-table.bad
            k=len(table)
            table['good_dist']=(table.good+.5)/(sum(y==0)+.5*k)
            table['bad_dist']=(table.bad+.5)/(sum(y==1)+.5*k)
            table['woe']=np.log(table.good_dist/table.bad_dist)
            table['iv_component']=(table.good_dist-table.bad_dist)*table.woe
            table['bad_rate']=table.bad/table['count']
            table['feature']=name
            table['bin_definition']=table['bin'].map(lambda b:self._definition(b,spec))
            spec['woe']={str(b):float(w) for b,w in zip(table.bin,table.woe)}
            self.specs_[name]=spec
            self.iv_[name]=float(table.iv_component.sum())
            self.tables_.append(table)
        self.table_=pd.concat(self.tables_,ignore_index=True)
        return self

    @staticmethod
    def _labels(s,spec):
        if spec['kind']=='numeric':
            a=pd.to_numeric(s,errors='coerce').to_numpy(dtype=float)
            labels=pd.Series(np.searchsorted(spec['edges'],a,side='left').astype(str),index=s.index)
            return labels.mask(~np.isfinite(a),'__MISSING__')
        strings=s.astype('string').fillna('__MISSING__')
        return strings.where(strings.isin(spec['keep']),'__OTHER__')

    @staticmethod
    def _definition(label,spec):
        if spec['kind']!='numeric' or label=='__MISSING__': return str(label)
        edges=[-np.inf]+spec['edges']+[np.inf]
        i=int(label)
        return f'({edges[i]:.8g}, {edges[i+1]:.8g}]'

    def transform(self,X,columns=None):
        columns=list(self.specs_) if columns is None else columns
        return pd.DataFrame({c:self._labels(X[c],self.specs_[c]).map(self.specs_[c]['woe']).fillna(0).astype(float) for c in columns},index=X.index)
