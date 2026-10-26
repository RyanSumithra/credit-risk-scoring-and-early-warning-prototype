"""Shared paths, reproducible split, and JSON serialization."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'
FIGURES = REPORTS / 'figures'
SEED = 20261026

def dump(path, value):
    def convert(obj):
        if isinstance(obj, np.generic): return obj.item()
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, Path): return str(obj)
        raise TypeError(type(obj).__name__)
    Path(path).write_text(json.dumps(value, indent=2, default=convert, allow_nan=False), encoding='utf-8')

def data():
    return pd.read_parquet(ROOT / 'data/processed/modeling.parquet')

def partitions(df):
    development, holdout = train_test_split(df.index, test_size=.2, stratify=df.TARGET, random_state=SEED)
    train_pool, test = train_test_split(development, test_size=.25, stratify=df.loc[development,'TARGET'], random_state=SEED)
    train, calibration = train_test_split(train_pool, test_size=.2, stratify=df.loc[train_pool,'TARGET'], random_state=SEED)
    parts = {'train':train, 'calibration':calibration, 'test':test, 'holdout':holdout}
    assert len(set.union(*(set(v) for v in parts.values()))) == len(df)
    assert sum(map(len, parts.values())) == len(df)
    return parts
