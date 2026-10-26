"""SQL feature aggregation with one row per labeled application."""
import json
import sqlite3
import pandas as pd
from src.data_loader import DB, ROOT

def build():
    connection = sqlite3.connect(DB)
    connection.execute('PRAGMA cache_size=-200000')
    connection.execute('PRAGMA temp_store=FILE')
    for filename in ['02_clean_and_join.sql', '03_feature_aggregation.sql']:
        print('Executing SQL: ' + filename, flush=True)
        connection.executescript((ROOT / 'sql' / filename).read_text(encoding='utf-8'))
    columns = [r[1] for r in connection.execute('PRAGMA table_info(application_train)')]
    select = [('NULLIF(a.DAYS_EMPLOYED,365243) AS DAYS_EMPLOYED' if c=='DAYS_EMPLOYED' else f'a."{c}"') for c in columns]
    select += [
        'a.AMT_CREDIT/NULLIF(a.AMT_INCOME_TOTAL,0) AS credit_income_ratio',
        'a.AMT_ANNUITY/NULLIF(a.AMT_INCOME_TOTAL,0) AS annuity_income_ratio',
        'a.AMT_CREDIT/NULLIF(a.AMT_ANNUITY,0) AS credit_annuity_ratio',
        'a.AMT_CREDIT/NULLIF(a.AMT_GOODS_PRICE,0) AS credit_goods_ratio',
        'a.AMT_INCOME_TOTAL/NULLIF(a.CNT_FAM_MEMBERS,0) AS income_per_family_member',
        '-a.DAYS_BIRTH/365.25 AS age_years',
    ]
    joins = []
    for alias, table in [('b','bureau_features'),('p','previous_features'),('i','installment_features'),('o','pos_features'),('c','card_features')]:
        select += [f'{alias}."{r[1]}"' for r in connection.execute(f'PRAGMA table_info({table})') if r[1]!='SK_ID_CURR']
        joins.append(f'LEFT JOIN {table} {alias} ON a.SK_ID_CURR={alias}.SK_ID_CURR')
    query = 'DROP TABLE IF EXISTS modeling;\nCREATE TABLE modeling AS SELECT\n' + ',\n'.join(select) + '\nFROM application_train a\n' + '\n'.join(joins) + ';\nCREATE UNIQUE INDEX ix_modeling_id ON modeling(SK_ID_CURR);'
    (ROOT / 'sql/05_modeling_table.sql').write_text(query, encoding='utf-8')
    connection.executescript(query)
    df = pd.read_sql_query('SELECT * FROM modeling ORDER BY SK_ID_CURR', connection)
    assert df.SK_ID_CURR.is_unique
    assert len(df)==connection.execute('SELECT COUNT(*) FROM application_train').fetchone()[0]
    assert set(df.TARGET.unique()) == {0,1}
    df.to_parquet(ROOT / 'data/processed/modeling.parquet', index=False)
    kpis = pd.read_sql_query((ROOT/'sql/04_portfolio_kpis.sql').read_text().split(';')[1], connection)
    kpis.to_csv(ROOT / 'reports/sql_income_segment_kpis.csv', index=False)
    report = {'rows':len(df), 'candidate_features':len(df.columns)-2, 'total_columns':len(df.columns),
              'bad_count':int(df.TARGET.sum()), 'bad_rate':float(df.TARGET.mean()),
              'unique_applicants':int(df.SK_ID_CURR.nunique())}
    (ROOT/'reports/sql_layer.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    connection.close()
    return report
