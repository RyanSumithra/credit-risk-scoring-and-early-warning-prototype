"""Chunked, restartable ingestion of the official CSVs into SQLite."""
from pathlib import Path
import hashlib
import json
import sqlite3
import time
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILES = ['application_train', 'bureau', 'bureau_balance', 'previous_application',
         'installments_payments', 'POS_CASH_balance', 'credit_card_balance']
DB = ROOT / 'data/interim/home_credit.sqlite'

def ingest():
    DB.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB)
    connection.execute('PRAGMA journal_mode=WAL')
    connection.execute('PRAGMA synchronous=NORMAL')
    connection.execute('PRAGMA cache_size=-200000')
    connection.execute('CREATE TABLE IF NOT EXISTS ingestion_log (name TEXT PRIMARY KEY, rows INTEGER, sha256 TEXT)')
    records = []
    ddl = ['-- Generated from actual CSV dtypes; executed by src/data_loader.py.']
    for name in FILES:
        path = ROOT / 'data/raw' / (name + '.csv')
        started = time.time()
        with path.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        prior = connection.execute('SELECT rows, sha256 FROM ingestion_log WHERE name=?', (name,)).fetchone()
        if prior and prior[1] == digest:
            count = prior[0]
        else:
            connection.execute(f'DROP TABLE IF EXISTS "{name}"')
            connection.commit()
            count = 0
            for chunk in pd.read_csv(path, chunksize=100000):
                chunk.to_sql(name, connection, if_exists='append', index=False, chunksize=2000)
                count += len(chunk)
            connection.execute('INSERT OR REPLACE INTO ingestion_log VALUES (?,?,?)', (name, count, digest))
            connection.commit()
        ddl.append(connection.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()[0] + ';')
        record = {'table': name, 'rows': count, 'sha256': digest, 'seconds': round(time.time()-started, 2)}
        records.append(record)
        print(json.dumps(record), flush=True)
    (ROOT / 'sql/01_create_tables.sql').write_text('\n\n'.join(ddl), encoding='utf-8')
    (ROOT / 'reports/ingestion.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    connection.close()
    return records
