"""Download the specified Kaggle competition, or import its official ZIP.

Uses only the standard library for local imports/checks. Does not read or print
credentials, create substitute data, or overwrite existing CSV files.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
REQUIRED = {
    'application_train.csv': {'SK_ID_CURR', 'TARGET'},
    'bureau.csv': {'SK_ID_CURR', 'SK_ID_BUREAU'},
    'bureau_balance.csv': {'SK_ID_BUREAU', 'MONTHS_BALANCE', 'STATUS'},
    'previous_application.csv': {'SK_ID_CURR', 'SK_ID_PREV'},
    'installments_payments.csv': {'SK_ID_CURR', 'SK_ID_PREV'},
    'POS_CASH_balance.csv': {'SK_ID_CURR', 'SK_ID_PREV'},
    'credit_card_balance.csv': {'SK_ID_CURR', 'SK_ID_PREV'},
}

def check():
    records = []
    for name, required_columns in REQUIRED.items():
        path = RAW / name
        record = {'file': name, 'exists': path.is_file(), 'header_valid': False}
        if path.is_file():
            record['bytes'] = path.stat().st_size
            with path.open(encoding='utf-8-sig', newline='') as source:
                columns = next(csv.reader(source), [])
            record['header_valid'] = required_columns.issubset(columns)
        records.append(record)
    ready = all(r['exists'] and r['header_valid'] for r in records)
    report = {
        'status': 'HEADERS_READY_FOR_FULL_AUDIT' if ready else 'BLOCKED_MISSING_OR_INVALID_DATA',
        'files': records,
        'note': 'Checks presence and required header columns only. No row counts, provenance assurance, or model metrics.',
    }
    (ROOT / 'reports').mkdir(exist_ok=True)
    (ROOT / 'reports' / 'data_access_check.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if ready else 2

def unpack(archive):
    with zipfile.ZipFile(archive) as source:
        for name in REQUIRED:
            members = [m for m in source.infolist() if Path(m.filename).name == name and not m.is_dir()]
            if len(members) != 1:
                raise RuntimeError(f'Expected one {name} in official ZIP; found {len(members)}. No substitution allowed.')
        for name in REQUIRED:
            target = RAW / name
            if target.exists():
                continue
            member = next(m for m in source.infolist() if Path(m.filename).name == name and not m.is_dir())
            temporary = target.with_suffix('.csv.partial')
            if temporary.exists():
                raise RuntimeError(f'Incomplete earlier extraction exists: {temporary}. Inspect it before retrying.')
            with source.open(member) as src, temporary.open('xb') as dst:
                shutil.copyfileobj(src, dst)
            temporary.rename(target)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--check-only', action='store_true')
    group.add_argument('--archive', type=Path, help='Official Home Credit competition ZIP downloaded manually')
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    if args.check_only:
        return check()
    archive = args.archive
    if archive is None:
        archive = RAW / 'home-credit-default-risk.zip'
        if not archive.exists():
            executable = shutil.which('kaggle')
            adjacent = Path(sys.executable).parent / ('kaggle.exe' if sys.platform == 'win32' else 'kaggle')
            if executable is None and adjacent.exists():
                executable = str(adjacent)
            if executable is None:
                print('BLOCKED: Kaggle CLI unavailable. Follow DATA_DOWNLOAD.md, or supply --archive.', file=sys.stderr)
                return 2
            result = subprocess.run([executable, 'competitions', 'download',
                                     'home-credit-default-risk', '-p', str(RAW)], check=False)
            if result.returncode:
                print('BLOCKED: Kaggle download failed. Check authentication and competition-rule acceptance. No analysis performed.', file=sys.stderr)
                return result.returncode
    unpack(archive)
    return check()

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, zipfile.BadZipFile, csv.Error) as exc:
        print(f'BLOCKED: {exc}', file=sys.stderr)
        raise SystemExit(2)
