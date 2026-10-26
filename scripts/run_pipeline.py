"""Single entry point; each stage writes execution evidence."""
import argparse
from pathlib import Path
import sys
import os
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['sql','audit','model','simulation','deliverables','all'], default='all')
    args = parser.parse_args()
    if args.stage in ['sql','all']:
        from src.data_loader import ingest
        from src.features import build
        ingest()
        build()
    if args.stage in ['audit','all']:
        from src.audit import run
        run()
    if args.stage in ['model','all']:
        from src.models import run
        run()
    if args.stage in ['simulation','all']:
        from early_warning.simulate import run
        run()
    if args.stage in ['deliverables','all']:
        from src.deliverables import run
        run()

if __name__=='__main__':
    main()
