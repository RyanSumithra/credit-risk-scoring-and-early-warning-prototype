"""Finish reproducible notebook execution, visual QA outputs, verification and ZIP."""
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]

def run(args):subprocess.run(args,cwd=ROOT,check=True)

def main():
    run([sys.executable,'scripts/make_notebooks.py'])
    run([sys.executable,'scripts/capture_dashboard.py'])
    poppler=shutil.which('pdftoppm')
    if poppler is None:raise RuntimeError('pdftoppm not on PATH. Install Poppler before PDF visual QA.')
    for name in ['business_memo','interview_prep']:
        run([poppler,'-scale-to','1400','-png',str(ROOT/'output/pdf'/f'{name}.pdf'),str(ROOT/'reports/figures'/name)])
    run([sys.executable,'scripts/verify_outputs.py'])
    run([sys.executable,'scripts/package_project.py'])
    print('Finalization complete. Inspect rendered PDF pages and dashboard screenshot before sharing.')

if __name__=='__main__':main()
