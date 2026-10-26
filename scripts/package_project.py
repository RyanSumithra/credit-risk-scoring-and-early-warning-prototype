"""Package code and finished outputs; omit raw data, DB, environments and caches."""
from pathlib import Path
import hashlib
import json
import zipfile
ROOT=Path(__file__).resolve().parents[1]

def main():
    dist=ROOT/'dist';dist.mkdir(exist_ok=True)
    target=dist/'msme_credit_risk_project.zip'
    skip={'.git','.venv','.cache','.runtime','__pycache__','dist','.ipynb_checkpoints'}
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(ROOT.rglob('*')):
            relative=path.relative_to(ROOT)
            if any(p in skip for p in relative.parts) or not path.is_file():continue
            if relative.parts[0]=='data' and path.name!='.gitkeep':continue
            if path.suffix in {'.pyc','.sqlite','.db','.zip','.partial'}:continue
            if path.name in {'.env','kaggle.json','access_token'}:continue
            archive.write(path,Path('msme_credit_risk_project')/relative)
    with target.open('rb') as source:digest=hashlib.file_digest(source,'sha256').hexdigest()
    report={'archive':str(target),'bytes':target.stat().st_size,'sha256':digest,
            'excludes':'Official source data, SQLite database, processed intermediates, environment, caches, credentials'}
    (dist/'archive_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        print('Archive member count:',len(archive.namelist()))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
