"""Create the initial project scaffold; never overwrite existing work."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = [
    'data/raw', 'data/interim', 'data/processed', 'sql', 'notebooks', 'src',
    'early_warning', 'dashboard', 'reports/figures', 'models', 'config',
]
MODULES = ['data_loader', 'features', 'woe_binning', 'models', 'validation', 'cutoff']
SQL_STAGES = ['01_create_tables', '02_clean_and_join', '03_feature_aggregation', '04_portfolio_kpis']
NOTEBOOKS = ['01_data_audit', '02_eda_business_insights', '03_woe_iv_binning',
             '04_scorecard_model', '05_gradient_boosting_shap', '06_validation', '07_cutoff_strategy']

def write_new(relative, content):
    path = ROOT / relative
    if not path.exists():
        path.write_text(content, encoding='utf-8')

def main():
    for relative in DIRECTORIES:
        (ROOT / relative).mkdir(parents=True, exist_ok=True)
        write_new(relative + '/.gitkeep', '')
    for name in MODULES:
        write_new(f'src/{name}.py',
                  f'"""{name}: planned module, not implemented. Awaiting real Home Credit data."""\n')
    write_new('src/__init__.py', '')
    for name in SQL_STAGES:
        write_new(f'sql/{name}.sql', '-- PLANNED: not implemented or executed; awaiting real data.\n')
    for name in NOTEBOOKS:
        notebook = {
            'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source': [
                f'# {name}\n',
                'Status: PLANNED. Awaiting real Home Credit competition data.\n',
                'No analysis has run and no results are available.\n']}],
            'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
            'nbformat': 4, 'nbformat_minor': 5,
        }
        write_new(f'notebooks/{name}.ipynb', json.dumps(notebook, indent=2) + '\n')
    print('SETUP COMPLETE: requested directories, SQL placeholders, notebook shells, and module shells exist.')
    print('Analysis status: NOT STARTED. These shells do not constitute a working analytics pipeline.')

if __name__ == '__main__':
    main()
