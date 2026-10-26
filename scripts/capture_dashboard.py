"""Render the offline dashboard in installed Chrome and verify live calculations.

Uses the Chrome headless CLI, avoiding a driver download or network service.
Run after deliverables. A temporary browser profile stays inside the project cache.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def main():
    chrome=Path(os.environ.get('PROGRAMFILES',r'C:\Program Files'))/'Google/Chrome/Application/chrome.exe'
    if not chrome.exists():raise FileNotFoundError('Google Chrome is required for dashboard visual QA.')
    target=ROOT/'reports/figures/dashboard_screenshot.png'
    profile=ROOT/'.cache/chrome-dashboard';profile.mkdir(parents=True,exist_ok=True)
    # Headless Chrome's CLI captures a fully self-contained file without a server.
    command=[str(chrome),'--headless=new','--disable-gpu','--hide-scrollbars','--no-first-run',
             '--no-default-browser-check','--disable-background-networking',
             '--user-data-dir='+str(profile),'--window-size=1500,1450',
             '--virtual-time-budget=10000','--screenshot='+str(target),
             (ROOT/'dashboard/index.html').as_uri()]
    result=subprocess.run(command,capture_output=True,text=True,timeout=60)
    (ROOT/'reports/dashboard_browser_log.txt').write_text(result.stdout+result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError('Chrome screenshot failed; see dashboard_browser_log.txt')
    if not target.is_file():raise RuntimeError('Chrome returned without producing dashboard screenshot.')
    # Re-open the file with a verification harness inserted before </body>.
    # Dumping the resulting DOM exposes actual JavaScript values after a slider change.
    source=(ROOT/'dashboard/index.html').read_text(encoding='utf-8')
    harness='''<script>
const initial={...window.dashboardState};
document.getElementById('cutoff').value=600;
document.getElementById('cutoff').dispatchEvent(new Event('input'));
const changed={...window.dashboardState};
document.getElementById('reset').click();
const reset={...window.dashboardState};
const pre=document.createElement('pre');pre.id='verification-output';
pre.textContent=JSON.stringify({initial,changed,reset,plots:document.querySelectorAll('.js-plotly-plot').length});
document.body.appendChild(pre);
</script>'''
    verification=ROOT/'.cache/dashboard_verify.html'
    verification.write_text(source.replace('</body>',harness+'</body>'),encoding='utf-8')
    result=subprocess.run([str(chrome),'--headless=new','--disable-gpu','--no-first-run',
                           '--disable-background-networking','--user-data-dir='+str(profile),
                           '--virtual-time-budget=1000','--dump-dom',verification.as_uri()],
                          capture_output=True,text=True,encoding='utf-8',timeout=60)
    marker='<pre id="verification-output">'
    if marker not in result.stdout:raise RuntimeError('Dashboard JavaScript verification did not execute.')
    from html import unescape
    state=json.loads(unescape(result.stdout.split(marker)[1].split('</pre>')[0]))
    metrics=json.loads((ROOT/'reports/metrics.json').read_text())
    policy=metrics['cutoff']['evaluation']['holdout']
    assert state['plots']==2,state
    assert abs(state['initial']['approval']-policy['approval_rate'])<1e-12
    assert abs(state['initial']['netCrore']-policy['estimated_annual_net_benefit_crore'])<1e-8
    assert state['initial']==state['reset']
    assert state['changed']['cutoff']==600
    (ROOT/'reports/dashboard_verification.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    print('Dashboard screenshot saved; charts, live slider, reset, and Python/JavaScript economics reconciliation passed.')
    print(json.dumps(state,indent=2))

if __name__=='__main__':main()
