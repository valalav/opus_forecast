"""Verify tariff scenario on the shipped offline HTML with actual WASM forecasts."""
import csv, hashlib, io, json, math
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT = Path(__file__).resolve().parents[1]
checks=[]
def check(name, ok):
    checks.append({"name": name, "pass": bool(ok)})
    assert ok, name
with sync_playwright() as pw:
    browser=pw.chromium.launch(executable_path="/usr/bin/google-chrome",headless=True,args=["--no-sandbox"])
    page=browser.new_page(offline=True,accept_downloads=True,viewport={"width":1440,"height":1000})
    errors=[]
    page.on("pageerror",lambda e:errors.append(str(e)))
    page.goto((ROOT/"regional-inflation.html").as_uri())
    page.wait_for_function("!document.querySelector('#runButton').disabled")
    def run(ok=True):
        page.locator('#runButton').click()
        page.wait_for_function("!document.querySelector('#runButton').disabled",timeout=120000)
        check('calculation visible' if ok else 'invalid scenario blocked',page.locator('#results').is_visible()==ok)
    def download(button):
        with page.expect_download() as dl:page.locator(button).click()
        return Path(dl.value.path()).read_text(encoding='utf-8-sig')
    run();base=json.loads(download('#saveReport'))
    page.locator('#tariffPanel summary').click()
    page.locator('#tariffEnabled').check()
    page.locator('#addTariff').click()
    row=page.locator('#tariffRows tr').first
    for key,value in {'date':'2026-10','rate':'10','baseline':'0'}.items():row.locator(f'[data-field="{key}"]').fill(value)
    run(False)
    check('blank weight visibly rejected','заполните' in page.locator('#errorMessage').inner_text())
    row.locator('[data-field="weight"]').fill('5')
    run();scenario=json.loads(download('#saveReport'))
    b=base['response']['forecast']['steps'];s=scenario['response']['forecast']['steps']
    check('model preserved exactly',scenario['response']['model_forecast']==base['response']['forecast'])
    check('October direct effect only',all(abs(y['all']-x['all']-(.5 if x['date'].startswith('2026-10') else 0))<1e-12 for x,y in zip(b,s)))
    check('table separates model and scenario','Тарифная поправка' in page.locator('#tableHead').inner_text() and 'Модель, % г/г' in page.locator('#tableHead').inner_text())
    exported=list(csv.DictReader(io.StringIO(download('#exportCsv')),delimiter=';'))
    history=[r for r in scenario['request']['rows'] if r['date']<=scenario['response']['forecast']['origin']][-11:]
    path=[r['y'] for r in history]+[100+r['all'] for r in s]
    expected=[(math.prod(v/100 for v in path[i:i+12])-1)*100 for i in range(len(s))]
    check('annual scenario independently compounded',all(abs(float(r['yoy'])-v)<1e-10 for r,v in zip(exported,expected)))
    check('CSV exposes base and contribution',all(k in exported[0] for k in ['model_all','tariff_delta','model_yoy','yoy']))
    page.locator('#valueMode').click()
    plotted=page.locator('#chart').evaluate('el=>JSON.parse(el.dataset.points)')
    check('annual chart uses scenario',all(abs(r['y']-v)<1e-10 for r,v in zip(plotted['forecast'],expected)))
    settings=download('#saveSettings')
    page.locator('#resetButton').click()
    page.locator('#fileInput').set_input_files({'name':'settings.json','mimeType':'application/json','buffer':settings.encode()})
    page.wait_for_function("document.querySelector('#tariffEnabled').checked")
    check('settings restore events and enabled',page.locator('#tariffEnabled').is_checked() and page.locator('#tariffRows [data-field="weight"]').input_value()=='5')
    page.locator('#operation').select_option('backtest');run()
    back=json.loads(download('#saveReport'))
    check('backtest excludes scenario','tariff_scenario' not in back['response'] and 'не применяется' in page.locator('#resultWarnings').inner_text())
    page.locator('#operation').select_option('forecast');page.locator('#representation').select_option('sa');run(False)
    check('SA scenario explicitly rejected','только RAW' in page.locator('#errorMessage').inner_text())
    page.locator('#representation').select_option('raw');run()
    (ROOT/'verification/screenshots').mkdir(exist_ok=True)
    page.screenshot(path=str(ROOT/'verification/screenshots/tariffs.png'),full_page=True)
    page.set_viewport_size({'width':390,'height':844})
    check('mobile viewport does not overflow',page.evaluate('document.documentElement.scrollWidth<=innerWidth'))
    check('no JavaScript errors',not errors)
    browser.close()
report={'status':'PASS','checks':checks,'html_sha256':hashlib.sha256((ROOT/'regional-inflation.html').read_bytes()).hexdigest(),'environment':'Chrome Linux offline; Rust/WASM; Windows not tested'}
(ROOT/'verification/tariff_browser.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'PASS','checks':len(checks)}))
