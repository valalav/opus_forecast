"""Real-source pilot import and model run, distinct from synthetic acceptance fixtures."""
import json,hashlib
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
calendar=ROOT/'data/tariffs/kbr_nalchik_monthly_payment_cap_proxy.json'
with sync_playwright() as pw:
    browser=pw.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox'])
    page=browser.new_page(offline=True,accept_downloads=True,viewport={'width':1440,'height':1000})
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto((ROOT/'regional-inflation.html').as_uri());page.wait_for_function("!document.querySelector('#runButton').disabled")
    page.locator('#tariffCalendarFile').set_input_files(str(calendar));page.wait_for_function("document.querySelector('#tariffCalendarNotice').textContent.includes('импортирован')")
    page.locator('#tariffFeatures').select_option('current');page.locator('#advancedToggle').click();page.locator('#horizon').fill('4')
    def run():
        page.locator('#runButton').click();page.wait_for_function("!document.querySelector('#runButton').disabled",timeout=120000)
        assert page.locator('#results').is_visible(),page.locator('#errorMessage').inner_text()
    def report():
        with page.expect_download() as d:page.locator('#saveReport').click()
        return json.loads(Path(d.value.path()).read_text())
    run();forecast=report();assert len(forecast['response']['forecast']['steps'])==4
    assert forecast['response']['fit']['tariff_diagnostics']['series_id']=='monthly_change_of_nalchik_payment_cap_proxy_pct'
    page.locator('#operation').select_option('backtest');run();page.locator('#compareTariffs').click();page.wait_for_function("!document.querySelector('#runButton').disabled",timeout=120000)
    comparison=report()['comparison'];assert comparison
    pair=comparison['paired']['calendar']['1'];assert pair['n_pairs']==20 and pair['p_value'] is None
    assert comparison['paired']['expert']['1']['n_pairs']==0
    assert 'недоступно' in page.locator('#comparisonResults').inner_text()
    assert not errors,errors
    page.locator('#tariffCalendarTitle').locator('xpath=ancestor::section').screenshot(path=str(ROOT/'verification/screenshots/tariff_calendar_pilot.png'))
    browser.close()
out={'status':'PASS','checks':['real-source offline calendar import','4-month actual WASM forecast','fit identifies cap proxy','20 matching h1 pairs','no inference for incomplete coverage','no fabricated expert weights','unavailable counts visible','no JavaScript errors'],'html_sha256':hashlib.sha256((ROOT/'regional-inflation.html').read_bytes()).hexdigest(),'calendar_sha256':hashlib.sha256(calendar.read_bytes()).hexdigest(),'forecast':forecast['response'],'paired_h1':pair,'scope':'mechanics on a real sourced proxy, NOT an approved forecast or model promotion'}
(ROOT/'verification/tariff_pilot_browser.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'PASS','checks':len(out['checks'])}))
