"""End-to-end tariff-calendar UI with synthetic labelled assumptions, offline."""
import json, hashlib, subprocess
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
checks=[]
def check(name,ok):
    checks.append({'name':name,'pass':bool(ok)})
    assert ok,name
calendar={'schema_version':1,'region_code':'7','series_id':'SYNTHETIC_UI_TEST_NOT_REAL_TARIFFS','records':[]}
for y in range(2015,2029):
    for m in range(1,13):calendar['records'].append({'date':f'{y}-{m:02d}-01','rate':8 if m==7 else 0,'known_at':'2014-12-01','source':'synthetic:ui-test','kind':'assumption','weight':5,'baseline':6 if m==7 else 0})
with sync_playwright() as pw:
    browser=pw.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox'])
    page=browser.new_page(offline=True,accept_downloads=True,viewport={'width':1440,'height':1000});errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto((ROOT/'regional-inflation.html').as_uri());page.wait_for_function("!document.querySelector('#runButton').disabled")
    def run():
        page.locator('#runButton').click();page.wait_for_function("!document.querySelector('#runButton').disabled",timeout=120000)
        check('calculation visible',page.locator('#results').is_visible())
    def download(button):
        with page.expect_download() as d:page.locator(button).click()
        return Path(d.value.path()).read_text(encoding='utf-8-sig')
    def import_calendar(value,name='tariffs.json'):
        data=json.dumps(value).encode() if isinstance(value,dict) else value.encode()
        page.locator('#tariffCalendarFile').set_input_files({'name':name,'mimeType':'application/json' if name.endswith('json') else 'text/csv','buffer':data})
        page.wait_for_function("document.querySelector('#tariffCalendarNotice').textContent.includes('импортирован')")
    import_calendar(calendar)
    check('pagination retains entire calendar',page.locator('#tariffCalendarRows tr').count()==24 and page.evaluate('getTariffCalendar().records.length')==168)
    page.locator('#tariffCalendarOlder').click()
    row=page.locator('#tariffCalendarRows tr').first;date=row.locator('[data-field="date"]').input_value()+'-01'
    row.locator('[data-field="rate"]').fill('2.25')
    page.locator('#tariffCalendarNewer').click()
    saved=json.loads(download('#exportTariffCalendarJson'))
    check('editing survives pagination',next(r for r in saved['records'] if r['date']==date)['rate']==2.25)
    csv=download('#exportTariffCalendarCsv');import_calendar(csv,'tariffs.csv')
    check('CSV preserves full calendar',json.loads(download('#exportTariffCalendarJson'))==saved)
    page.locator('#tariffFeatures').select_option('current');run();report=json.loads(download('#saveReport'))
    check('calendar attached to request',report['request']['region_code']=='7' and len(report['request']['tariff_calendar']['records'])==168)
    check('fitted tariff coefficient exposed','tariff_rate_lag0' in report['response']['fit']['features'])
    native=json.loads(subprocess.run([str(ROOT/'core/target/release/regional-inflation-core')],input=json.dumps(report['request']),capture_output=True,text=True,check=True).stdout)
    check('WASM native calendar parity',max(abs(a['all']-b['all']) for a,b in zip(native['forecast']['steps'],report['response']['forecast']['steps']))<1e-8)
    page.locator('#tariffAsOf').fill('2026-09-28');run();asof_report=json.loads(download('#saveReport'))
    check('explicit information date reaches core',asof_report['request']['config']['tariff_as_of']=='2026-09-28')
    settings=download('#saveSettings');page.locator('#resetButton').click()
    page.locator('#fileInput').set_input_files({'name':'settings.json','mimeType':'application/json','buffer':settings.encode()})
    page.wait_for_function("document.querySelector('#tariffFeatures').value==='current'")
    check('settings restore full calendar',page.evaluate('getTariffCalendar().records.length')==168 and page.locator('#tariffAsOf').input_value()=='2026-09-28')
    page.locator('#tariffFeatures').select_option('lags3');page.locator('#operation').select_option('backtest');run()
    page.locator('#compareTariffs').click();page.wait_for_function("!document.querySelector('#runButton').disabled",timeout=120000)
    report=json.loads(download('#saveReport'));comp=report['comparison'];assert comp is not None,page.locator('#errorMessage').inner_text()
    check('three variants available',set(comp['variants'])=={'baseline','calendar','expert'})
    check('same complete horizon pairs',all(comp['paired'][kind][str(h)]['n_pairs']==36 for kind in ['calendar','expert'] for h in [1,2,12]))
    check('group diagnostics rendered','Ошибки в тарифные месяцы' in page.locator('#comparisonResults').inner_text())
    check('p values are not coefficient significance','HAC' in page.locator('#comparisonResults').inner_text())
    page.locator('#regionSelect').select_option('77');page.locator('#runButton').click()
    check('wrong region blocked',not page.locator('#results').is_visible() and 'регион' in page.locator('#errorMessage').inner_text().lower())
    page.locator('#regionSelect').select_option('7');page.locator('#representation').select_option('sa');page.locator('#runButton').click()
    page.wait_for_function("!document.querySelector('#runButton').disabled")
    check('SA rejected',not page.locator('#results').is_visible() and page.locator('#errorMessage').is_visible())
    page.locator('#representation').select_option('raw');run();page.locator('#compareTariffs').click();page.wait_for_function("!document.querySelector('#runButton').disabled",timeout=120000)
    page.screenshot(path=str(ROOT/'verification/screenshots/tariff_calendar.png'),full_page=True)
    page.set_viewport_size({'width':390,'height':844})
    check('mobile no overflow',page.evaluate('document.documentElement.scrollWidth<=innerWidth'))
    old_settings={'schema_version':1,'config':{'horizon':12},'tariff_calendar':None}
    page.locator('#fileInput').set_input_files({'name':'empty-settings.json','mimeType':'application/json','buffer':json.dumps(old_settings).encode()})
    page.wait_for_function("document.querySelector('#tariffFeatures').value==='off'")
    check('empty settings clear previous calendar',page.locator('#tariffCalendarRows tr').count()==0)
    check('no JS errors',not errors)
    browser.close()
(ROOT/'verification/tariff_calendar_browser.json').write_text(json.dumps({'status':'PASS','checks':checks,'html_sha256':hashlib.sha256((ROOT/'regional-inflation.html').read_bytes()).hexdigest(),'fixture':'SYNTHETIC, tests mechanics only','platform':'Chrome Linux offline'},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'PASS','checks':len(checks)}))
