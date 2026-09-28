"""Exercise local-root launch, model controls, CBR online/offline rate cycle."""
import argparse,json,subprocess,tempfile,shutil,socket,time,re,hashlib
from pathlib import Path
import requests
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--pwsh',default='pwsh');args=parser.parse_args()
checks=[]
def check(name,ok):
    checks.append({'name':name,'pass':bool(ok)});assert ok,name
with tempfile.TemporaryDirectory(prefix='regional test ') as tmp:
    folder=Path(tmp)
    for name in ['Start.ps1','regional-inflation.html','macro_rates.json']:shutil.copy2(ROOT/name,folder/name)
    shutil.copy2(ROOT/'data/regional_indices.json',folder/'regional_indices.json')
    alternate=folder/'другая папка';alternate.mkdir()
    source=json.loads((ROOT/'data/regional_indices.json').read_text());source['source']['test_marker']='alternate-folder'
    (alternate/'regional_indices.json').write_text(json.dumps(source,ensure_ascii=False))
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    process=subprocess.Popen([args.pwsh,'-NoProfile','-File',str(folder/'Start.ps1'),'-Port',str(port),'-NoBrowser'],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
    try:
        for _ in range(100):
            try:
                response=requests.get(f'http://127.0.0.1:{port}/regional-inflation.html',timeout=1);break
            except requests.RequestException:time.sleep(.1)
        else:raise RuntimeError('PowerShell server did not start')
        config=json.loads(re.search(r'window.LOCAL_SOURCES_CONFIG=(\{.*?\});',response.text).group(1));base=config['baseUrl'];headers={'X-Local-Token':config['token']}
        check('default root beside HTML',config['defaultPath']==str(folder))
        check('missing token rejected',requests.get(base+'/api/local-sources',timeout=5).status_code==403)
        check('foreign origin rejected',requests.get(base+'/api/local-sources',headers=dict(headers,Origin='https://foreign.example'),timeout=5).status_code==403)
        check('arbitrary file rejected',requests.get(base+'/api/local-file',params={'path':str(folder/'Start.ps1')},headers=headers,timeout=5).status_code==404)
        check('relative root file resolves',requests.get(base+'/api/local-file',params={'path':'./regional_indices.json'},headers=headers,timeout=5).status_code==200)
        with sync_playwright() as pw:
            browser=pw.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox'])
            context=browser.new_context(accept_downloads=True);page=context.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(base+'/regional-inflation.html')
            page.wait_for_function("document.querySelector('#dataNotice').textContent.includes('Локальные данные загружены')",timeout=30000)
            check('root auto-load',page.locator('#regionSelect option').count()==101)
            check('CBR default complete month',page.locator('#cbrTo').input_value().endswith('-31') or page.locator('#cbrTo').input_value().endswith('-30') or page.locator('#cbrTo').input_value().endswith('-28') or page.locator('#cbrTo').input_value().endswith('-29'))
            def run():
                page.locator('#runButton').click();page.wait_for_function("!document.querySelector('#runButton').disabled",timeout=90000)
                check('calculation visible',page.locator('#results').is_visible())
            def download(button):
                with page.expect_download() as dl:page.locator(button).click()
                return Path(dl.value.path()).read_bytes()
            def report():return json.loads(download('#saveReport'))
            run();original=report()
            check('annual column beside MoM',page.locator('#tableHead th').count()==3)
            csv=download('#exportCsv').decode('utf-8-sig');check('annual included in CSV','yoy' in csv.splitlines()[0])
            page.locator('#advancedToggle').click();page.locator('#advancedConfig details summary').click();page.locator('#seasonalityMode').select_option('off');run();off=report()
            check('seasonality changes forecasts',any(abs(a['all']-b['all'])>1e-5 for a,b in zip(original['response']['forecast']['steps'],off['response']['forecast']['steps'])))
            check('seasonal features removed','month_sin' not in off['response']['fit']['features'])
            page.locator('#seasonalityMode').select_option('legacy');page.locator('#outlierMode').select_option('mad_winsor');page.locator('#outlierThreshold').fill('2');run();trim=report()
            check('outliers really processed',trim['response']['fit']['outlier_diagnostics']['n_flagged']>0)
            page.locator('#operation').select_option('backtest');run()
            page.locator('#compareOutliers').click();page.wait_for_function("!document.querySelector('#runButton').disabled",timeout=90000)
            compared=report()['comparison'];check('comparison exported',compared is not None)
            check('same target pairs',all(s['n_pairs']==36 for s in compared['paired_by_horizon'].values()))
            check('comparison p or stated reason',all(s['status']!='approximate' or s['p_value_holm'] is not None for s in compared['paired_by_horizon'].values()))
            page.locator('#sourcePath').fill(str(alternate));page.locator('#loadLocalSource').click()
            page.wait_for_function("!document.querySelector('#loadLocalSource').disabled")
            selected=json.loads(download('#savePackage'));check('editable source path used',selected['source']['test_marker']=='alternate-folder')
            page.locator('#operation').select_option('forecast')
            page.locator('#outlierMode').select_option('none');page.locator('#cbrFrom').fill('2016-01-01');page.locator('#cbrTo').fill('2026-08-31')
            page.locator('#downloadCbr').click();page.wait_for_function("!document.querySelector('#downloadCbr').disabled",timeout=90000)
            check('CBR actual network download','notice-success' in page.locator('#macroNotice').get_attribute('class'))
            saved=download('#saveRates');rates=json.loads(saved);check('rate file has128 months',len(rates['macro_rows'])==128)
            check('official September2023 rate',next(r for r in rates['macro_rows'] if r['date']=='2023-09-01')['ki']==13)
            run();online=report()['response']['forecast']['steps']
            offline=browser.new_context(offline=True,accept_downloads=True);op=offline.new_page();op.goto((ROOT/'regional-inflation.html').as_uri());op.wait_for_function("!document.querySelector('#runButton').disabled")
            op.locator('#rateFiles').set_input_files({'name':'macro_rates.json','mimeType':'application/json','buffer':saved})
            op.wait_for_function("!document.querySelector('#runButton').disabled")
            check('offline import success','notice-success' in op.locator('#macroNotice').get_attribute('class'))
            op.locator('#runButton').click();op.wait_for_function("!document.querySelector('#runButton').disabled")
            with op.expect_download() as dl:op.locator('#saveReport').click()
            restored=json.loads(Path(dl.value.path()).read_text());check('offline forecast equals online',restored['response']['forecast']['steps']==online)
            with op.expect_download() as dl:op.locator('#saveRates').click()
            reexport=json.loads(Path(dl.value.path()).read_text());check('macro roundtrip no loss',reexport['macro_rows']==rates['macro_rows'] and reexport['aggregation']==rates['aggregation'])
            partial=json.loads(saved);partial['macro_rows']=partial['macro_rows'][-12:];partial['daily']={'ki':[],'ruonia':[]}
            op.locator('#rateFiles').set_input_files({'name':'partial_rates.json','mimeType':'application/json','buffer':json.dumps(partial).encode()})
            op.wait_for_function("!document.querySelector('#runButton').disabled")
            op.locator('#runButton').click();op.wait_for_function("!document.querySelector('#runButton').disabled")
            check('missing early macro history rejected',op.locator('#errorMessage').is_visible() and 'all training months' in op.locator('#errorMessage').inner_text())
            op.locator('#advancedToggle').click();op.locator('#useMacro').uncheck()
            op.locator('#runButton').click();op.wait_for_function("!document.querySelector('#runButton').disabled")
            check('explicit no-macro mode works',op.locator('#results').is_visible())
            page.set_viewport_size({'width':390,'height':844});check('v2 mobile no overflow',page.evaluate('document.documentElement.scrollWidth<=innerWidth'))
            page.set_viewport_size({'width':1440,'height':1100});page.screenshot(path=str(ROOT/'verification/screenshots/v2.png'),full_page=True)
            check('no JavaScript errors',not errors);browser.close()
    finally:
        process.terminate();_,err=process.communicate(timeout=5)
        if err:print(err[-500:])
report={'passed':True,'checks':checks,'platform':'Linux Chrome and actual PowerShell 7 loopback TcpListener; Windows5.1 untested','html_sha256':hashlib.sha256((ROOT/'regional-inflation.html').read_bytes()).hexdigest()}
(ROOT/'verification/browser_v2.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({'passed':True,'checks':len(checks)}))
