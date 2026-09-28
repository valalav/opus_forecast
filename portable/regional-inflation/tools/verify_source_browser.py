"""Verify the browser XLSX importer against the independently prepared package."""
from pathlib import Path
import json
import tempfile
import time
import argparse
from playwright.sync_api import sync_playwright

parser=argparse.ArgumentParser()
parser.add_argument('workbook', type=Path)
args=parser.parse_args()
P=Path(__file__).resolve().parents[1]
reference=json.loads((P/'data/regional_indices.json').read_text())
with tempfile.TemporaryDirectory() as tmp:
    pagefile=Path(tmp)/'parser.html'
    pagefile.write_text('<input type="file" id="file"><script src="'+(P/'node_modules/fflate/umd/index.js').as_uri()+'"></script><script src="'+(P/'web/source-import.js').as_uri()+'"></script>')
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--no-sandbox'])
        page=browser.new_page();page.goto(pagefile.as_uri())
        page.locator('#file').set_input_files(args.workbook.resolve())
        started=time.monotonic()
        imported=page.evaluate('''async previous => importWorkbook(await document.querySelector('#file').files[0].arrayBuffer(), {previousDataset:previous})''',reference)
        elapsed=time.monotonic()-started
        expected={str(r['code']):r for r in reference['regions']}
        assert len(imported['regions'])==len(expected)
        checked=0
        for region in imported['regions']:
            base=expected[str(region['code'])]
            for key in ['rows','sa_rows']:
                assert len(region[key])==len(base[key]),(region['code'],key)
                for a,b in zip(region[key],base[key]):
                    assert a==b,(region['code'],key,a,b)
                    checked+=len(a)-1
        browser.close()
result={'status':'PASS','regions':len(expected),'numeric_values_compared':checked,'seconds':elapsed,
        'file_protocol':True,'sha256':imported['source']['sha256']}
(P/'verification').mkdir(exist_ok=True)
(P/'verification/browser_import.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
