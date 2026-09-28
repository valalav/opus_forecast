"""Rebuild portable macro inputs from saved official CBR XML; never changes production CSV."""
from pathlib import Path
import json,hashlib,xml.etree.ElementTree as ET
from statistics import mean
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
old=json.loads((ROOT/"data/regional_indices.json").read_text())
xml={k:(ROOT/"data/cbr"/name).read_text() for k,name in [("ki","KeyRateXML.xml"),("ruonia","RuoniaXML.xml")]}
with sync_playwright() as pw:
    browser=pw.chromium.launch(executable_path="/usr/bin/google-chrome",headless=True,args=["--no-sandbox"])
    page=browser.new_page();page.goto((ROOT/"regional-inflation.html").as_uri())
    data=page.evaluate("""async ({xml,old})=>{
      const files=[new File([xml.ki],'KeyRateXML.xml'),new File([xml.ruonia],'RuoniaXML.xml')];
      const pkg=await CbrRates.importFiles(files,{from:'2016-01-01',to:'2026-08-31',ki_method:'last',ruonia_method:'mean'});
      return {pkg,dataset:CbrRates.merge(old,pkg)};
    }""",{"xml":xml,"old":old})
    browser.close()
expected={}
for key,tag,datefield,valuefield in [("ki","KR","DT","Rate"),("ruonia","ro","D0","ruo")]:
    groups={}
    for r in ET.fromstring(xml[key]).iter(tag):
        date=r.findtext(datefield)[:10];value=float(r.findtext(valuefield));groups.setdefault(date[:7]+"-01",[]).append((date,value))
    for month,points in groups.items():expected.setdefault(month,{})[key]=sorted(points)[-1][1] if key=="ki" else mean(v for _,v in points)
assert len(expected)==128
for row in data["pkg"]["macro_rows"]:
    assert all(abs(row[k]-expected[row["date"]][k])<1e-10 for k in ["ki","ruonia"])
prior={r['date']:r for r in old['macro_rows']}
differences=[{'date':r['date'],'old_ki':prior[r['date']]['ki'],'cbr_ki':r['ki'],'old_ruonia':prior[r['date']]['ruonia'],'cbr_ruonia':r['ruonia']} for r in data['pkg']['macro_rows'] if any(abs(r[k]-prior[r['date']][k])>1e-8 for k in ['ki','ruonia'])]
report={"status":"PASS","source":"Bank of Russia DailyInfo SOAP XML","monthly_rows":128,"independent_numeric_checks":256,"old_macro_source":old.get('macro_source'),"new_macro_source":{k:v for k,v in data['pkg']['macro_source'].items() if k!='raw_xml'},"differences":differences,"note":"Ki last published observation per month; RUONIA arithmetic mean of published daily observations. No production data modified."}
if differences or not (ROOT/'verification/rate_source_audit_v2.json').exists():
    (ROOT/'verification/rate_source_audit_v2.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
(ROOT/"macro_rates.json").write_text(json.dumps(data['pkg'],ensure_ascii=False,separators=(',',':'))+'\n')
(ROOT/"data/regional_indices.json").write_text(json.dumps(data['dataset'],ensure_ascii=False,separators=(',',':'))+'\n')
print(json.dumps({'monthly_rows':128,'numeric_checks':256,'changed_months':len(differences),'sample':next((r for r in differences if r['date']=='2023-09-01'),None)}))
