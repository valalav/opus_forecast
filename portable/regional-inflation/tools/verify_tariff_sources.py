"""Independently verify saved source hashes and monthly cap-ratio arithmetic."""
import csv, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'data/tariffs';manifest=json.loads((p/'source_manifest.json').read_text())
checks=[]
for source in manifest['sources']:
    file=p/source['local_file'];assert hashlib.sha256(file.read_bytes()).hexdigest()==source['sha256'];checks.append('hash '+file.name)
    if 'publication_proof_file' in source:
        assert hashlib.sha256((p/source['publication_proof_file']).read_bytes()).hexdigest()==source['publication_proof_sha256']
calendar=p/'kbr_nalchik_monthly_payment_cap_proxy.json';cal=json.loads(calendar.read_text());by={r['date']:r for r in cal['records']}
raw=list(csv.DictReader((p/'kbr_nalchik_payment_caps.csv').open()));previous={}
for row in raw:
    start,end=row['period_start'],row['period_end'];year=int(start[:4]);begin=int(start[5:7]);finish=int(end[5:7]);cap=float(row['cap_pct']);prior=previous.get(year,0)
    for month in range(begin,finish+1):
        date=f'{year}-{month:02d}-01';expected=(cap-prior)/(100+prior)*100;record=by[date]
        assert abs(expected-record['rate'])<5.01e-7
        assert record['known_at']==row['known_at'] and record['kind']=='plan'
        prior=cap
    previous[year]=cap
assert len(by)==60
assert not any(d.startswith(('2021','2022','2023','2024')) for d in by)
report={'status':'PASS','n_records':len(by),'checked_source_hashes':checks,'calendar_sha256':hashlib.sha256(calendar.read_bytes()).hexdigest(),'checks':['independent algebraic cap-ratio calculation','no invented gap years','publication dates preserved','all records plan not observed CPI','primary source hashes'],'note':'Nalchik payment cap proxy only; authoritative tables checked separately via pdftotext.'}
(ROOT/'verification/tariff_source_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'PASS','records':len(by),'sources':len(checks)}))
