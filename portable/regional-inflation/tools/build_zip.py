"""Build and verify the end-user ZIP with local-root sources and launcher."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
inputs=['regional-inflation.html','Start.cmd','Start.ps1','macro_rates.json','data/regional_indices.json','core/src/lib.rs','core/pkg/regional_inflation_core_bg.wasm','web/app.js','web/local-sources.js','web/cbr-rates.js','web/comparison.js','web/source-import.js']
manifest={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in inputs}
(ROOT/'verification/artifact_hashes.json').write_text(json.dumps(manifest,indent=2)+'\n')
files={name:name for name in ['regional-inflation.html','Start.cmd','Start.ps1','macro_rates.json','README.md','LOCAL_FILES.md']}
files['data/regional_indices.json']='regional_indices.json'
for name in ['REPORT_V2.md','artifact_hashes.json','browser_ui.json','browser_v2.json','comparison_v2.json','rate_source_audit_v2.json','converged_reference.json','kernel_contract.json','all_regions.json']:
    files['verification/'+name]='verification/'+name
with ZipFile(ROOT/'regional-inflation.zip','w',ZIP_DEFLATED,compresslevel=9) as archive:
    for source,target in files.items():archive.write(ROOT/source,'regional-inflation/'+target)
with ZipFile(ROOT/'regional-inflation.zip') as archive:
    assert archive.testzip() is None
    assert len(archive.namelist())==len(files)
    for source,target in files.items():assert archive.read('regional-inflation/'+target)==(ROOT/source).read_bytes()
print(json.dumps({'zip':str(ROOT/'regional-inflation.zip'),'files':len(files),'bytes':(ROOT/'regional-inflation.zip').stat().st_size,'verified':True}))
