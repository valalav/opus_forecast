"""Include source originals, OPR comparisons and scenario details in the package."""
from pathlib import Path
import zipfile,shutil,importlib.util,json,hashlib
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
SOURCE=ROOT/"data/external/med_forecast_2027_2029_20260924"
package=OUT/"forecast_package_20260928.zip"
extras=["OPR102026_Прогноз.xlsx","OPR092026_source.xlsx","opr092026_reference.csv","2027_conditioning.csv","item_price_scenarios.csv","finish_package.py"]
with zipfile.ZipFile(package) as z:parts={n:z.read(n) for n in z.namelist()}
for name in extras:parts[name]=(OUT/name).read_bytes()
for p in SOURCE.rglob("*"):
    if p.is_file():parts["sources/med_20260924/"+str(p.relative_to(SOURCE))]=p.read_bytes()
with zipfile.ZipFile(package,"w",compression=zipfile.ZIP_DEFLATED) as z:
    for name,content in parts.items():z.writestr(name,content)
spec=importlib.util.spec_from_file_location("package",ROOT/"archive/results/full_forecast_package_2026_2027/build_package.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
shutil.copy2(package,m.ZIP_PATH)
with zipfile.ZipFile(package) as z:
    assert z.testzip() is None
    assert all(n in z.namelist() for n in ["verification_report.md","forecast_explanation.pdf","OPR102026_Прогноз.xlsx","sources/med_20260924/tariff_calendar.csv"])
result={"status":"PASS","package_sha256":hashlib.sha256(package.read_bytes()).hexdigest(),"stable_package_matches":package.read_bytes()==m.ZIP_PATH.read_bytes(),"pdf_included":True,"source_originals_included":True}
(OUT/"package_verification.json").write_text(json.dumps(result,indent=2)+"\n")
print(result)
