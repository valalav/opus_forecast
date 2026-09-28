"""Independent Newey-West comparison check and matched historical model experiments."""
import json,subprocess,math,hashlib
from pathlib import Path
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac
from scipy.stats import norm
ROOT=Path(__file__).resolve().parents[1]
JS=ROOT/"web/comparison.js"
checks=[]
def assert_check(name,condition):
    checks.append({"name":name,"pass":bool(condition)})
    assert condition,name

def paired(a,b):
    runner="import fs from 'node:fs';import vm from 'node:vm';vm.runInThisContext(fs.readFileSync(process.argv[1],'utf8'));const d=JSON.parse(fs.readFileSync(0,'utf8'));console.log(JSON.stringify(RegionalComparison.paired(d.a,d.b)));"
    result=subprocess.run(["node","--input-type=module","-e",runner,str(JS)],input=json.dumps({"a":a,"b":b}),text=True,capture_output=True,check=True)
    return json.loads(result.stdout)

a=[];b=[]
for h in [1,2,12]:
    for i in range(60):
        target=f"{2020+i//12:04d}-{i%12+1:02d}-01"
        row={"horizon":h,"target_date":target,"cutoff":target,"actual":0.0,"status":"available"}
        a.append(dict(row,forecast=1+.2*math.sin(i*.47)+.1*math.cos(i*.13)))
        b.append(dict(row,forecast=1+.1*math.sin(i*.31)))
pa={"backtest":{"rows":a}};pb={"backtest":{"rows":b}}
out=paired(pa,pb)
for h in [1,2,12]:
    rows=[x for x in a if x["horizon"]==h];other=[x for x in b if x["horizon"]==h]
    d=np.array([abs(x["forecast"])-abs(y["forecast"]) for x,y in zip(rows,other)])
    result=sm.OLS(d,np.ones((len(d),1))).fit()
    se=float(np.sqrt(cov_hac(result,nlags=out[str(h)]["bandwidth"],use_correction=False)[0,0]))
    assert_check(f"HAC independent h{h}",abs(se-out[str(h)]["standard_error"])<1e-12)
    assert_check(f"normal p independent h{h}",abs(2*norm.sf(abs(float(d.mean())/se))-out[str(h)]["p_value"])<2e-7)
identical=paired(pa,pa)
assert_check("identical forecasts do not invent significance",all(x["status"]=="degenerate_variance" and x["p_value"] is None for x in identical.values()))
missing=json.loads(json.dumps(pb));missing["backtest"]["rows"][0].update(status="unavailable",forecast=None)
assert_check("missing pairs suppress p",paired(pa,missing)["1"]["status"]=="incomplete_pairs")
package=json.loads((ROOT/"data/regional_indices.json").read_text())
experiments=[]
for code in [7,16,77]:
    region=next(r for r in package["regions"] if r["code"]==code)
    for frequency,key in [("raw","rows"),("sa","sa_rows")]:
        rows=region[key];base={"schema_version":1,"model":"ridge","frequency":frequency,"rows":rows,"config":{"horizon":12,"outlier_mode":"none","seasonality_mode":"legacy"},"backtests":{"targets":[r["date"] for r in rows[-36:]],"horizons":[1,2,12]}}
        outputs={}
        for mode in ["none","mad_winsor"]:
            request=json.loads(json.dumps(base));request["config"]["outlier_mode"]=mode
            run=subprocess.run([str(ROOT/"core/target/release/regional-inflation-core")],input=json.dumps(request),text=True,capture_output=True,check=True)
            outputs[mode]=json.loads(run.stdout);assert_check(f"region {code} {frequency} {mode}",outputs[mode]["ok"])
        summary=paired(outputs["none"],outputs["mad_winsor"])
        assert_check(f"region {code} {frequency} matched pairs",all(r["n_pairs"]==36 for r in summary.values()))
        experiments.append({"region":code,"frequency":frequency,"paired":summary,"outliers":outputs["mad_winsor"]["fit"]["outlier_diagnostics"]})
report={"passed":True,"checks":checks,"experiments":experiments,"comparison_sha256":hashlib.sha256(JS.read_bytes()).hexdigest(),"binary_sha256":hashlib.sha256((ROOT/"core/target/release/regional-inflation-core").read_bytes()).hexdigest(),"note":"Exploratory same-window comparison, not model promotion. No automatic hyperparameter selection."}
(ROOT/"verification/comparison_v2.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({"checks":len(checks),"passed":True,"experiments":len(experiments)}))
