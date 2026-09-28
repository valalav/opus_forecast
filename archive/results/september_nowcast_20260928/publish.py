"""Publish the dated diagnostic without changing model or policy trajectories."""
from pathlib import Path
import json, importlib.util, sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("precompute",ROOT/"scripts/precompute_forecasts.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
data=json.loads((ROOT/"data/precomputed_forecasts.json").read_text())
summary=json.loads((OUT/"summary.json").read_text())
import hashlib
for path,expected in summary["source_hashes"].items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
assert data["forecasts"]["Nowcast"][0]==summary["standard_bridge_nowcast"]
data["diagnostics"]["weighted_micro_nowcast"]=summary
m.save_results(data,ROOT/"data/precomputed_forecasts.json")
