"""Exercise shipped single-file app offline, including numerical exports."""
import json, hashlib, subprocess
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
checks=[]
def check(name, condition):
    checks.append({"name":name,"pass":bool(condition)})
    assert condition,name
with sync_playwright() as pw:
    browser=pw.chromium.launch(executable_path="/usr/bin/google-chrome",headless=True,args=["--no-sandbox"])
    ctx=browser.new_context(offline=True,accept_downloads=True)
    page=ctx.new_page(); errors=[]
    page.on("pageerror",lambda error: errors.append(str(error)))
    page.goto((ROOT/"regional-inflation.html").as_uri())
    page.wait_for_function("!document.querySelector('#runButton').disabled")
    check("101 regions offline",page.locator("#regionSelect option").count()==101)
    def run():
        page.locator("#runButton").click()
        page.wait_for_function("!document.querySelector('#runButton').disabled",timeout=120000)
        check("calculation visible",page.locator("#results").is_visible())
    def report():
        with page.expect_download() as dl: page.locator("#saveReport").click()
        return json.loads(Path(dl.value.path()).read_text())
    run(); out=report()
    check("12 forecast table rows",page.locator("#tableBody tr").count()==12)
    check("forecast cells populated","—" not in page.locator("#tableBody").inner_text())
    native=json.loads(subprocess.run([str(ROOT/"core/target/release/regional-inflation-core")],input=json.dumps(out["request"]),text=True,capture_output=True,check=True).stdout)
    check("WASM equals native",max(abs(a["all"]-b["all"]) for a,b in zip(native["forecast"]["steps"],out["response"]["forecast"]["steps"]))<1e-8)
    with page.expect_download() as dl: page.locator("#exportCsv").click()
    csv=Path(dl.value.path()).read_text(encoding="utf-8-sig")
    check("CSV dates and values",len(csv.strip().splitlines())==13 and "2026-09" in csv and "undefined" not in csv)
    page.locator("#valueMode").click()
    plotted=page.locator("#chart").evaluate("el=>JSON.parse(el.dataset.points)")
    history=out["request"]["rows"]
    projected=history[-11:]+[{"date":r["date"],"y":100+r["all"]} for r in out["response"]["forecast"]["steps"]]
    import math
    expected=[(math.prod(float(r["y"])/100 for r in projected[i-11:i+1])-1)*100 for i in range(11,len(projected))]
    check("forecast YoY compounds monthly indices",max(abs(a["y"]-b) for a,b in zip(plotted["forecast"],expected))<1e-8)
    page.locator("#valueMode").click()
    page.locator("#modelSelect").select_option("huber")
    check("parameter edit clears result",not page.locator("#results").is_visible())
    check("Huber exclusion defaults",page.locator("#excludedYears").input_value()=="")
    run(); check("Huber converges",report()["response"]["fit"]["converged"])
    page.locator("#modelSelect").select_option("ridge")
    page.locator("#representation").select_option("sa");run()
    check("SA request",report()["request"]["frequency"]=="sa")
    check("SA annual toggle hidden",not page.locator("#valueMode").is_visible())
    page.locator("#modelSelect").select_option("ridge")
    page.locator("#representation").select_option("raw")
    page.locator("#regionSelect").select_option("16")
    page.locator("#operation").select_option("backtest")
    page.locator("#backtestPeriod").select_option("36");run(); out=report()
    check("selected region exported",str(out["region"]["code"])=="16")
    bt=out["response"]["backtest"]
    check("backtest 108 slots",len(bt["rows"])==108)
    check("three horizons complete",all(bt["metrics"][str(h)]["n_valid"]==36 for h in [1,2,12]))
    check("backtest cells populated","—" not in page.locator("#tableBody").inner_text())
    paths=page.locator("#chart path.series").evaluate_all("els=>els.map(el=>el.getAttribute('d'))")
    import re
    end_x=[float(re.findall(r"[ML]([0-9.eE+-]+),",d)[-1]) for d in paths]
    check("backtest dates overlay",len(end_x)==2 and abs(end_x[0]-end_x[1])<1e-8)
    check("backtest annual toggle hidden",not page.locator("#valueMode").is_visible())
    page.set_viewport_size({"width":390,"height":844})
    check("mobile no horizontal overflow",page.evaluate("document.documentElement.scrollWidth<=innerWidth"))
    (ROOT/"verification/screenshots").mkdir(exist_ok=True)
    page.screenshot(path=str(ROOT/"verification/screenshots/mobile.png"),full_page=True)
    page.set_viewport_size({"width":1440,"height":1000})
    page.screenshot(path=str(ROOT/"verification/screenshots/desktop.png"),full_page=True)
    with page.expect_download() as dl: page.locator("#saveSettings").click()
    saved=Path(dl.value.path()).read_bytes()
    page.locator("#advancedToggle").click()
    page.locator("#alpha").fill("2")
    page.locator("#fileInput").set_input_files({"name":"settings.json","mimeType":"application/json","buffer":saved})
    page.wait_for_function("!document.querySelector('#runButton').disabled")
    check("settings roundtrip",page.locator("#alpha").input_value()=="0.3")
    page.locator("#operation").select_option("forecast")
    page.locator("#cutoff").fill("2024-12");run()
    plotted=page.locator("#chart").evaluate("el=>JSON.parse(el.dataset.points)")
    check("chart respects cutoff",max(r["date"] for r in plotted["history"])=="2024-12-01")
    page.locator("#fileInput").set_input_files({"name":"broken.json","mimeType":"application/json","buffer":b"{broken"})
    page.wait_for_function("!document.querySelector('#runButton').disabled")
    check("bad JSON visible error",page.locator("#errorMessage").is_visible())
    check("no JS errors",not errors)
    browser.close()
report={"passed":all(c["pass"] for c in checks),"checks":checks,"platform":"Linux Chrome file:// offline", "html_sha256":hashlib.sha256((ROOT/"regional-inflation.html").read_bytes()).hexdigest()}
(ROOT/"verification/browser_ui.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report))
