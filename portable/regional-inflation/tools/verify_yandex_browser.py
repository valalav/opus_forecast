"""Exercise the actual Yandex update button in the shipped file:// app."""
import json,time,hashlib
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
with sync_playwright() as pw:
    browser=pw.chromium.launch(executable_path="/usr/bin/google-chrome",headless=True,args=["--no-sandbox"])
    page=browser.new_page(accept_downloads=True)
    page.goto((ROOT/"regional-inflation.html").as_uri())
    page.wait_for_function("!document.querySelector('#runButton').disabled")
    started=time.monotonic();page.locator("#loadSource").click()
    page.wait_for_function("!document.querySelector('#loadSource').disabled",timeout=120000)
    assert not page.locator("#errorMessage").is_visible(),page.locator("#errorMessage").inner_text()
    with page.expect_download() as dl:page.locator("#savePackage").click()
    data=json.loads(Path(dl.value.path()).read_text())
    assert len(data["regions"])==101
    assert data["source"]["sha256"]
    result={"ok":True,"file_protocol":True,"actual_update_button":True,"regions":len(data["regions"]),"seconds":time.monotonic()-started,"source":data["source"],"html_sha256":hashlib.sha256((ROOT/"regional-inflation.html").read_bytes()).hexdigest()}
    (ROOT/"verification/browser_yandex.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!="source"}))
    browser.close()
