"""Reuse XML-preserving OPR update; put September OPR in comparison columns."""
from pathlib import Path
import importlib.util,zipfile,xml.etree.ElementTree as ET,shutil,io,datetime
import openpyxl
from openpyxl.utils.datetime import to_excel
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("previous_update",ROOT/"archive/results/forecast_update_20260928/update_opr.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.OUT=OUT
m.main()
target=ROOT/"assets/06_2026_02_Прогноз.xlsx"
with zipfile.ZipFile(target) as z:parts={n:z.read(n) for n in z.namelist()}
ns=m.NS
wb=ET.fromstring(parts["xl/workbook.xml"])
sheet=next(s for s in wb.find("s:sheets",ns) if s.attrib["name"]=="Прогноз")
rid=sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
rels=ET.fromstring(parts["xl/_rels/workbook.xml.rels"])
path=next(r.attrib["Target"] for r in rels if r.attrib["Id"]==rid)
path=path.lstrip("/") if path.startswith("/") else "xl/"+path
xml=ET.fromstring(parts[path])
old=openpyxl.load_workbook(OUT/"OPR092026_source.xlsx",data_only=True,read_only=True)["Прогноз"]
for row in xml.findall("s:sheetData/s:row",ns):
    n=int(row.attrib["r"]);cells={c.attrib["r"]:c for c in row}
    edits={}
    if n==1:edits={"E":to_excel(datetime.datetime(2026,9,28)),"F":to_excel(datetime.datetime(2026,9,28)),"G":to_excel(datetime.datetime(2026,9,1)),"H":to_excel(datetime.datetime(2026,9,1))}
    elif n>=3 and hasattr(old[f"A{n}"].value,"year"):
        edits={"G":old[f"E{n}"].value,"H":old[f"F{n}"].value}
    for col,value in edits.items():
        if not isinstance(value,(int,float)):continue
        key=f"{col}{n}";cell=cells.get(key)
        if cell is None:cell=ET.SubElement(row,f'{{{ns["s"]}}}c',{"r":key})
        cell.attrib.pop("t",None)
        for tag in ["f","is"]:
            for child in cell.findall("s:"+tag,ns):cell.remove(child)
        v=cell.find("s:v",ns)
        if v is None:v=ET.SubElement(cell,f'{{{ns["s"]}}}v')
        v.text=format(value,".12g")
parts[path]=ET.tostring(xml,encoding="utf-8",xml_declaration=True)
with zipfile.ZipFile(target,"w",compression=zipfile.ZIP_DEFLATED) as z:
    for name,value in parts.items():z.writestr(name,value)
shutil.copy2(target,OUT/"OPR102026_Прогноз.xlsx")
print("OPR10 primary E:F; OPR09 comparison G:H; all other workbook parts preserved")
