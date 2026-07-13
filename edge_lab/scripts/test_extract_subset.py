#!/usr/bin/env python3

import sys

sys.path.insert(0, "/home/valalav/_projects/sirena-kbr/edge_lab")

from scripts.extract_sectoral_kbr import extract_sheet_data
import openpyxl

EXCEL_FILE = "assets/charts/ОПР_статистика/Основная статистика ЮГУ.xlsx"
OUTPUT_FILE = "data/test_kbr_sectoral.csv"

test_sheets = ["100", "101", "102", "118", "162", "180", "182", "183", "184", "200"]

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = __import__("csv").writer(f)
    writer.writerow(["Date", "Value", "Indicator", "Metric_Type", "Sheet"])

    wb = openpyxl.load_workbook(EXCEL_FILE, read_only=True, data_only=True)

    for sheet_name in test_sheets:
        sheet = wb[sheet_name]
        data = extract_sheet_data(sheet_name, sheet)
        print(f"Sheet {sheet_name}: {len(data)} records")
        writer.writerows(data)

    wb.close()

print(f"\nOutput: {OUTPUT_FILE}")
