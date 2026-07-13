#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

FILE_PATH = (
    "assets/charts/ОПР_статистика/Бюджеты/Консолидированные бюджеты субъектов РФ.xlsx"
)
OUTPUT_PATH = "data/kbr_budget_consolidated.csv"


def extract_kbr_budget():
    file_path = Path(FILE_PATH)
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load sheets
    df_income_total = pd.read_excel(file_path, sheet_name="Доходы_всего", header=None)
    df_expense = pd.read_excel(file_path, sheet_name="Расходы", header=None)

    # Find KBR rows
    kbr_income_row = None
    kbr_expense_row = None

    # Find KBR in Доходы_всего
    for idx, row in df_income_total.iterrows():
        if pd.notna(row[0]) and "кабардино" in str(row[0]).lower():
            kbr_income_row = row
            break

    # Find KBR TOTAL expenses in Расходы (look for "ИТОГО")
    for idx, row in df_expense.iterrows():
        if pd.notna(row[0]) and "кабардино" in str(row[0]).lower():
            code = str(row[1]) if pd.notna(row[1]) else ""
            if "итог" in code.lower() and "расхо" in code.lower():
                kbr_expense_row = row
                break

    if kbr_income_row is None or kbr_expense_row is None:
        raise ValueError("Could not find KBR data in budget file")

    # Extract dates and data from both sheets
    # Income sheet: dates in row 1, starting at column 2
    income_header = df_income_total.iloc[1]
    # Expense sheet: dates in row 0, starting at column 2
    expense_header = df_expense.iloc[0]

    # Use minimum length
    max_cols = min(
        len(income_header),
        len(expense_header),
        len(kbr_income_row),
        len(kbr_expense_row),
    )

    extracted_data = []

    for i in range(2, max_cols):
        inc_date = income_header[i]
        exp_date = expense_header[i]
        inc_val = kbr_income_row[i]
        exp_val = kbr_expense_row[i]

        # Only add rows where we have valid dates (using income date as primary)
        if pd.notna(inc_date):
            date_str = (
                str(inc_date)[:10] if isinstance(inc_date, str) else str(inc_date)[:10]
            )

            income_float = float(inc_val) if pd.notna(inc_val) else None
            expense_float = float(exp_val) if pd.notna(exp_val) else None

            # Calculate deficit (negative = deficit, positive = surplus)
            if pd.notna(inc_val) and pd.notna(exp_val):
                deficit = float(inc_val) - float(exp_val)
            else:
                deficit = None

            extracted_data.append(
                {
                    "Date": date_str,
                    "Доходы": income_float,
                    "Расходы": expense_float,
                    "Дефицит_Профицит": deficit,
                }
            )

    df = pd.DataFrame(extracted_data)

    # Filter out rows with all None values
    df = df[df["Date"].notna() | df["Доходы"].notna() | df["Расходы"].notna()]

    # Save to CSV
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Saved {len(df)} rows to {output_path}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"\nSample data:")
    print(df.head(15))
    print(df.tail(10))

    # Summary statistics
    print(f"\nSummary:")
    print(f"  Rows with Income: {df['Доходы'].notna().sum()}")
    print(f"  Rows with Expense: {df['Расходы'].notna().sum()}")
    print(f"  Rows with Deficit/Surplus: {df['Дефицит_Профицит'].notna().sum()}")

    return df


if __name__ == "__main__":
    extract_kbr_budget()
