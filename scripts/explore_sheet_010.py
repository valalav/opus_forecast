
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

path = 'assets/charts/ОПР_статистика/ОПР_статистика/Основная статистика ЮГУ.xlsx'
df = pd.read_excel(path, sheet_name='010', header=None)

# 1. Extract territory
territory = df.iloc[6, 1]
print(f"Territory: {territory}")

# 2. Extract Dates from row 10
# Dates start from column 2
header_row = df.iloc[10]
dates = []
for val in header_row[2:]:
    if pd.notna(val):
        dates.append(val)

print(f"Dates range: {dates[0]} to {dates[-1]} (Total: {len(dates)} months)")

# 3. Extract Indicators
indicators_data = []
for i in range(11, len(df)):
    row = df.iloc[i]
    name = row[0]
    category = row[1]
    
    if pd.notna(name) and pd.notna(category):
        values = row[2:2+len(dates)].values
        # Clean values (convert to float, handle errors)
        clean_values = []
        for v in values:
            try:
                clean_values.append(float(v))
            except:
                clean_values.append(None)
        
        non_null = [v for v in clean_values if v is not None]
        if len(non_null) > 0:
            indicators_data.append({
                'Indicator': name,
                'Category': category,
                'Points': len(non_null),
                'First_Value': non_null[0],
                'Last_Value': non_null[-1]
            })

res_df = pd.DataFrame(indicators_data)
print("\n--- Found Indicators on Sheet '010' ---")
print(res_df.to_string(index=False))
