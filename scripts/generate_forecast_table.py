
import json
import pandas as pd
import os

# Production Ensemble Weights (v4.8 from sirena/models/__init__.py)
MODEL_WEIGHTS = {
    'Huber': 0.18,
    'RidgeShockDummies': 0.17,
    'ElasticNet': 0.17,
    'NGBoostShock': 0.16,
    'NGBoost': 0.12,
    'Ridge': 0.08,
    'RidgeExtended': 0.05,
    'Prophet': 0.04,
    'EBM': 0.03
}

def load_forecasts(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def generate_html_table(data, output_path):
    forecasts = data['forecasts']
    dates = data['forecast_dates']
    
    # Create DataFrame
    df = pd.DataFrame({'Date': dates})
    
    available_models = []
    
    # 1. Models with weights
    for model, weight in MODEL_WEIGHTS.items():
        if model in forecasts and forecasts[model] is not None:
             weight_pct = int(weight * 100)
             col_name = f"{model} ({weight_pct}%)"
             df[col_name] = forecasts[model]
             available_models.append(col_name)
    
    # 2. Other models (Auxiliary/Experimental)
    for model, values in forecasts.items():
        # Skip if already added or if it's the Ensemble itself
        if model in MODEL_WEIGHTS or model == 'Ensemble' or values is None:
            continue
        
        col_name = f"{model} (Aux)"
        df[col_name] = values
        available_models.append(col_name)

    # 3. Add Ensemble column
    if 'Ensemble' in forecasts:
        df['Ensemble'] = forecasts['Ensemble']
    
    # Final column order: Date, Ensembe, Models...
    cols = ['Date', 'Ensemble'] + available_models
    # Filter columns that actually exist in the dataframe
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    
    # Convert dates to readable string
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%b %Y')
    
    # Generate HTML with custom styling
    html = """
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { border-collapse: collapse; width: 100%; box-shadow: 0 0 20px rgba(0,0,0,0.1); }
            th, td { padding: 12px 15px; text-align: center; border-bottom: 1px solid #ddd; }
            th { background-color: #009879; color: #ffffff; position: sticky; top: 0; }
            tr:hover { background-color: #f5f5f5; }
            tr:nth-child(even) { background-color: #f3f3f3; }
            .ensemble-col { font-weight: bold; background-color: #e8f8f5 !important; border-left: 2px solid #009879; border-right: 2px solid #009879; }
            h2 { color: #333; }
        </style>
    </head>
    <body>
        <h2>Прогноз инфляции по моделям (v5.2)</h2>
    """
    
    # Use Pandas styling directly for formatted values
    styled_df = df.style.format({col: "{:.2f}" for col in df.columns if col != 'Date'}) \
                  .set_table_attributes('class="dataframe"') \
                  .set_properties(subset=['Ensemble'], **{'class': 'ensemble-col', 'font-weight': 'bold'}) \
                  .hide(axis="index")
    
    html += styled_df.to_html()
    html += "</body></html>"
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"Table saved to {output_path}")

if __name__ == "__main__":
    INPUT_FILE = "data/precomputed_forecasts.json"
    OUTPUT_FILE = "assets/charts/forecast_table.html"
    
    if os.path.exists(INPUT_FILE):
        data = load_forecasts(INPUT_FILE)
        generate_html_table(data, OUTPUT_FILE)
    else:
        print(f"Error: {INPUT_FILE} not found. Run scripts/precompute_forecasts.py first.")
