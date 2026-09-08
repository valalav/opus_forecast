
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


def generate_nowcast_html(data, output_path, policy=None):
    """Render available weekly nowcasts from the same cache as the forecast table."""
    from html import escape
    from pathlib import Path

    by_month = data.get("diagnostics", {}).get("weekly_bridge", {}).get("by_month", {})
    policy = policy or {}
    policy_values = dict(zip(policy.get("forecast_dates", []), policy.get("mom_pp", [])))
    sections = []
    for date, value in zip(data["forecast_dates"], data["forecasts"].get("Nowcast", [])):
        if value is None:
            continue
        month = date[:7]
        bridge = by_month[month]
        chain = bridge["chain"]
        blend = bridge["nowcast_blend"]
        weeks = chain["weeks"]
        rows = [{
            "Дата отсечки": w["date"],
            "Недельное изменение, %": w["mom"],
            "Накопленное изменение, %": w["cumulative_mom"],
            "Позиций": w["n_items"],
        } for w in weeks]
        table = pd.DataFrame(rows).to_html(
            index=False, border=0, float_format=lambda x: f"{x:+.4f}"
        )
        month_end = bridge.get("month_end") or {}
        price_signal = month_end.get("mom")
        price_text = "нет данных" if price_signal is None else f"{price_signal:+.4f}%"
        policy_text = ""
        if date in policy_values:
            label = ("Предварительный операционный якорь"
                     if date == policy.get("operational_anchor_month")
                     else "Рабочая точка отправочной траектории")
            policy_text = (
                f"<p>{label}: <b>{100 + policy_values[date]:.2f}</b> "
                f"({policy_values[date]:+.2f}% м/м). "
                "Это отдельная оценка; она не заменяется техническим nowcast.</p>"
            )
        sections.append(f"""
        <section id="month-{month}">
          <h2>Nowcast {month}</h2>
          <p class="value">{value:+.3f}% м/м · индекс {100 + value:.2f}</p>
          <p>Недельные отсечки: {len(weeks)} из {chain['weeks_expected']};
          последняя: {weeks[-1]['date']}.</p>
          {policy_text}
          {table}
          <p>Недельная цепочка: {chain['mom']:+.4f}%;
          сигнал с учетом оставшихся {chain['remaining_weeks']} отсечек:
          {chain['extrapolated_mom']:+.4f}%.
          Сопоставление уровней цен: {price_text}.</p>
          <p>Расчет: {blend['weekly_weight']:.0%} ×
          {chain['extrapolated_mom']:+.4f}% +
          {blend['model_weight']:.0%} × {blend['model_proxy']:+.4f}% =
          {value:+.4f}%. Модельная часть — среднее прогнозов,
          использованных в расчете nowcast.</p>
        </section>""")
    if not sections:
        sections.append("<p>Недельных данных для прогнозных месяцев пока нет.</p>")
    html = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nowcast | СИРЕНА-КБР</title>
<style>
body {{font:16px/1.6 Arial,sans-serif;color:#243447;background:#f4f6f8;
max-width:1050px;margin:30px auto;padding:0 20px}}
section {{background:white;border:1px solid #dde3e8;border-radius:10px;
padding:24px;margin:24px 0;overflow-x:auto}}
a {{color:#116681}} h1,h2 {{line-height:1.25}}
.value {{font-size:28px;font-weight:bold;color:#116681}}
table {{border-collapse:collapse;width:100%}} th,td {{padding:10px;
border-bottom:1px solid #dde3e8;text-align:right}} th {{background:#eef5f7}}
</style></head><body>
<nav><a href="index.html">Главная</a> ·
<a href="forecast_table.html">Все прогнозы</a></nav>
<h1>Оперативный nowcast инфляции КБР</h1>
<p>Расчет обновлен: {escape(str(data.get('generated_at', '')))}.
Месячные данные в кэше: {escape(str(data.get('last_data_date', '')))}.</p>
<p>Недельная выборка дает диагностическую оценку и не является официальным
месячным ИПЦ. Nowcast не входит в модельный Ensemble.</p>
{''.join(sections)}
<p>Источники: data/precomputed_forecasts.json,
data/Сравнение еженедельных цен_01.csv,
data/weekly_accounting_month_overrides.csv,
data/send_ready_policy_trajectory.json.
Дата наблюдения учитывается по операционному месяцу из календаря overrides.</p>
</body></html>"""
    Path(output_path).write_text(html, encoding="utf-8")
    return str(output_path)
