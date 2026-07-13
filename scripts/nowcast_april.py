#!/usr/bin/env python3
"""
Nowcast April 2026:
1. Estimate March MoM: prices 30.03 vs 24.02 (last weeks of each month)
2. Calculate April week 1 contribution: prices 06.04 vs 30.03
3. Build April nowcast with 1 week of data
"""
import csv
import json
from collections import defaultdict

# === Config ===
CSV_PATH = "data/Сравнение еженедельных цен_01.csv"
FORECASTS_PATH = "data/precomputed_forecasts.json"

# KBR component weights (Jan 2026)
W_PROD = 0.3986
W_NONPROD = 0.3638
W_SERV = 0.2376
W_TOTAL = W_PROD + W_NONPROD + W_SERV

# Dates
DATE_FEB_LAST = "24.02.2026"
DATE_MAR_LAST = "30.03.2026"
DATE_APR_W1 = "06.04.2026"

# March weekly dates for cumulative check
MARCH_WEEKS = ["02.03.2026", "10.03.2026", "16.03.2026", "23.03.2026", "30.03.2026"]

def parse_price(s):
    """Parse Russian-format price: '749,67' -> 749.67"""
    try:
        return float(s.strip().replace(',', '.').replace('\xa0', ''))
    except:
        return None

def parse_change(s):
    """Parse change index: '100,23' -> 100.23"""
    try:
        return float(s.strip().replace(',', '.').replace('\xa0', ''))
    except:
        return None

def classify_component(comp):
    """Classify component string to category"""
    comp = comp.strip().lower()
    # IMPORTANT: check 'непродовольств' BEFORE 'продовольств' (substring match!)
    if 'непродовольств' in comp:
        return 'nonfood'
    elif 'продовольств' in comp:
        return 'food'
    elif 'услуг' in comp:
        return 'services'
    return None

def main():
    # Read CSV
    with open(CSV_PATH, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, delimiter=';')
        rows = list(reader)
    
    # Strip column names
    for r in rows:
        stripped = {}
        for k, v in r.items():
            stripped[k.strip()] = v
        r.clear()
        r.update(stripped)
    
    # Build price lookup: {date: {item_num: {name, price, change, component}}}
    data = defaultdict(dict)
    for r in rows:
        date = r.get('Name', '').strip()
        name = r.get('Наименование', '').strip()
        price = parse_price(r.get('Средние цены, рублей', ''))
        change = parse_change(r.get('Изменение цен, в % к предыдущей неделе', ''))
        comp_raw = r.get('Справка_нед.Компоненты', r.get('Компонент', '')).strip()
        item_num = r.get('№', '').strip()
        
        if not date or not name:
            continue
        
        comp = classify_component(comp_raw)
        key = item_num if item_num else name
        data[date][key] = {
            'name': name,
            'price': price,
            'change': change,
            'component': comp,
            'comp_raw': comp_raw,
        }
    
    print("=" * 80)
    print("NOWCAST АПРЕЛЬ 2026")
    print("=" * 80)
    
    # === PART 1: Estimate March MoM (30.03 vs 24.02 prices) ===
    print("\n" + "=" * 80)
    print("ЧАСТЬ 1: ОЦЕНКА MoM МАРТА (30.03 vs 24.02 по абсолютным ценам)")
    print("=" * 80)
    
    feb_data = data.get(DATE_FEB_LAST, {})
    mar_data = data.get(DATE_MAR_LAST, {})
    
    print(f"\nТоваров на {DATE_FEB_LAST}: {len(feb_data)}")
    print(f"Товаров на {DATE_MAR_LAST}: {len(mar_data)}")
    
    # Match items and compute price indices
    march_indices = {'food': [], 'nonfood': [], 'services': []}
    march_details = {'food': [], 'nonfood': [], 'services': []}
    
    matched = 0
    for key in mar_data:
        if key in feb_data:
            p_feb = feb_data[key]['price']
            p_mar = mar_data[key]['price']
            comp = mar_data[key]['component']
            name = mar_data[key]['name']
            
            if p_feb and p_mar and p_feb > 0 and comp:
                idx = (p_mar / p_feb) * 100
                march_indices[comp].append(idx)
                march_details[comp].append((name, p_feb, p_mar, idx))
                matched += 1
    
    print(f"Сопоставлено товаров: {matched}")
    
    # Compute weighted March MoM
    food_mean = sum(march_indices['food']) / len(march_indices['food']) if march_indices['food'] else 100
    nonfood_mean = sum(march_indices['nonfood']) / len(march_indices['nonfood']) if march_indices['nonfood'] else 100
    serv_mean = sum(march_indices['services']) / len(march_indices['services']) if march_indices['services'] else 100
    
    march_weighted = (food_mean * W_PROD + nonfood_mean * W_NONPROD + serv_mean * W_SERV) / W_TOTAL
    march_mom = march_weighted - 100
    
    print(f"\n--- Средние индексы по компонентам (30.03 vs 24.02) ---")
    print(f"  Продовольственные: {food_mean:.3f}  ({food_mean - 100:+.3f}%)  [{len(march_indices['food'])} товаров]")
    print(f"  Непродовольственные: {nonfood_mean:.3f}  ({nonfood_mean - 100:+.3f}%)  [{len(march_indices['nonfood'])} товаров]")
    print(f"  Услуги: {serv_mean:.3f}  ({serv_mean - 100:+.3f}%)  [{len(march_indices['services'])} товаров]")
    print(f"\n  >>> ОЦЕНКА MoM МАРТА (взвешенный): {march_weighted:.3f}  ({march_mom:+.3f}%)")
    
    # Top drivers March
    all_march = []
    for comp in march_details:
        for name, p_feb, p_mar, idx in march_details[comp]:
            weight = W_PROD if comp == 'food' else (W_NONPROD if comp == 'nonfood' else W_SERV)
            n_items = len(march_indices[comp])
            contribution = (idx - 100) * weight / n_items / W_TOTAL  # approx contribution to total
            all_march.append((name, comp, idx, p_feb, p_mar, contribution))
    
    print(f"\n--- Топ-10 драйверов РОСТА Марта ---")
    for name, comp, idx, p_feb, p_mar, contr in sorted(all_march, key=lambda x: x[2], reverse=True)[:10]:
        print(f"  {idx:7.2f} ({idx-100:+.2f}%)  {p_feb:.2f}→{p_mar:.2f}  [{comp[:4]}]  {name}")
    
    print(f"\n--- Топ-10 драйверов СНИЖЕНИЯ Марта ---")
    for name, comp, idx, p_feb, p_mar, contr in sorted(all_march, key=lambda x: x[2])[:10]:
        print(f"  {idx:7.2f} ({idx-100:+.2f}%)  {p_feb:.2f}→{p_mar:.2f}  [{comp[:4]}]  {name}")

    # === PART 1b: Verify with cumulative weekly method for March ===
    print(f"\n--- Проверка: Кумулятивный цепной метод за Март ---")
    cumulative = 1.0
    for week_date in MARCH_WEEKS:
        week_data = data.get(week_date, {})
        if not week_data:
            print(f"  {week_date}: НЕТ ДАННЫХ")
            continue
        
        comp_changes = {'food': [], 'nonfood': [], 'services': []}
        for key, item in week_data.items():
            ch = item['change']
            comp = item['component']
            if ch is not None and comp:
                comp_changes[comp].append(ch)
        
        f_m = sum(comp_changes['food']) / len(comp_changes['food']) if comp_changes['food'] else 100
        n_m = sum(comp_changes['nonfood']) / len(comp_changes['nonfood']) if comp_changes['nonfood'] else 100
        s_m = sum(comp_changes['services']) / len(comp_changes['services']) if comp_changes['services'] else 100
        
        w_idx = (f_m * W_PROD + n_m * W_NONPROD + s_m * W_SERV) / W_TOTAL
        cumulative *= (w_idx / 100)
        week_change = w_idx - 100
        
        print(f"  {week_date}: прод={f_m:.3f} непрод={n_m:.3f} усл={s_m:.3f} → взвеш={w_idx:.3f} ({week_change:+.3f}%)  cum={(cumulative-1)*100:+.3f}%")
    
    march_cumulative = (cumulative - 1) * 100
    print(f"\n  >>> Кумулятивный MoM Марта (цепной): {march_cumulative:+.3f}%")
    print(f"  >>> Оценка MoM Марта (абс. цены):   {march_mom:+.3f}%")
    
    # === PART 2: April Week 1 (06.04 vs 30.03) ===
    print("\n" + "=" * 80)
    print("ЧАСТЬ 2: ПЕРВАЯ НЕДЕЛЯ АПРЕЛЯ (06.04 vs 30.03)")
    print("=" * 80)
    
    apr_data = data.get(DATE_APR_W1, {})
    print(f"\nТоваров на {DATE_APR_W1}: {len(apr_data)}")
    
    # Use price index change column (wow)
    apr_comp_changes = {'food': [], 'nonfood': [], 'services': []}
    apr_details = []
    
    for key, item in apr_data.items():
        ch = item['change']
        comp = item['component']
        name = item['name']
        price = item['price']
        
        if ch is not None and comp:
            apr_comp_changes[comp].append(ch)
            
            # Also compute from absolute prices
            mar_item = mar_data.get(key)
            p_mar = mar_item['price'] if mar_item else None
            
            apr_details.append((name, comp, ch, price, p_mar))
    
    f_apr = sum(apr_comp_changes['food']) / len(apr_comp_changes['food']) if apr_comp_changes['food'] else 100
    n_apr = sum(apr_comp_changes['nonfood']) / len(apr_comp_changes['nonfood']) if apr_comp_changes['nonfood'] else 100
    s_apr = sum(apr_comp_changes['services']) / len(apr_comp_changes['services']) if apr_comp_changes['services'] else 100
    
    w_apr = (f_apr * W_PROD + n_apr * W_NONPROD + s_apr * W_SERV) / W_TOTAL
    apr_w1_change = w_apr - 100
    
    print(f"\n--- Индексы первой недели Апреля (06.04 vs 30.03) ---")
    print(f"  Продовольственные: {f_apr:.3f}  ({f_apr - 100:+.3f}%)  [{len(apr_comp_changes['food'])} товаров]")
    print(f"  Непродовольственные: {n_apr:.3f}  ({n_apr - 100:+.3f}%)  [{len(apr_comp_changes['nonfood'])} товаров]")
    print(f"  Услуги: {s_apr:.3f}  ({s_apr - 100:+.3f}%)  [{len(apr_comp_changes['services'])} товаров]")
    print(f"\n  >>> Вклад 1-й недели Апреля (взвешенный): {apr_w1_change:+.3f}%")
    
    # Top movers this week
    print(f"\n--- Топ-10 РОСТ (06.04) ---")
    for name, comp, ch, price, p_mar in sorted(apr_details, key=lambda x: x[2], reverse=True)[:10]:
        mar_str = f"{p_mar:.2f}→" if p_mar else "?→"
        print(f"  {ch:7.2f} ({ch-100:+.2f}%)  {mar_str}{price:.2f}  [{comp[:4]}]  {name}")
    
    print(f"\n--- Топ-10 СНИЖЕНИЕ (06.04) ---")
    for name, comp, ch, price, p_mar in sorted(apr_details, key=lambda x: x[2])[:10]:
        mar_str = f"{p_mar:.2f}→" if p_mar else "?→"
        print(f"  {ch:7.2f} ({ch-100:+.2f}%)  {mar_str}{price:.2f}  [{comp[:4]}]  {name}")

    # === PART 3: Nowcast April ===
    print("\n" + "=" * 80)
    print("ЧАСТЬ 3: NOWCAST АПРЕЛЬ 2026")
    print("=" * 80)
    
    # Weekly signal: 1 week cumulative + extrapolation
    cumulative_apr = apr_w1_change  # Only 1 week
    
    # Extrapolation: avg_weekly × decay × remaining_weeks
    avg_weekly = apr_w1_change
    remaining_weeks = 3  # 4 - 1
    decay = 0.6  # April is "normal" month
    extrapolated = avg_weekly * decay * remaining_weeks
    weekly_signal = cumulative_apr + extrapolated
    
    # Read ensemble h=1
    try:
        with open(FORECASTS_PATH, 'r') as f:
            forecasts = json.load(f)
        ensemble_vals = []
        skip_models = {'Nowcast', 'Micro', 'SubcomponentMulti'}
        for model, vals in forecasts.get('forecasts', {}).items():
            if model not in skip_models and vals and vals[0] is not None:
                ensemble_vals.append(vals[0])
        ensemble_h1 = sum(ensemble_vals) / len(ensemble_vals) if ensemble_vals else 0.3
    except:
        ensemble_h1 = 0.3  # fallback
    
    # 1 week → 60% weekly / 40% model
    w_weekly = 0.6
    w_model = 0.4
    nowcast_apr = w_weekly * weekly_signal + w_model * ensemble_h1
    
    print(f"\n  Кумулятивный (1 нед):     {cumulative_apr:+.3f}%")
    print(f"  Экстраполяция (3 нед):    {extrapolated:+.3f}%")
    print(f"  Weekly Signal:            {weekly_signal:+.3f}%")
    print(f"  Ensemble h=1:             {ensemble_h1:+.3f}%")
    print(f"  Веса:                     {w_weekly:.0%} weekly / {w_model:.0%} model")
    print(f"\n  >>> NOWCAST АПРЕЛЬ 2026:  {nowcast_apr:+.3f}%")
    
    # Context: what ensemble models say
    print(f"\n--- Ensemble h=1 (модели) ---")
    try:
        for model, vals in sorted(forecasts.get('forecasts', {}).items()):
            if model not in skip_models and vals and vals[0] is not None:
                print(f"  {model:25s}: {vals[0]:+.3f}%")
    except:
        pass
    
    # === SUMMARY ===
    print("\n" + "=" * 80)
    print("ИТОГ")
    print("=" * 80)
    print(f"  Оценка MoM Марта (абс):     {march_mom:+.3f}%")
    print(f"  Оценка MoM Марта (цепной):   {march_cumulative:+.3f}%")
    print(f"  Вклад 1-й недели Апреля:     {apr_w1_change:+.3f}%")
    print(f"  Nowcast Апрель (1 нед):      {nowcast_apr:+.3f}%")
    print(f"  Инсайдерская оценка Марта:   ~+0.50%")
    print("=" * 80)

if __name__ == '__main__':
    main()
