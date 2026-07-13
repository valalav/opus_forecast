# Патч для замены генерации экзогенных прогнозов

old_block = '''    # --- EXOGENOUS PROJECTIONS ---
    # Generate simple projections for visualization
    exog_dates = forecast['Date']
    
    # USD Projection: Current * (1 + Shock) distributed over time or immediate
    # Visualizing the "Scenario" implies showing the path resulting from the shock
    # If shock is 10%, we show USD rising by 10%.
    usd_proj = []
    for i in range(horizon):
        # Simple drift + shock
        # Shock is applied as pass-through in model, here we visualize the rate itself
        # Assume shock fully realizes over 3 months
        shock_mult = 1.0 + (fx_shock_pct / 100.0) * (min(i+1, 3)/3.0)
        usd_proj.append(current_usd * shock_mult)
        
    key_rate_proj = [current_key_rate] * horizon # Flat assumption for now'''

new_block = '''    # --- EXOGENOUS PROJECTIONS ---
    exog_dates = forecast['Date']
    
    # Проверяем ручные значения, иначе генерируем авто
    manual_exog = load_manual_exog()
    if manual_exog and manual_exog.get('source') == 'manual':
        # Используем ручные значения
        usd_proj = manual_exog['USD'][:horizon]
        key_rate_proj = manual_exog['KeyRate'][:horizon]
        ruonia_proj = manual_exog['RUONIA'][:horizon]
        exog_source = "Ручной"
    else:
        # Генерируем AR(1) прогноз от последних фактов
        auto_exog = generate_auto_exog_forecast(last_date, horizon)
        usd_proj = auto_exog['USD']
        key_rate_proj = auto_exog['KeyRate']
        ruonia_proj = auto_exog['RUONIA']
        exog_source = "Авто (AR1)"
    
    # Добавляем FX шок поверх прогноза если задан
    if fx_shock_pct != 0:
        usd_proj = [u * (1 + fx_shock_pct/100) for u in usd_proj]
        exog_source += f" + FX шок {fx_shock_pct:+.1f}%"'''

with open('dashboard_v32.py', 'r', encoding='utf-8') as f:
    content = f.read()

if old_block in content:
    content = content.replace(old_block, new_block)
    with open('dashboard_v32.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ Патч применён")
else:
    print("❌ Блок не найден")
    # Попробуем найти часть
    if '# --- EXOGENOUS PROJECTIONS ---' in content:
        print("   Найден заголовок, но формат отличается")
