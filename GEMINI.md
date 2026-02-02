# GEMINI.md

This file provides guidance to Gemini when working with code in this repository.

## Архитектура документации (Универсальные принципы)

> **Эти принципы применимы ко ВСЕМ проектам для оптимизации контекстного окна.**

### 1. Lazy Loading (Ленивая загрузка)

Не читай всю документацию сразу. Открывай только то, что нужно для текущей задачи:
- Добавляешь модель → `docs/ADDING_MODEL_GUIDE.md`
- Проверяешь систему → `docs/VERIFICATION_GUIDE.md`
- Нужен API → `docs/API.md`

### 2. Index Pattern (Индексный паттерн)

`docs/index.md` — навигационный хаб с кратким описанием каждого файла. Начинай с индекса.

### 3. Skill Pattern (Паттерн скилов)

Сложные workflow в `.agent/workflows/`:
- `/update-nowcast` — обновление nowcast
- `/add-model` — добавление модели
- `/run-backtest` — запуск бэктеста

### 4. Компактный GEMINI.md

GEMINI.md содержит только:
- Критические директивы (не врать)
- Ссылки на документацию
- Базовые команды

**Цель:** < 400 строк

---

## КРИТИЧЕСКАЯ ДИРЕКТИВА: НЕ ВРАТЬ

**Это самое важное правило. Нарушение = потеря доверия.**

### Что запрещено:

1. **Говорить "готово" без РЕАЛЬНОЙ проверки**
   - "Скрипт запустился" ≠ "UI работает"
   - "Код без ошибок" ≠ "Функционал доступен"

2. **Создавать верификацию которая проверяет не то**

3. **Оптимизировать на "выглядит как работа"**

### Что обязательно:

1. **Перед "готово" — проверь ИМЕННО то, что просил пользователь**
2. **Если не можешь проверить — честно скажи**
3. **Если "не работает" — сначала ПОСМОТРИ код, потом отвечай**

---

## При открытии новой сессии

1. Проверь `CURRENT_TASK.md` — незавершенные задачи
2. Проверь `git log -1` — последний коммит
3. Предложи продолжить работу

---

## Project Overview

**СИРЕНА-КБР v5.4** — система прогнозирования инфляции (ИПЦ) в Кабардино-Балкарской Республике.

**Ансамбль:** 9 моделей (Huber, ElasticNet, NGBoost, Prophet и др.)

**Dashboard:** `http://localhost:8503`

---

## Ключевая документация

| Документ | Когда открывать |
|----------|----------------|
| **[docs/index.md](docs/index.md)** | Навигация по всей документации |
| **[docs/ADDING_MODEL_GUIDE.md](docs/ADDING_MODEL_GUIDE.md)** | Добавление новой модели |
| **[docs/VERIFICATION_GUIDE.md](docs/VERIFICATION_GUIDE.md)** | Верификация системы |
| **[docs/MODEL_CATALOG.md](docs/MODEL_CATALOG.md)** | Каталог моделей с примерами |
| **[docs/EDGE_LAB_REFERENCE.md](docs/EDGE_LAB_REFERENCE.md)** | Ralph / Edge Lab |
| **[docs/NOWCASTING.md](docs/NOWCASTING.md)** | Недельный nowcast |

---

## Основные команды

```bash
# Dashboard (порт 8503!)
streamlit run dashboard.py --server.port 8503

# Пересчёт прогнозов
python3 scripts/precompute_forecasts.py

# Генерация графиков
python3 scripts/generate_charts.py

# Бэктест
python3 scripts/run_backtest_h1.py

# Тесты
pytest tests/ -v

# Верификация
python3 scripts/verify_all_tabs.py
```

---

## Workflows (Slash Commands)

| Команда | Описание |
|---------|----------|
| `/update-nowcast` | Обновить nowcast из недельных данных |
| `/add-model` | Добавить новую модель (11 шагов) |
| `/run-backtest` | Запустить бэктест |

---

## File Organization

- **Root**: Только `README.md`, `GEMINI.md`, `dashboard.py`, `requirements.txt`
- **Scripts**: `scripts/` — все скрипты
- **Data**: `data/` — все CSV/Excel
- **Docs**: `docs/` — вся документация
- **Archive**: `archive/` — устаревшее

---

## Верификация перед "Готово"

```bash
# Полная проверка
python3 scripts/verify_all_tabs.py

# Результат должен быть: ✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ
```

Детали: [docs/VERIFICATION_GUIDE.md](docs/VERIFICATION_GUIDE.md)

---

## Nowcast (Оперативная корректировка)

Формула: `Nowcast = 80% × Weekly + 20% × Model`

Обновление: `/update-nowcast` или:
```bash
python3 scripts/precompute_forecasts.py
python3 scripts/generate_charts.py
```

**ВАЖНО:** Использовать столбец `Справка_нед.Компоненты` (НЕ `Компонент`).

Детали: [docs/NOWCASTING.md](docs/NOWCASTING.md)

---

## Ralph Edge Lab

Автономная система Worker-Critic-Refiner.

```bash
cd edge_lab
python3 system/orchestrator.py
```

---

## 🛡️ Critical Review Protocol
For financial or legal analysis, use the `/critical-review` skill.
- **Trigger**: "Check this critically", "Verify sources".
- **Action**: Enforces Tier 1 Source verification and Counter-Argument search.

### ✅ Verification Protocol (New)
- **Use `/critical-review`**: For any sensitive task (Legal, Financial, Medical), invoke this skill to enforce adversarial verification.
- **Protocol**: Source Hierarchy Audit -> Devil's Advocate Loop -> Verified Output.

## 🧮 Методология (Methodology)

### Расчет вклада в инфляцию (Contribution)
> **Формула:** `Вклад (п.п.) = Прирост цены (%) × Вес в корзине (доля)`

**Пример:**
- Товар: Говядина
- Рост цены: +29.6%
- Вес в корзине: 0.0158 (1.58%)
- Вклад: `29.6 * 0.0158 = 0.467 п.п.` (в общем индексе)

**Источники весов:**
1. **`data/micro_sprav.csv`** — Первичный источник (Веса Росстата). Колонки: `Item_code; Товар; Weight`.
2. **`data/access_weights.csv`** — Вторичный (агрегаты). Использовать с осторожностью (коды могут не совпадать).

**Правило:** При анализе драйверов инфляции **ВСЕГДА** учитывать вес. Товар с ростом +100% и весом 0.0001% (Спички) менее важен, чем товар с ростом +10% и весом 1.5% (Мясо).
