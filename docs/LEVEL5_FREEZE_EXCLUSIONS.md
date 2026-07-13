# Исключения Level-5 для анализа заморозок

## Область применения

Этот файл фиксирует пересобранный список исключений для анализа заморозок, ограниченного **только позициями Level-5 / микрокомпонентами**.

Правила текущей версии:

- исключать **нотариальные** позиции;
- исключать **ЖКХ Level-5** позиции;
- **не использовать** агрегаты, укрупнённые buckets и дублирующие более широкие строки;
- при наличии и микропозиции, и более широкого bucket-а оставлять в анализе только **точную Level-5 позицию**.

## Что в репозитории подтверждает Level-5 / микрокомпоненты

- `docs/FREEZE_ANALYSIS.md` указывает, что freeze-analysis построен на **552 товарах Level-5**.
- `scripts/level5_basket_analysis.py` строит `docs/level5_basket_analysis.csv` как анализ **Top Level 5** на базе `data/kbr_full_monthly.csv`, `data/access_weights.csv` и `data/items_names.csv`.
- `sirena/models/microcomponent.py` описывает микрокомпонентную модель как **Bottom-Up, Level 5**.
- `sirena/models/hierarchical_micro.py` задаёт иерархию **Micro -> Subcomp -> Component -> Total**.
- `data/raw/micro_sprav.csv` даёт наиболее прямую repo-привязку микрокомпонента к более широким уровням через поля `Item_code;Товар;Компонент;Субкомпонент;Weight`.

## Исключить сейчас

Ниже приведены только точные позиции Level-5, без агрегатов и без укрупнённых категорий.

| Item_code | Item_name | Основание для исключения | Подтверждение в репозитории |
|---|---|---|---|
| 162 | Взносы на капитальный ремонт, м2 | ЖКХ Level-5 | `data/raw/micro_sprav.csv`, `docs/level5_basket_analysis.csv`, `docs/micro_basket_signals.csv` |
| 400 | Наём жилых помещений в государственном и муниципальном жилищных фондах, м2 общей площади | ЖКХ Level-5 | `data/items_names.csv`, `docs/level5_basket_analysis.csv`, `docs/micro_basket_signals.csv` |
| 416 | Обращение с твердыми коммунальными отходами, с человека | ЖКХ Level-5 | `data/raw/micro_sprav.csv`, `docs/level5_basket_analysis.csv`, `docs/micro_basket_signals.csv` |
| 444 | Оплата горячего водоснабжения | ЖКХ Level-5 | `data/items_names.csv`, `docs/level5_basket_analysis.csv`, `docs/micro_basket_signals.csv` |
| 445 | Оплата жилья в домах государственного и муниципального жилищных фондов | ЖКХ Level-5 | `data/items_names.csv`, `docs/level5_basket_analysis.csv`, `docs/micro_basket_signals.csv` |
| 446 | Оплата холодного водоснабжения и водоотведения | ЖКХ Level-5 | `data/items_names.csv`, `docs/level5_basket_analysis.csv`, `docs/micro_basket_signals.csv` |
| 449 | Оформление доверенности в нотариальной конторе, услуга | Нотариальная Level-5 позиция | `data/items_names.csv`, `docs/level5_basket_analysis.csv`, `docs/micro_basket_signals.csv` |
| 545 | Ремонт жилищ | ЖКХ Level-5 | `data/items_names.csv`, `docs/level5_basket_analysis.csv`, `docs/micro_basket_signals.csv` |
| 589 | Содержание и ремонт жилья в государственном и муниципальном жилищных фондах, м2 общей площади | ЖКХ Level-5 | `data/items_names.csv`, `docs/level5_basket_analysis.csv`, `docs/micro_basket_signals.csv` |
| 590 | Содержание, ремонт жилья для граждан-собственников жилья в результате приватизации, граждан-собственников жилых помещений по иным основаниям, м2 общей площади | ЖКХ Level-5 | `data/raw/micro_sprav.csv`, `data/items_names.csv` |
| 664 | Удостоверение завещания в нотариальной конторе, услуга | Нотариальная Level-5 позиция | `data/items_names.csv`, `docs/level5_basket_analysis.csv`, `docs/micro_basket_signals.csv` |
| 676 | Услуги организаций ЖКХ, оказываемые населению | ЖКХ Level-5 | `data/items_names.csv`, `data/all_anomalies_2025.csv` |
| 679 | Услуги по организации и выполнению работ по эксплуатации домов ЖК, ЖСК, ТСЖ, м2 общей площади | ЖКХ Level-5 | `data/raw/micro_sprav.csv`, `data/items_names.csv` |
| 681 | Услуги по снабжению электроэнергией | ЖКХ Level-5 | `data/items_names.csv`, `docs/level5_basket_analysis.csv`, `docs/micro_basket_signals.csv` |
| 969 | Обращение с твердыми коммунальными отходами | ЖКХ Level-5 | `docs/level5_basket_analysis.csv`, `docs/micro_basket_signals.csv`, `data/all_anomalies_2025.csv` |

## Что не включать в анализ как Level-5

Следующие строки являются более широкими категориями / buckets и не должны использоваться в пересобранном Level-5 анализе:

| Item_code | Item_name |
|---|---|
| 1 | Все товары и услуги |
| 2 | Непродовольственные товары |
| 4 | Услуги |
| 6 | Все товары и услуги без плодоовощей, топлива и ЖКУ |
| 7 | Непродовольственные товары без топлива моторного |
| 8 | Продовольственные товары без плодоовощной продукции |
| 9 | Услуги без жилищно-коммунальных услуг |
| 14 | Жилищные и коммунальные услуги (включая аренду квартир) |
| 53 | Другие продовольственные товары |
| 54 | Другие непродовольственные товары |
| 55 | Другие услуги |
| 436 | Одежда |
| 510 | Посреднические и прочие услуги |

Эти коды нужны только как пример того, что следует вычищать из Level-5 выборки как укрупнённые сущности.

## Отдельно изучать, но пока не смешивать с exclusion-листом

Образовательные позиции пока не входят в текущий exclusion-лист. Их нужно держать в отдельном блоке анализа, чтобы не смешивать с уже подтверждёнными исключениями ЖКХ и нотариальных услуг.

## Проверенные исходники

- `docs/FREEZE_ANALYSIS.md`
- `scripts/level5_basket_analysis.py`
- `docs/level5_basket_analysis.csv`
- `docs/micro_basket_signals.csv`
- `data/items_names.csv`
- `data/raw/micro_sprav.csv`
- `sirena/models/microcomponent.py`
- `sirena/models/hierarchical_micro.py`
- `data/all_anomalies_2025.csv`
