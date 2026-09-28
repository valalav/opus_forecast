"""Calendar-conditioned three-month produce retrospective; no model promotion.

Compound indices, preserve August 2026 as the information cutoff, and do not
treat overlapping rolling windows as independent observations for p-values.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
SOURCES = ['data/kbr_indices.csv', 'data/access_weights.csv', 'data/items_structure.csv',
           'data/mom_sa_kbr.csv', 'data/micro_forecast_components.csv', 'data/inflation_data.csv']


def hashes():
    return {s: hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in SOURCES}


def main():
    from sirena.data_loader import load_model_data
    before = hashes()
    raw = load_model_data('raw', include_macro=False)
    assert raw.index.max() == pd.Timestamp('2026-08-01')
    indices = pd.read_csv(ROOT/SOURCES[0], parse_dates=['Date'])
    assert indices.Region_code.eq(7).all()
    series = indices[indices.Item_code.eq(33)].set_index('Date').MoM.sort_index()/100
    assert series.index.is_unique and series.notna().all()
    assert series.index.equals(pd.date_range('2010-01-01','2026-08-01',freq='MS'))
    weights = pd.read_csv(ROOT/SOURCES[1])
    weights = weights[weights.Region_code.eq(7) & weights.Day.eq('01/01/26 00:00:00')]
    weight = float(weights.set_index('Item_code').loc[33,'Weight_vertical'])
    windows = {'jan_mar':(1,3), 'apr_jun':(4,6), 'may_jul':(5,7),
               'jun_aug':(6,8), 'aug':(8,8), 'sep':(9,9), 'oct':(10,10),
               'nov':(11,11), 'sep_oct':(9,10), 'sep_nov':(9,11), 'oct_dec':(10,12)}
    def compound(year, start, end):
        values = series.loc[f'{year}-{start:02}':f'{year}-{end:02}']
        return (values.prod()-1)*100 if len(values)==end-start+1 else np.nan
    annual = pd.DataFrame([{'year':year, **{name:compound(year,*months)
        for name, months in windows.items()}} for year in range(2010,2027)]).set_index('year')
    history = annual.loc[:2025]
    annual.to_csv(OUT/'annual_windows.csv', float_format='%.8f')
    monthly = pd.DataFrame({'raw_mom_pct':100*(series-1),
        'trailing_3m_pct':100*(series.rolling(3).apply(np.prod,raw=True)-1)})
    sa = pd.read_csv(ROOT/SOURCES[3],sep=';',decimal=',')
    sa = sa[sa['Код'].eq(33)].iloc[0,2:]
    sa.index = pd.to_datetime(sa.index)
    sa = sa.astype(float)/100
    monthly['sa_mom_pct'] = 100*(sa-1)
    monthly['sa_trailing_3m_pct'] = 100*(sa.rolling(3).apply(np.prod,raw=True)-1)
    monthly['next_3m_pct'] = monthly.trailing_3m_pct.shift(-3)
    monthly.to_csv(OUT/'monthly_windows.csv',index_label='date',float_format='%.8f')
    summaries = []
    for start in [2010,2016,2021]:
        for name in ['sep','oct','nov','sep_oct','sep_nov']:
            y = history.loc[start:,name].dropna()
            positive = int((y>0).sum())
            lo,hi = proportion_confint(positive,len(y),method='wilson')
            summaries.append(dict(sample=f'{start}-2025',period=name,n=len(y),positive=positive,
                positive_share=positive/len(y),wilson_low=lo,wilson_high=hi,
                mean=y.mean(),median=y.median(),q25=y.quantile(.25),q75=y.quantile(.75),
                minimum=y.min(),maximum=y.max()))
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT/'calendar_summary.csv',index=False)
    tests = []
    for feature in ['jun_aug','aug']:
        for outcome in ['sep','oct','sep_nov']:
            r,p = spearmanr(history[feature],history[outcome])
            tests.append(dict(feature=feature,outcome=outcome,n=len(history),rho=r,p_value=p))
    tests = pd.DataFrame(tests)
    tests['holm_p'] = multipletests(tests.p_value,method='holm')[1]
    tests.to_csv(OUT/'summer_associations.csv',index=False)
    # Both predictors are observed by August. k=5 and equal robust scaling are
    # declared descriptive choices, not optimized on autumn outcomes.
    features = ['apr_jun','jun_aug']
    scale = history[features].quantile(.75)-history[features].quantile(.25)
    distance = (((history[features]-annual.loc[2026,features])/scale)**2).sum(axis=1)**.5
    analogs = history.assign(distance=distance).sort_values('distance').iloc[:5]
    analogs.to_csv(OUT/'analogs.csv')
    # Sensitivity: change the descriptor, not the definition of a successful autumn.
    simple = history.assign(distance=(history.jun_aug-annual.loc[2026,'jun_aug']).abs()).sort_values('distance').iloc[:5]
    simple.to_csv(OUT/'analogs_summer_only.csv')
    rolling = monthly.dropna(subset=['trailing_3m_pct','next_3m_pct']).copy()
    rolling = rolling[rolling.index <= '2025-09-01']
    pooled_rho = float(spearmanr(rolling.trailing_3m_pct,rolling.next_3m_pct).statistic)
    for col in ['trailing_3m_pct','next_3m_pct']:
        rolling[col+'_demeaned'] = rolling[col]-rolling.groupby(rolling.index.month)[col].transform('median')
    demeaned_rho = float(spearmanr(rolling.trailing_3m_pct_demeaned,rolling.next_3m_pct_demeaned).statistic)
    # These two correlations are descriptive ONLY: adjacent windows overlap.
    structure = pd.read_csv(ROOT/SOURCES[2])
    components = pd.read_csv(ROOT/SOURCES[4]).merge(
        structure[['Item_code','Subcomponent']],on='Item_code',validate='many_to_one')
    produce = components[components.Subcomponent.eq(33)]
    model = produce.groupby('date').agg(weight=('Weight','sum'),headline_contribution_pp=('Contribution','sum'))
    assert np.allclose(model.weight,weight)
    model['produce_mom_pct'] = model.headline_contribution_pp/model.weight
    model.to_csv(OUT/'current_micro_produce_path.csv')
    produce[produce.date.le('2026-11-01')].to_csv(OUT/'micro_item_forecasts.csv',index=False)
    items = []
    for code in [252,249,349,382,435,506,753,167]:
        item = indices[indices.Item_code.eq(code)].set_index('Date').MoM/100
        recent = item.loc['2026-06':'2026-08']
        old = (item[item.index.year <= 2025]-1)*100
        name = produce.loc[produce.Item_code.eq(code),'Name'].iloc[0]
        items.append(dict(code=code,name=name,weight=float(weights.set_index('Item_code').loc[code,'Weight_vertical']),
            jun_aug_pct=(recent.prod()-1)*100,aug_pct=(item.loc['2026-08-01']-1)*100,
            sep_median_pct=old[old.index.month == 9].median(),
            oct_median_pct=old[old.index.month == 10].median()))
    pd.DataFrame(items).to_csv(OUT/'item_patterns.csv',index=False)
    sensitivity = pd.DataFrame({'produce_mom_pct':[-5,-2,0,2,3,5,7,10]})
    sensitivity['headline_contribution_pp'] = sensitivity.produce_mom_pct*weight
    sensitivity.to_csv(OUT/'headline_sensitivity.csv',index=False)
    actual = monthly.loc['2026-01-01':,['raw_mom_pct']].copy()
    actual['headline_mom_pct'] = raw['Все товары и услуги']-100
    actual['produce_contribution_approx_pp'] = actual.raw_mom_pct*weight
    actual['remaining_contribution_approx_pp'] = actual.headline_mom_pct-actual.produce_contribution_approx_pp
    actual.to_csv(OUT/'actual_2026_contributions.csv',index_label='date')
    assert before == hashes(), 'Input changed during analysis'
    manifest = dict(cutoff='2026-08',weight_2026=weight,complete_years=16,
        source_hashes=before,source_manifest='data/access_source_manifest.json',
        joint_positive_sep_oct=int(((history.sep>0)&(history.oct>0)).sum()),
        rolling_pooled_rho=pooled_rho,rolling_month_demeaned_rho=demeaned_rho,
        rolling_inference='No iid p-value: overlapping monthly windows; descriptive seasonal confounding check.',
        august_produce_headline_contribution_approx_pp=annual.loc[2026,'aug']*weight,
        mean_sep_without_2010=history.loc[2011:,'sep'].mean(),
        no_new_forecast_model=True,production_forecast_changed=False,
        analogy_features=features,analogy_count=5,analogy_scale='historical IQR; no future data')
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    with pd.ExcelWriter(OUT/'produce_retrospective.xlsx',engine='openpyxl') as writer:
        for file,sheet in [('annual_windows','Годы и три месяца'),('calendar_summary','Статистика осени'),
            ('analogs','Аналоги 2026'),('monthly_windows','RAW и SA'),('summer_associations','Связи'),
            ('current_micro_produce_path','Текущая Micro'),('item_patterns','Товары'),('headline_sensitivity','Вклад в ИПЦ')]:
            pd.read_csv(OUT/f'{file}.csv').to_excel(writer,sheet_name=sheet,index=False)
        for sheet in writer.book:
            sheet.freeze_panes='B2'
            sheet.auto_filter.ref=sheet.dimensions
            for col in sheet.columns:
                sheet.column_dimensions[col[0].column_letter].width=min(44,max(17,len(str(col[0].value))+2))
    chart(annual,monthly)
    print(summary[summary['sample'].eq('2010-2025')][['period','positive','n','mean','median']].round(3).to_string(index=False))
    print('rolling correlations',round(pooled_rho,3),round(demeaned_rho,3))
    print('current RAW / SA summer',monthly.loc['2026-08-01',['trailing_3m_pct','sa_trailing_3m_pct']].round(3).to_dict())


def chart(annual,monthly):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history=annual.loc[:2025]
    fig,axes=plt.subplots(1,2,figsize=(13,5.5))
    x=np.arange(4)
    cols=['jun_aug','sep','oct','nov']
    axes[0].boxplot([history[c] for c in cols],positions=x,tick_labels=['Июнь–август','Сентябрь','Октябрь','Ноябрь'],showfliers=True)
    axes[0].scatter([0],[annual.loc[2026,'jun_aug']],color='#d94d36',s=85,zorder=5,label='2026: −7,79%')
    axes[0].axhline(0,color='gray',lw=1)
    axes[0].set_title('КБР, плодоовощи: 2010–2025\nКвартальное снижение не гарантирует рост сентября')
    axes[0].set_ylabel('Изменение цен за указанный период, %')
    axes[0].legend(loc='lower right')
    years=[2016,2018,2021,2023,2024,2026]
    for year in years:
        values=monthly[(monthly.index.year == year) & (monthly.index.month >= 4) & (monthly.index.month <= 11)].raw_mom_pct
        axes[1].plot(values.index.month,values,marker='o',lw=2.7 if year==2026 else 1.3,
                     alpha=1 if year==2026 else .65,label=str(year))
    axes[1].set_xticks(range(4,12),['Апр','Май','Июн','Июл','Авг','Сен','Окт','Ноя'])
    axes[1].set_xlim(3.7,11.3)
    axes[1].axhline(0,color='gray',lw=1)
    axes[1].set_ylabel('Изменение к предыдущему месяцу, %')
    axes[1].set_title('Текущий рисунок и похожие годы\n2026 заканчивается фактом августа')
    axes[1].legend(ncol=2)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT/'produce_patterns.png',dpi=170)
    plt.close(fig)


if __name__=='__main__':
    main()
