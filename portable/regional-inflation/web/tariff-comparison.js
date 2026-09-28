/* Common-origin tariff comparisons; no fitting or tuning on the evaluation window. */
(function(root){
  'use strict';
  const finite=x=>typeof x==='number'&&Number.isFinite(x);
  function monthShift(date,delta){const [y,m]=date.split('-').map(Number),d=new Date(Date.UTC(y,m-1+delta,1));return d.toISOString().slice(0,10);}
  function visible(calendar,date,origin){
    const end=monthShift(origin,1); // strict < next month, includes last day of origin month
    return calendar.records.filter(r=>r.date===date&&r.known_at<end).sort((a,b)=>a.known_at.localeCompare(b.known_at)).at(-1);
  }
  function metric(rows){
    const good=rows.filter(r=>['ok','available'].includes(r.status)&&finite(r.forecast)&&finite(r.actual));
    const e=good.map(r=>r.forecast-r.actual),n=e.length;
    return {n_planned:rows.length,n_valid:n,mae:n?e.reduce((a,v)=>a+Math.abs(v),0)/n:null,rmse:n?Math.sqrt(e.reduce((a,v)=>a+v*v,0)/n):null,bias:n?e.reduce((a,v)=>a+v,0)/n:null,hit_within_0_5:n?e.filter(v=>Math.abs(v)<=.5).length/n:null};
  }
  function grouped(rows,calendar){
    const out={};
    for(const h of [1,2,12]){
      const selected=rows.filter(r=>r.horizon===h),groups={all:selected,tariff:[],after_tariff:[],other:[],unknown:[]};
      for(const row of selected){
        const records=[0,1,2,3].map(l=>visible(calendar,monthShift(row.target_date,-l),row.cutoff));
        const key=!records[0]?'unknown':Math.abs(records[0].rate)>1e-12?'tariff':records.some(r=>!r)?'unknown':records.slice(1).some(r=>Math.abs(r.rate)>1e-12)?'after_tariff':'other';
        groups[key].push(row);
      }
      out[h]=Object.fromEntries(Object.entries(groups).map(([key,value])=>[key,metric(value)]));
    }
    return out;
  }
  function expertRows(rows,calendar){return rows.map(row=>{
    if(!['ok','available'].includes(row.status))return {...row};
    const event=visible(calendar,row.target_date,row.cutoff);
    if(!event||!finite(event.weight)||!finite(event.baseline))return {...row,forecast:null,error:null,status:'unavailable',reason:'Нет известных на отсечение ставки, веса или учтённого базой роста для экспертной поправки.'};
    const delta=event.weight*(event.rate-event.baseline)/100,forecast=row.forecast+delta;
    if(!finite(forecast)||forecast<=-100)return {...row,forecast:null,error:null,status:'unavailable',reason:'Невозможный индекс после экспертной поправки.'};
    return {...row,forecast,error:forecast-row.actual,tariff_delta:delta,status:'ok'};
  });}
  async function compare(request,compute,onProgress=()=>{}){
    if(!request.backtests)throw new Error('Выберите режим бэктеста для сравнения тарифных вариантов.');
    if(request.frequency!=='raw')throw new Error('Сравнение тарифных вариантов доступно только для RAW.');
    const calendar=root.TariffCalendar.validate(request.tariff_calendar);
    if(calendar.region_code!==request.region_code)throw new Error('Календарь относится к другому региону.');
    const base=structuredClone(request);base.config.tariff_features='off';delete base.tariff_calendar;
    const treatment=structuredClone(request);treatment.tariff_calendar=calendar;treatment.config.tariff_features=request.config.tariff_features==='lags3'?'lags3':'current';
    async function run(req){const raw=await compute(JSON.stringify(req));return typeof raw==='string'?JSON.parse(raw):raw;}
    onProgress({message:'Базовая модель: одинаковые отсечения h=1, h=2, h=12…'});
    const baseline=await run(base);if(!baseline.ok)throw new Error(JSON.stringify(baseline.error));
    onProgress({message:'Модель с календарём: на каждой отсечке своя доступная версия тарифов…'});
    let learned=await run(treatment);
    if(!learned.ok){const reason=JSON.stringify(learned.error);learned={ok:false,error:learned.error,backtest:{rows:baseline.backtest.rows.map(r=>({...r,forecast:null,error:null,status:'unavailable',reason}))}};}
    const expert={backtest:{rows:expertRows(baseline.backtest.rows,calendar)}};
    const comparisons={calendar:root.RegionalComparison.paired(baseline,learned),expert:root.RegionalComparison.paired(baseline,expert)};
    // Correct across both model alternatives and all three horizons, not separately per alternative.
    const tests=Object.values(comparisons).flatMap(x=>Object.values(x)).filter(x=>finite(x.p_value)).sort((a,b)=>a.p_value-b.p_value);
    let last=0;tests.forEach((x,i)=>{last=Math.max(last,Math.min(1,(tests.length-i)*x.p_value));x.p_value_holm=last;});
    const variants={baseline,calendar:learned,expert};
    return {kind:'tariff_comparison',created_at:new Date().toISOString(),requests:{baseline:base,calendar:treatment},variants,paired:comparisons,metrics:Object.fromEntries(Object.entries(variants).map(([k,v])=>[k,grouped(v.backtest.rows,calendar)])),notes:[
      'Положительная разница MAE означает меньшую ошибку варианта относительно базы на совпадающих наблюдениях. У вариантов может различаться покрытие; без сопоставимых пар победитель не определяется.',
      'При пропусках тарифной истории календарная модель пропускает соответствующие строки обучения. База может использовать больше наблюдений; разницу MAE нельзя приписывать только новым признакам.',
      'Тарифные месяцы и следующие 1–3 месяца выделены по календарю, известному на каждой отсечке; неизвестные расписания выделены отдельно.',
      'Экспертная поправка требует сохранённых на дату отсечения веса и базового предположения для целевого месяца. Современные ручные настройки не подставляются в прошлое.',
      'Тест разницы MAE приблизительный HAC с поправкой Holm на доступные сравнения (до шести). Настройки не подбираются автоматически; это исследовательская проверка, не доказательство преимущества.',
      'ИПЦ и ставки используют текущую загруженную версию истории; доступность тарифного календаря соблюдена, но это не полный real-time vintage бэктест всех источников.'
    ]};
  }
  function render(container,result){
    container.replaceChildren();container.hidden=false;
    const p=text=>{const el=document.createElement('p');el.textContent=text;container.append(el);};
    const fmt=x=>finite(x)?x.toLocaleString('ru-RU',{maximumFractionDigits:4}):'—';
    const names={baseline:'База',calendar:'Календарь в модели',expert:'Экспертная поправка'};
    p('Сравнение тарифных вариантов на одинаковых отсечениях. База — без новых тарифных признаков; остальные настройки модели сохранены.');
    function table(parent,headers,rows){const wrap=document.createElement('div');wrap.className='table-scroll';const table=document.createElement('table'),head=document.createElement('thead'),body=document.createElement('tbody');const tr=document.createElement('tr');headers.forEach(v=>{const th=document.createElement('th');th.textContent=v;tr.append(th);});head.append(tr);rows.forEach(values=>{const tr=document.createElement('tr');values.forEach(v=>{const td=document.createElement('td');td.textContent=String(v);tr.append(td);});body.append(tr);});table.append(head,body);wrap.append(table);parent.append(wrap);}
    const rows=[];
    for(const variant of ['calendar','expert'])for(const h of [1,2,12]){const m=result.paired[variant][h];rows.push([names[variant],h,`${m.n_pairs}/${m.n_planned}`,fmt(m.baseline_mae),fmt(m.treatment_mae),fmt(m.improvement_mae),m.ci95?m.ci95.map(fmt).join(' … '):'—',m.p_value_holm!=null?fmt(m.p_value_holm):'Недостаточно данных / неполные пары']);}
    table(container,['Вариант','Горизонт','Пар / план','MAE базы, п.п.','MAE варианта, п.п.','Снижение MAE, п.п.','95% интервал разницы','p с поправкой Holm'],rows);
    const detail=document.createElement('details'),summary=document.createElement('summary');summary.textContent='Ошибки в тарифные месяцы и после них';detail.append(summary);container.append(detail);
    const labels={all:'Все месяцы',tariff:'Тарифный месяц',after_tariff:'Через 1–3 месяца',other:'Остальные',unknown:'Расписание неизвестно'},groupRows=[];
    for(const [variant,byH] of Object.entries(result.metrics))for(const h of [1,2,12])for(const [key,m] of Object.entries(byH[h]))groupRows.push([names[variant],h,labels[key],`${m.n_valid}/${m.n_planned}`,fmt(m.mae),fmt(m.rmse),fmt(m.bias)]);
    table(detail,['Вариант','Горизонт','Группа','Валидных / план','MAE','RMSE','Смещение'],groupRows);
    for(const [name,v] of Object.entries(result.variants)){const missing=v.backtest.rows.filter(r=>r.status==='unavailable');if(missing.length){const reasons=[...new Set(missing.map(r=>r.reason))].slice(0,3);p(`${names[name]}: недоступно ${missing.length} прогнозов. ${reasons.join(' ')}`);}}
    result.notes.forEach(p);
  }
  root.TariffComparison={compare,visible,metric,grouped,expertRows,render};
})(globalThis);
