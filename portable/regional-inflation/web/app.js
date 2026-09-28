(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const els = {
    representation: $('representation'), region: $('regionSelect'), model: $('modelSelect'), operation: $('operation'), periodField: $('periodField'), period: $('backtestPeriod'), backtestFromField:$('backtestFromField'),backtestToField:$('backtestToField'),
    loadSource: $('loadSource'), file: $('fileInput'), reset: $('resetButton'), dataNotice: $('dataNotice'), packageMeta: $('packageMeta'), run: $('runButton'), cancel: $('cancelButton'),
    engine: $('engineState'), advanced: $('advancedConfig'), advancedToggle: $('advancedToggle'), errors: $('errorMessage'), results: $('results'), resultTitle: $('resultTitle'), warnings: $('resultWarnings'),
    metrics: $('metrics'), chart: $('chart'), chartTitle: $('chartTitle'), chartSubtitle: $('chartSubtitle'), legend: $('chartLegend'), tableHead: $('tableHead'), tableBody: $('tableBody'), tableTitle: $('tableTitle'), tableSubtitle: $('tableSubtitle'), rowCount: $('rowCount'),
    csv: $('exportCsv'), settings: $('saveSettings'), package: $('savePackage'), valueMode: $('valueMode')
  };
  const DEFAULTS = {horizon:12,train_start:'2016-01',alpha:.3,epsilon:1.35,max_iter:500,tol:.00001,use_macro:true,min_train:36,ets_weights12:'[0.9,0,0.5,0.3,0.9,0.5,0,0.5,0.9,0.9,0,0]',excluded_years:'2010, 2022',seasonal_excluded_years:'2010, 2022',tariff_month_by_year:'',cutoff:''};
  let dataset = null, activeResult = null, activeRequest = null, lastRequest = null, worker = null, workerCancel = null, valueMode = 'mom', lastModel = 'ridge', disabledBeforeBusy = null;
  let wasmReady = false;
  const sourceParam = new URLSearchParams(location.search).get('source');
  if(sourceParam)$('sourceUrlInput').value=sourceParam;
  const updateSourceLink=()=>{try{$('sourceLink').href=new URL($('sourceUrlInput').value).href;}catch{$('sourceLink').removeAttribute('href');}};
  updateSourceLink();$('sourceUrlInput').addEventListener('input',updateSourceLink);

  function status(text, kind='info') { els.dataNotice.className = `notice notice-${kind}`; els.dataNotice.textContent = text; }
  function showError(message) { els.errors.textContent = message; els.errors.hidden = false; }
  function clearError() { els.errors.hidden = true; els.errors.textContent = ''; }
  function lockControls() {
    const controls=[...document.querySelectorAll('main button,main input,main select')].filter(el=>el!==els.cancel);
    if(!disabledBeforeBusy){disabledBeforeBusy=new Map(controls.map(el=>[el,el.disabled]));for(const el of controls)el.disabled=true;}
  }
  function unlockControls() {
    if(disabledBeforeBusy)for(const [el,disabled] of disabledBeforeBusy)el.disabled=disabled;
    disabledBeforeBusy=null;els.run.disabled=!dataset||!activeRows().length||!wasmReady;
  }
  function setBusy(busy) {
    if(busy)lockControls();else unlockControls();
    els.cancel.hidden=!busy;els.cancel.disabled=false;
  }
  const nfmt = (v, digits=2) => Number.isFinite(Number(v)) ? Number(v).toLocaleString('ru-RU',{minimumFractionDigits:digits,maximumFractionDigits:digits}) : '—';
  const dateLabel = d => { if(!d)return '—'; const x=String(d).slice(0,7).split('-'); return x.length===2 ? `${x[1]}.${x[0]}` : String(d); };
  const parseYears = id => $(id).value.split(/[;,\s]+/).filter(Boolean).map(x=>Number(x)).filter(Number.isFinite);
  function parseJsonField(id, fallback) { const s=$(id).value.trim(); if(!s)return fallback; try{return JSON.parse(s)}catch{throw new Error(`Проверьте формат JSON в поле «${$(id).closest('label')?.firstChild?.textContent?.trim()||id}».`);} }
  function selectedRegion() { return (dataset?.regions || []).find(r=>String(r.code)===els.region.value); }
  function getRows(region, frequency) {
    if(!region)return [];
    if(frequency==='sa') return region.sa_rows || region.representations?.sa?.rows || (region.kind==='sa'?region.rows:null) || [];
    return region.raw_rows || region.representations?.raw?.rows || (region.kind==='sa'?[]:region.rows) || [];
  }
  function activeRows() { return getRows(selectedRegion(),els.representation.value); }
  function updateRegions(preferred) {
    const regions=dataset?.regions || [];
    els.region.innerHTML='';
    if(!regions.length){els.region.add(new Option('Нет регионов в пакете',''));els.region.disabled=true;return;}
    els.region.disabled=false;
    for(const r of regions) els.region.add(new Option(`${r.name || r.code} · ${r.code}`,String(r.code)));
    const wanted=String(preferred??'');
    const kbr=regions.find(r=>['7','07','RUS07','RU-KB'].includes(String(r.code).toUpperCase())||String(r.name).toLocaleLowerCase('ru').includes('кабардино'));
    const initial=kbr?String(kbr.code):String(regions[0].code);
    els.region.value = regions.some(r=>String(r.code)===wanted)?wanted:initial;
    updateRepresentation();
  }
  function updateRepresentation() {
    const reg=selectedRegion(), sa=getRows(reg,'sa');
    const saOpt=els.representation.querySelector('option[value="sa"]');
    saOpt.disabled=!sa.length;
    if(els.representation.value==='sa'&&!sa.length)els.representation.value='raw';
    $('representationHint').textContent=els.representation.value==='sa'?'Экспериментальный SA ряд: не соединяется с RAW; годовой темп не показывается.':'RAW используется по умолчанию; годовой темп рассчитывается только для RAW.';
    if(els.operation.value==='backtest'&&valueMode==='yoy')valueMode='mom';
    els.valueMode.hidden=els.representation.value==='sa'||els.operation.value==='backtest';
    els.valueMode.textContent=valueMode==='mom'?'Показать г/г':'Показать м/м';
    const rows=activeRows();
    els.run.disabled=!rows.length||!wasmReady;
    if(reg&&!rows.length) status(`Для региона ${reg.name || reg.code} нет рядов в представлении ${els.representation.value.toUpperCase()}.`, 'warning');
    else if(reg) {
      const range=arr=>arr.length?`${arr.length} мес. (${dateLabel(arr[0].date)}—${dateLabel(arr.at(-1).date)})`:'нет данных';
      const source=dataset?.source||{};const meta=[source.name,source.modified?`файл: ${source.modified}`:null].filter(Boolean).join(' · ');
      status(`${reg.name||reg.code}: RAW ${range(getRows(reg,'raw'))}; SA ${range(getRows(reg,'sa'))}.${meta?` Источник: ${meta}.`:''} Данные доступны офлайн.`, 'success');
    }
  }
  function validatePackage(value) {
    if(!value||!Array.isArray(value.regions)||!value.regions.length)throw new Error('Ожидается JSON-пакет с массивом regions.');
    for(const r of value.regions) {
      const both=[r.rows,r.raw_rows,r.sa_rows,r.representations?.raw?.rows,r.representations?.sa?.rows].filter(Array.isArray);
      if(!both.length)throw new Error(`В регионе ${r.name||r.code||'(без кода)'} нет массива месячных строк.`);
      for(const arr of both) for(const row of arr) if(!row.date||!Number.isFinite(Number(row.y)))throw new Error(`Некорректная строка в регионе ${r.name||r.code}: нужны date и числовой y.`);
    }
    return value;
  }
  function installDataset(value, label='Локальный JSON-пакет') {
    dataset=validatePackage(value);
    activeResult=null;lastRequest=null;els.results.hidden=true;
    const count=dataset.regions.length, src=dataset.source?.name || label;
    els.packageMeta.textContent=`${src} · ${count} регион${count===1?'':count<5?'а':'ов'}`;
    els.package.disabled=false;
    updateRegions(els.region.value);
    els.run.disabled=!activeRows().length||!wasmReady;
    if(!wasmReady) status(`Данные загружены (${count} регионов), но вычислительный модуль ещё недоступен.`, 'warning');
    return dataset;
  }
  function setEngine(text, kind='') { els.engine.className=`engine-state ${kind}`;els.engine.textContent=text; }

  // Each calculation runs in a dedicated worker. Terminating it cancels a long rolling backtest.
  async function startCompute(requestJSON) {
    if(workerCancel)workerCancel();
    const bindgen = window.__WASM_BINDGEN_SOURCE__;
    const wasmBase64 = window.__REGIONAL_WASM_BASE64__;
    if(!bindgen||!wasmBase64)throw new Error('В HTML не встроен Rust/WASM вычислительный модуль. Пересоберите автономную страницу.');
    worker = new Worker(URL.createObjectURL(new Blob([`
      const jsUrl=URL.createObjectURL(new Blob([${JSON.stringify(bindgen)}],{type:'text/javascript'}));
      importScripts(jsUrl);
      const binary=Uint8Array.from(atob(${JSON.stringify(wasmBase64)}),c=>c.charCodeAt(0));
      const ready=wasm_bindgen({module_or_path:binary});
      self.onmessage=e=>{ready.then(()=>{try{const out=wasm_bindgen.process_json(e.data);self.postMessage({ok:true,value:out})}catch(err){self.postMessage({ok:false,error:String(err)})}}).catch(err=>self.postMessage({ok:false,error:'Не удалось запустить WASM: '+err}));};
    `],{type:'text/javascript'})));
    return new Promise((resolve,reject)=>{
      const finish=()=>{const w=worker;worker=null;workerCancel=null;w?.terminate();};
      workerCancel=()=>{finish();const e=new Error('Вычисление остановлено.');e.name='AbortError';reject(e);};
      worker.onmessage=e=>{finish();e.data.ok?resolve(e.data.value):reject(new Error(e.data.error));};
      worker.onerror=e=>{finish();reject(new Error(e.message||'Ошибка вычислительного потока.'));};
      worker.postMessage(requestJSON);
    });
  }
  // Public bridge kept stable so a future WASM adapter can replace the worker internals.
  window.computeRegional = requestJSON => startCompute(typeof requestJSON==='string'?requestJSON:JSON.stringify(requestJSON));
  function getConfig() {
    const cutoff=$('cutoff').value;
    const etsWeights=parseJsonField('etsWeights12',undefined),tariffMonths=parseJsonField('tariffMonths',{});
    if(etsWeights!==undefined&&(!Array.isArray(etsWeights)||etsWeights.length!==12||etsWeights.some(v=>!Number.isFinite(Number(v)))))throw new Error('В поле «Вес ETS для горизонта 12» укажите JSON-массив из 12 чисел.');
    if(!tariffMonths||Array.isArray(tariffMonths)||typeof tariffMonths!=='object'||Object.entries(tariffMonths).some(([year,month])=>!/^\d{4}$/.test(year)||!Number.isInteger(month)||month<1||month>12))throw new Error('В поле «Месяц тарифного сдвига» укажите JSON-объект вида {"2026":10}, где месяц от 1 до 12.');
    return {
      horizon:Number($('horizon').value),train_start:$('trainStart').value?`${$('trainStart').value}-01`:undefined,cutoff:cutoff?`${cutoff}-01`:undefined,
      alpha:Number($('alpha').value),epsilon:Number($('epsilon').value),max_iter:Number($('maxIter').value),tol:Number($('tol').value),
      use_macro:$('useMacro').checked,ets_weights12:etsWeights,excluded_years:parseYears('excludedYears'),seasonal_excluded_years:parseYears('seasonalExcludedYears'),min_train:Number($('minTrain').value),
      tariff_month_by_year:tariffMonths,future_components:'hold_last',future_rates:'hold_last',allow_nonconverged:false
    };
  }
  function validateConfigInputs() {
    const invalid=[...document.querySelectorAll('#advancedConfig input[type="number"][required]')].find(el=>!el.checkValidity());
    if(invalid){els.advanced.hidden=false;els.advancedToggle.setAttribute('aria-expanded','true');els.advancedToggle.textContent='Скрыть параметры';invalid.reportValidity();invalid.focus();throw new Error(`Проверьте числовое поле «${invalid.closest('label')?.firstChild?.textContent?.trim()||invalid.id}».`);}
    const start=$('trainStart'),cutoff=$('cutoff');
    if(start.value&&cutoff.value&&cutoff.value<start.value)throw new Error('Дата отсечения должна быть не раньше начала обучения.');
    if($('backtestFrom').value&&$('backtestTo').value&&$('backtestTo').value<$('backtestFrom').value)throw new Error('Конец периода отсечек должен быть не раньше начала.');
  }
  function makeRequest() {
    const region=selectedRegion(), rows=activeRows(); if(!region||!rows.length)throw new Error('Выберите регион с данными.');
    if(rows.some(r=>String(r.date).length<10))throw new Error('Даты в пакете должны быть в формате YYYY-MM-01.');
    const request={schema_version:1,model:els.model.value,frequency:els.representation.value,rows,config:getConfig()};
    if(els.operation.value==='backtest') {
      let targets=rows.slice();
      if($('cutoff').value)targets=targets.filter(r=>r.date<=`${$('cutoff').value}-01`);
      if($('backtestFrom').value)targets=targets.filter(r=>r.date>=`${$('backtestFrom').value}-01`);
      if($('backtestTo').value)targets=targets.filter(r=>r.date<=`${$('backtestTo').value}-01`);
      const period=els.period.value==='all'?targets.length:Number(els.period.value), first=Math.max(0,targets.length-period);
      const selectedTargets=targets.slice(first).map(r=>r.date);
      if(!selectedTargets.length)throw new Error('Недостаточно данных для выбранного периода проверки.');
      request.backtests={targets:selectedTargets,horizons:[1,2,12]};
    }
    return request;
  }
  function renderMetrics(result) {
    els.metrics.innerHTML=''; const add=(label,value,note='')=>{const d=document.createElement('div');d.className='metric-card';d.innerHTML=`<span>${label}</span><strong>${value}</strong><small>${note}</small>`;els.metrics.append(d);};
    if(els.operation.value==='backtest') {
      const metrics=result.backtest?.metrics||{};
      for(const h of [1,2,12]) { const m=metrics[String(h)]; if(!m)continue; add(`MAE · h=${h}`,`${nfmt(m.mae,3)} п.п.`,`RMSE ${nfmt(m.rmse,3)} · смещение ${nfmt(m.bias,3)} п.п. · ${m.n_valid??0}/${m.n_planned??0} прогнозов`); }
      const m=metrics['1']; if(m?.hit_within_0_5!=null)add('В пределах ±0,5 п.п.',`${nfmt(100*m.hit_within_0_5,1)}%`,`h=1 · ${m.n_valid??0}/${m.n_planned??0} прогнозов`);
    } else {
      const fc=result.forecast, steps=fc?.steps||[]; add('Первый месяц',steps[0]?`${nfmt(steps[0].all)}%`:'—',steps[0]?.date||'Нет прогноза');
      add('Горизонт',`${steps.length} мес.`,`до ${steps.at(-1)?.date||'—'}`); add('Обучающая выборка',`${result.fit?.n_train??'—'}`,'наблюдений');
      add('Оценка модели',result.fit?.converged===false?'Не сошлась':'Рассчитана',result.fit?.iterations!=null?`${result.fit.iterations} итераций`:'локальный расчёт');
    }
  }
  function renderWarnings(result) {
    els.warnings.innerHTML=''; const msgs=[];
    if(Array.isArray(result.warnings))for(const warning of result.warnings)msgs.push(typeof warning==='string'?warning:JSON.stringify(warning));
    if(result.fit?.converged===false)msgs.push('Модель не сошлась. Результат показан для диагностики и требует проверки.');
    const rows=result.backtest?.rows||[], invalid=rows.filter(r=>r.status==='unavailable');
    if(invalid.length)msgs.push(`${invalid.length} наблюдений бэктеста имеют статус ошибки/пропуска; метрики рассчитаны только по валидным прогнозам.`);
    if(dataset?.source?.updated_at)msgs.push(`Дата источника: ${dataset.source.updated_at}.`);
    if(msgs.length){const d=document.createElement('div');d.className='notice notice-warning';d.textContent=msgs.join(' ');els.warnings.append(d);}
  }
  function rowsForTable(result) {
    if(els.operation.value==='backtest')return result.backtest?.rows||[];
    return (result.forecast?.steps||[]).map(x=>({date:x.date,all:x.all}));
  }
  function renderTable(result) {
    const back=els.operation.value==='backtest', rows=rowsForTable(result);
    els.tableTitle.textContent=back?'Наблюдения скользящей проверки':'Помесячный прогноз';
    els.tableSubtitle.textContent=back?'Ошибка = прогноз минус факт. Пропуски отмечены отдельно.':'Модельный общий индекс, % к предыдущему месяцу.';
    const cols=back?['Отсечение','Горизонт','Целевой месяц','Факт, %','Прогноз, %','Ошибка, п.п.','Статус']:['Месяц','Прогноз, % к пред. месяцу'];
    els.tableHead.innerHTML=`<tr>${cols.map(c=>`<th>${c}</th>`).join('')}</tr>`; els.tableBody.innerHTML='';
    const shown=back?rows.slice(-100):rows;
    for(const r of shown){const tr=document.createElement('tr'); const cells=back?[dateLabel(r.cutoff),r.horizon,dateLabel(r.target_date),nfmt(r.actual),nfmt(r.forecast),nfmt(r.error),['available','ok',undefined].includes(r.status)?'Валидный':(r.reason||r.status)]:[dateLabel(r.date),`${nfmt(r.all)}%`]; for(const c of cells){const td=document.createElement('td');td.textContent=String(c??'—');tr.append(td);}els.tableBody.append(tr);}
    els.rowCount.textContent=`${rows.length} строк${back&&rows.length>100?' · показаны последние 100':''}`;
  }
  function movingYoY(rows) {
    const result=[]; for(let i=11;i<rows.length;i++){const window=rows.slice(i-11,i+1);if(window.some(r=>!Number.isFinite(Number(r.y))))continue;const factor=window.reduce((acc,r)=>acc*Number(r.y)/100,1);result.push({date:rows[i].date,y:(factor-1)*100});} return result;
  }
  function renderChart(result) {
    const frequency=els.representation.value, history=activeRows(), back=els.operation.value==='backtest';
    if(back&&valueMode==='yoy'){valueMode='mom';els.valueMode.textContent='Показать г/г';}
    els.valueMode.hidden=frequency==='sa'||back;
    const origin=result.forecast?.origin;
    const historyForPlot=back?history:history.filter(r=>!origin||r.date<=origin);
    const allHistory=frequency==='raw'&&valueMode==='yoy'?movingYoY(historyForPlot):historyForPlot.map(r=>({date:r.date,y:Number(r.y)-100}));
    const histShown=allHistory.slice(-60);
    let forecast=[];
    if(back) {
      forecast=(result.backtest?.rows||[]).filter(r=>r.horizon===1&&Number.isFinite(Number(r.forecast))).map(r=>({date:r.target_date,y:Number(r.forecast)})).slice(-60);
    } else {
      const steps=result.forecast?.steps||[];
      forecast=steps.map(r=>({date:r.date,y:Number(r.all)}));
      if(frequency==='raw'&&valueMode==='yoy') {
        const projectedIndices=steps.map(r=>({date:r.date,y:100+Number(r.all)}));
        forecast=movingYoY(historyForPlot.slice(-11).concat(projectedIndices)).slice(-projectedIndices.length);
      }
    }
    const all=histShown.concat(forecast), values=all.map(x=>x.y).filter(Number.isFinite);
    if(!values.length){els.chart.innerHTML='<div class="empty-chart">Нет значений для отображения</div>';return;}
    const W=1000,H=270,L=57,R=12,T=12,B=28, min=Math.min(...values),max=Math.max(...values),pad=(max-min||1)*.12,lo=min-pad,hi=max+pad;
    const allDates=[...new Set(all.map(p=>p.date))].sort();
    const dateIndex=new Map(allDates.map((date,i)=>[date,i]));
    const x=date=>L+(W-L-R)*(dateIndex.get(date)/Math.max(1,allDates.length-1)), y=v=>T+(H-T-B)*(1-(v-lo)/(hi-lo));
    const grid=Array.from({length:5},(_,i)=>{const v=hi-(hi-lo)*i/4, yy=y(v);return `<line class="gridline" x1="${L}" x2="${W-R}" y1="${yy}" y2="${yy}"/><text x="${L-8}" y="${yy+3}" text-anchor="end">${nfmt(v,1)}%</text>`;}).join('');
    const path=arr=>arr.map((p,i)=>`${i?'L':'M'}${x(p.date)},${y(p.y)}`).join(' ');
    const historyPath=path(histShown);
    const forecastLine=back?forecast:(forecast.length&&histShown.length?[histShown.at(-1),...forecast]:forecast);
    const forecastPath=forecast.length?path(forecastLine):'';
    const labels=[0,Math.floor((allDates.length-1)/2),allDates.length-1].filter((v,i,a)=>a.indexOf(v)===i&&v>=0).map(i=>`<text x="${x(allDates[i])}" y="${H-6}" text-anchor="${i===0?'start':i===allDates.length-1?'end':'middle'}">${dateLabel(allDates[i])}</text>`).join('');
    els.chart.dataset.points=JSON.stringify({history:histShown,forecast});
    const points=p=>escapeAttribute(JSON.stringify(p));
    els.chart.innerHTML=`<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">${grid}<line class="axis" x1="${L}" x2="${W-R}" y1="${H-B}" y2="${H-B}"/><path class="series" data-series="history" data-points="${points(histShown)}" stroke="#1978bd" d="${historyPath}"/>${forecastPath?`<path class="series" data-series="forecast" data-points="${points(forecast)}" stroke="#15917d" stroke-dasharray="6 4" d="${forecastPath}"/>`:''}${labels}</svg>`;
    els.chartTitle.textContent=back?'Скользящие прогнозы h=1 и факт':'Фактическая история и прогноз';
    els.chartSubtitle.textContent=frequency==='sa'?'SA · месячный темп, %':(back?'RAW · месячный темп, % к предыдущему месяцу':'RAW · '+(valueMode==='yoy'?'годовой темп, % г/г':'месячный темп, % к предыдущему месяцу'));
    els.legend.innerHTML=`<span><i style="background:#1978bd"></i>Факт</span>${forecast.length?'<span><i style="background:#15917d"></i>Прогноз</span>':''}`;
  }
  const escapeAttribute=value=>String(value).replaceAll('&','&amp;').replaceAll('"','&quot;').replaceAll('<','&lt;').replaceAll('>','&gt;');
  function render(result) {
    activeResult=result; els.results.hidden=false; els.resultTitle.textContent=`${selectedRegion()?.name||selectedRegion()?.code} · ${els.model.value==='ridge'?'Ridge':'Huber'} · ${els.representation.value.toUpperCase()}`;
    renderWarnings(result);renderMetrics(result);renderChart(result);renderTable(result);
  }
  function csvDownload() {
    if(!activeResult)return; const back=els.operation.value==='backtest'; const rows=rowsForTable(activeResult);
    const columns=back?['cutoff','horizon','target_date','actual','forecast','error','status','reason']:['date','all'];
    const escape=x=>`"${String(x??'').replaceAll('"','""')}"`;
    const csv='\ufeff'+[columns.join(';'),...rows.map(r=>columns.map(k=>escape(r[k])).join(';'))].join('\r\n');
    download(csv,`${safeName(selectedRegion()?.code)}_${els.model.value}_${back?'backtest':'forecast'}.csv`,'text/csv;charset=utf-8');
  }
  function download(content,name,type) { const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([content],{type}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000); }
  const safeName=x=>String(x||'regional').replace(/[^\p{L}\p{N}_-]+/gu,'_');
  function settingsSnapshot() { return {schema_version:1,model:els.model.value,frequency:els.representation.value,operation:els.operation.value,backtest_period:els.period.value,backtest_from: $('backtestFrom').value,backtest_to:$('backtestTo').value,source_url:$('sourceUrlInput').value,config:getConfig()}; }
  function saveSettings() { try{clearError();validateConfigInputs();const snapshot=settingsSnapshot();localStorage.setItem('regional-inflation-settings',JSON.stringify(snapshot));download(JSON.stringify(snapshot,null,2),`${safeName(selectedRegion()?.code)}_settings.json`,'application/json');}catch(e){showError(e.message||String(e));} }
  function applySettings(settings) {
    if(!settings||typeof settings!=='object'||!settings.config||typeof settings.config!=='object')throw new Error('Файл настроек должен содержать объект config.');
    if(settings.model&&!['ridge','huber'].includes(settings.model))throw new Error('В настройках указана неизвестная модель.');
    const map={train_start:'trainStart',cutoff:'cutoff',max_iter:'maxIter',use_macro:'useMacro',min_train:'minTrain',ets_weights12:'etsWeights12',excluded_years:'excludedYears',seasonal_excluded_years:'seasonalExcludedYears',tariff_month_by_year:'tariffMonths'};
    for(const [key,value] of Object.entries(settings.config)) {const el=$(map[key]||key);if(!el)continue;if(el.type==='checkbox')el.checked=!!value;else if(key==='excluded_years'||key==='seasonal_excluded_years')el.value=Array.isArray(value)?value.join(', '):value;else if(key==='cutoff'||key==='train_start')el.value=String(value).slice(0,7);else el.value=Array.isArray(value)||typeof value==='object'?JSON.stringify(value):value;}
    if(settings.model){els.model.value=settings.model;lastModel=settings.model;}
    if(['raw','sa'].includes(settings.frequency))els.representation.value=settings.frequency;
    if(['forecast','backtest'].includes(settings.operation))els.operation.value=settings.operation;
    if(['36','60','all'].includes(String(settings.backtest_period)))els.period.value=String(settings.backtest_period);
    if(settings.backtest_from)$('backtestFrom').value=String(settings.backtest_from).slice(0,7);
    if(settings.backtest_to)$('backtestTo').value=String(settings.backtest_to).slice(0,7);
    if(typeof settings.source_url==='string')$('sourceUrlInput').value=settings.source_url;
    updateSourceLink();const show=els.operation.value==='backtest';els.periodField.hidden=!show;els.backtestFromField.hidden=!show;els.backtestToField.hidden=!show;
    activeResult=null;lastRequest=null;els.results.hidden=true;updateRepresentation();
  }
  async function readFile(file) {
    clearError();
    lockControls();
    try {
      if(/\.xlsx$/i.test(file.name)) {
        if(typeof window.importWorkbook!=='function')throw new Error('XLSX-парсер не встроен в эту сборку.');
        setEngine('Разбор XLSX…');status(`Чтение ${file.name}. Книга будет обработана в браузере.`, 'info');
        const result=await window.importWorkbook(new Uint8Array(await file.arrayBuffer()),{previousDataset:dataset,source:{name:file.name}},p=>{if(p?.message)status(p.message,'info');});
        installDataset(result,file.name);
      } else {const parsed=JSON.parse(await file.text());if(parsed?.config&&!parsed?.regions){applySettings(parsed);status(`Настройки из ${file.name} применены.`, 'success');}else installDataset(parsed,file.name);}
    }catch(e){showError(e.message||String(e));}
    finally{unlockControls();setEngine(wasmReady?'Модуль готов · локальный расчёт':'Модуль недоступен',wasmReady?'ready':'offline');}
  }
  async function loadSourceData() {
    clearError();
    if(typeof window.loadYandexDataset!=='function'){showError('Модуль загрузки XLSX не встроен в страницу. Откройте JSON-пакет или пересоберите автономный HTML с vendor-зависимостями.');return;}
    lockControls();setEngine('Чтение исходных файлов…');status('Загрузка таблиц из открытой папки источника. Объём XLSX большой, обработка займёт некоторое время.', 'info');
    try {
      const sourceUrl=$('sourceUrlInput').value.trim();if(!sourceUrl)throw new Error('Укажите ссылку на папку источника.');
      const loaded=await window.loadYandexDataset(sourceUrl,{previousDataset:dataset},(p)=>{if(p?.message)status(p.message,'info');});
      installDataset(loaded,'Данные источника');
    } catch(e) { showError(`Не удалось прочитать источник: ${e.message||e}. Проверьте подключение и формат опубликованных файлов; либо откройте сохранённый JSON-пакет.`);status('Источник не загружен. Можно открыть локальный JSON-пакет.', 'warning'); }
    finally { unlockControls();setEngine(wasmReady?'Модуль готов · локальный расчёт':'Модуль недоступен',wasmReady?'ready':'offline'); }
  }
  async function calculate() {
    clearError(); let req; try {validateConfigInputs();req=makeRequest();}catch(e){showError(e.message);return;}
    activeResult=null;els.results.hidden=true;activeRequest=true;lastRequest=req;setBusy(true);setEngine('Расчёт выполняется в отдельном потоке…');
    try {
      const raw=await window.computeRegional(JSON.stringify(req)); const result=typeof raw==='string'?JSON.parse(raw):raw;
      if(!result?.ok)throw new Error(result?.error?.message||'Rust/WASM вернул ошибку без описания.');
      render(result);
      const latest=activeRows().at(-1)?.date; status(`Расчёт завершён. Последнее наблюдение: ${dateLabel(latest)}.`, 'success');
    } catch(e) { if(e.name!=='AbortError'){const message=e.message||String(e);showError(/nonconverged/i.test(message)?`${message}. Выберите Ridge или измените допуск сходимости в параметрах; это меняет критерий принятия оценки. Увеличение числа итераций может не помочь.`:message);} }
    finally {activeRequest=null;setBusy(false);setEngine(wasmReady?'Модуль готов · локальный расчёт':'Модуль недоступен',wasmReady?'ready':'offline');}
  }
  function savePackage() { if(dataset)download(JSON.stringify(dataset),`${safeName(selectedRegion()?.code||'regional')}_data.json`,'application/json'); }
  function saveReport() {
    if(!activeResult||!lastRequest)return;
    const region=selectedRegion();
    const report={schema_version:1,created_at:new Date().toISOString(),source:dataset?.source||null,region:{code:region?.code,name:region?.name},model:els.model.value,frequency:els.representation.value,units:{input:'monthly index levels, approximately 100',forecast:'monthly percent change from the previous month'},request:lastRequest,response:activeResult};
    download(JSON.stringify(report),`${safeName(region?.code)}_${els.model.value}_calculation.json`,'application/json');
  }
  function resetSettings() {
    for(const [key,value] of Object.entries(DEFAULTS)) { const id={train_start:'trainStart',max_iter:'maxIter',use_macro:'useMacro',min_train:'minTrain',ets_weights12:'etsWeights12',excluded_years:'excludedYears',seasonal_excluded_years:'seasonalExcludedYears',tariff_month_by_year:'tariffMonths'}[key]||key;const el=$(id);if(!el)continue;if(el.type==='checkbox')el.checked=value;else el.value=value; }
    els.model.value='ridge';lastModel='ridge';$('excludedYears').value=DEFAULTS.excluded_years;els.operation.value='forecast';els.period.value='36';$('backtestFrom').value='';$('backtestTo').value='';els.periodField.hidden=true;els.backtestFromField.hidden=true;els.backtestToField.hidden=true;els.representation.value='raw';els.results.hidden=true;activeResult=null;clearError();updateRepresentation();
  }
  els.file.addEventListener('change',e=>{const f=e.target.files?.[0];if(f)readFile(f);e.target.value='';});
  els.loadSource.addEventListener('click',loadSourceData);els.run.addEventListener('click',calculate);els.cancel.addEventListener('click',()=>{workerCancel?.();activeRequest=null;setBusy(false);setEngine('Расчёт остановлен','offline');});
  els.reset.addEventListener('click',resetSettings);els.advancedToggle.addEventListener('click',()=>{const open=els.advanced.hidden;els.advanced.hidden=!open;els.advancedToggle.setAttribute('aria-expanded',String(open));els.advancedToggle.textContent=open?'Скрыть параметры':'Показать параметры';});
  els.operation.addEventListener('change',()=>{const show=els.operation.value==='backtest';els.periodField.hidden=!show;els.backtestFromField.hidden=!show;els.backtestToField.hidden=!show;if(show)valueMode='mom';els.valueMode.hidden=show||els.representation.value==='sa';els.valueMode.textContent=valueMode==='mom'?'Показать г/г':'Показать м/м';});
  els.representation.addEventListener('change',()=>{activeResult=null;els.results.hidden=true;updateRepresentation();});els.region.addEventListener('change',()=>{activeResult=null;els.results.hidden=true;updateRepresentation();});
  els.model.addEventListener('change',()=>{const next=els.model.value;for(const [id,key] of [['excludedYears','excluded_years'],['seasonalExcludedYears','seasonal_excluded_years']]){const fallback=DEFAULTS[key],prevDefault=lastModel==='ridge'?fallback:'';if($(id).value.trim()===prevDefault)$(id).value=next==='ridge'?fallback:'';}lastModel=next;});
  const invalidateResults=()=>{if(activeRequest)return;activeResult=null;lastRequest=null;els.results.hidden=true;};
  for(const selector of ['#representation','#regionSelect','#modelSelect','#operation','#backtestPeriod','#backtestFrom','#backtestTo','#advancedConfig input'])for(const el of document.querySelectorAll(selector))for(const event of ['input','change'])el.addEventListener(event,invalidateResults);
  els.csv.addEventListener('click',csvDownload);els.settings.addEventListener('click',saveSettings);els.package.addEventListener('click',savePackage);$('saveReport').addEventListener('click',saveReport);
  els.valueMode.addEventListener('click',()=>{valueMode=valueMode==='mom'?'yoy':'mom';els.valueMode.textContent=valueMode==='mom'?'Показать г/г':'Показать м/м';if(activeResult)renderChart(activeResult);});
  try {
    const saved=JSON.parse(localStorage.getItem('regional-inflation-settings')||'null'); if(saved?.config)applySettings(saved);
  } catch {}
  const embedded = window.__REGIONAL_WASM_BASE64__ && window.__WASM_BINDGEN_SOURCE__;
  wasmReady=!!embedded;
  setEngine(wasmReady?'Модуль готов · локальный расчёт':'Модуль не встроен',wasmReady?'ready':'offline');
  els.run.disabled=true;
  if(!embedded)status('Автономная страница не содержит вычислительный модуль. Сборка должна встроить Rust/WASM в HTML.', 'warning');
  if(els.operation.value==='backtest'){els.periodField.hidden=false;els.backtestFromField.hidden=false;els.backtestToField.hidden=false;}
  if(window.__REGIONAL_DATA__) {
    try { installDataset(window.__REGIONAL_DATA__,'Встроенный пакет данных'); }
    catch(e) { showError(`Встроенный пакет данных не прошёл проверку: ${e.message||e}`); }
  }
})();
