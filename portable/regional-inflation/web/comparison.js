/* Paired, cutoff-safe forecast comparison. No model selection or coefficient p-values. */
(function (global) {
  "use strict";
  const finite = x => typeof x === "number" && Number.isFinite(x);
  const mean = xs => xs.reduce((a,b)=>a+b,0)/xs.length;
  function monthIndex(date) {
    const m=/^(\d{4})-(\d{2})-01$/.exec(date||"");
    if(!m || +m[2]<1 || +m[2]>12) throw new Error("Некорректный целевой месяц проверки");
    return +m[1]*12 + +m[2]-1;
  }
  // Abramowitz-Stegun 7.1.26, absolute CDF error < 1.5e-7.
  function normalCdf(z) {
    const x=Math.abs(z)/Math.sqrt(2), t=1/(1+0.3275911*x);
    const erf=1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-x*x);
    return .5*(1+(z<0?-erf:erf));
  }
  function summarize(baselineRows,treatmentRows,horizon,options={}) {
    function indexed(rows) {
      const map=new Map();
      for(const r of rows.filter(r=>r.horizon===horizon)) {
        const key=r.target_date;
        if(map.has(key)) throw new Error("Повтор целевого месяца в сравнении");
        map.set(key,r);
      }
      return map;
    }
    const a=indexed(baselineRows), b=indexed(treatmentRows), keys=[...new Set([...a.keys(),...b.keys()])].sort();
    const pairs=[];
    for(const key of keys) {
      const x=a.get(key), y=b.get(key);
      if(!x||!y||!["available","ok"].includes(x.status)||!["available","ok"].includes(y.status)||!finite(x.forecast)||!finite(y.forecast)||!finite(x.actual)||!finite(y.actual)) continue;
      if(Math.abs(x.actual-y.actual)>1e-9 || x.cutoff!==y.cutoff) throw new Error("Несопоставимые факты или отсечения при сравнении");
      pairs.push({target_date:key,month:monthIndex(key),a:Math.abs(x.forecast-x.actual),b:Math.abs(y.forecast-y.actual)});
    }
    const n=pairs.length, defaultLag=Math.max(horizon-1,Math.floor(4*Math.pow(Math.max(n,1)/100,2/9)));
    const lag=options.bandwidth==null?defaultLag:Number(options.bandwidth);
    if(!Number.isInteger(lag)||lag<0||lag>60)throw new Error("Число лагов HAC должно быть целым от 0 до 60");
    const out={horizon,n_planned:keys.length,n_pairs:n,baseline_mae:n?mean(pairs.map(r=>r.a)):null,treatment_mae:n?mean(pairs.map(r=>r.b)):null,improvement_mae:null,mean_loss_difference:null,ci95:null,statistic:null,p_value:null,p_value_holm:null,bandwidth:lag,status:"insufficient_pairs",method:"Paired absolute-error difference; Bartlett Newey-West HAC; asymptotic normal approximation"};
    if(!n)return out;
    const d=pairs.map(r=>r.a-r.b), avg=mean(d);out.improvement_mae=avg;out.mean_loss_difference=avg;
    if(n!==keys.length){out.status="incomplete_pairs";return out;}
    if(pairs.some((r,i)=>i&&r.month!==pairs[i-1].month+1)){out.status="calendar_gaps";return out;}
    if(n<Math.max(24,2*(lag+1)) || lag<horizon-1){out.status="insufficient_pairs_or_lags";return out;}
    const centered=d.map(v=>v-avg);
    let longRun=mean(centered.map(v=>v*v));
    for(let k=1;k<=lag;k++) {
      let covariance=0;
      for(let i=k;i<n;i++)covariance+=centered[i]*centered[i-k];
      longRun+=2*(1-k/(lag+1))*covariance/n;
    }
    if(!(longRun>1e-20)){out.status="degenerate_variance";return out;}
    const se=Math.sqrt(longRun/n),stat=avg/se;
    out.standard_error=se;out.statistic=stat;out.ci95=[avg-1.959963984540054*se,avg+1.959963984540054*se];
    out.p_value=Math.min(1,Math.max(0,2*(1-normalCdf(Math.abs(stat)))));out.status="approximate";
    return out;
  }
  function paired(baseline,treatment,options={}) {
    const result={};
    for(const h of [1,2,12])result[h]=summarize(baseline.backtest?.rows||[],treatment.backtest?.rows||[],h,options);
    const valid=Object.values(result).filter(r=>finite(r.p_value)).sort((a,b)=>a.p_value-b.p_value);
    let previous=0;
    valid.forEach((r,i)=>{previous=Math.max(previous,Math.min(1,(valid.length-i)*r.p_value));r.p_value_holm=previous;});
    return result;
  }
  async function compare(baseRequest,compute,onProgress=()=>{},options={}) {
    if(!baseRequest.backtests)throw new Error("Для сравнения задайте исторические месяцы проверки");
    const base=JSON.parse(JSON.stringify(baseRequest)), treatment=JSON.parse(JSON.stringify(baseRequest));
    base.config.outlier_mode="none";
    treatment.config.outlier_mode=baseRequest.config.outlier_mode==="none"||!baseRequest.config.outlier_mode?"mad_winsor":baseRequest.config.outlier_mode;
    onProgress({fraction:0,message:"Проверка без обработки выбросов"});
    const decode=raw=>typeof raw==="string"?JSON.parse(raw):raw;
    const baseline=decode(await compute(JSON.stringify(base)));
    if(!baseline.ok)throw new Error(baseline.error?.message||JSON.stringify(baseline.error)||"Не выполнена базовая проверка");
    onProgress({fraction:.5,message:"Проверка с обработкой выбросов"});
    const treated=decode(await compute(JSON.stringify(treatment)));
    if(!treated.ok)throw new Error(treated.error?.message||JSON.stringify(treated.error)||"Не выполнена проверка с обработкой");
    onProgress({fraction:1,message:"Сравнение завершено"});
    return {baseline,treatment:treated,requests:{baseline:base,treatment},paired_by_horizon:paired(baseline,treated,options),notes:[
      "Положительная разница MAE означает меньшую ошибку варианта с обработкой выбросов.",
      "Сравниваются одинаковые целевые месяцы и отсечения; неполные пары показаны отдельно.",
      "p-value и 95%-ный интервал приближённые, с поправкой HAC на зависимость ошибок; p-value Holm учитывает три горизонта.",
      "Это исследовательское сравнение на выбранном окне, не независимое подтверждение параметров, подобранных по этому же окну.",
      "Проверяется разница ошибок прогнозов, не значимость отдельных коэффициентов Ridge/Huber."
    ]};
  }
  global.RegionalComparison={compare,paired,summarize};
})(typeof window!=="undefined"?window:globalThis);
