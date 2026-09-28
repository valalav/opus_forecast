import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
const context=vm.createContext({structuredClone,Date});
for(const file of ['tariff-calendar.js','comparison.js','tariff-comparison.js'])vm.runInContext(readFileSync(new URL('../web/'+file,import.meta.url),'utf8'),context);
const T=context.TariffComparison;
const calendar={schema_version:1,region_code:'7',series_id:'synthetic_test_only',records:[
 {date:'2024-10-01',rate:10,weight:5,baseline:0,known_at:'2023-12-20',source:'synthetic:test',kind:'assumption'},
 {date:'2024-10-01',rate:20,weight:5,baseline:0,known_at:'2024-10-02',source:'synthetic:test',kind:'assumption'}
]};
const row={cutoff:'2024-09-01',target_date:'2024-10-01',horizon:1,forecast:.3,actual:.7,status:'ok',error:-.4};
test('publication vintage cutoff is end of origin month',()=>{assert.equal(T.visible(calendar,'2024-10-01','2024-09-01').rate,10);assert.equal(T.visible(calendar,'2024-10-01','2024-10-01').rate,20);});
test('expert backtest uses original vintage, not revised tariff',()=>{const x=T.expertRows([row],calendar)[0];assert.equal(x.tariff_delta,.5);assert.equal(x.forecast,.8);assert.equal(row.forecast,.3);});
test('missing expert weights never default to arbitrary values',()=>{const c=structuredClone(calendar);delete c.records[0].weight;assert.equal(T.expertRows([row],c)[0].status,'unavailable');});
test('unknown month does not become zero',()=>{const x=T.expertRows([{...row,target_date:'2024-11-01'}],calendar)[0];assert.equal(x.forecast,null);assert.equal(T.grouped([{...row,target_date:'2024-11-01'}],calendar)[1].unknown.n_planned,1);});
test('same target but h12 uses earlier available information',()=>{assert.equal(T.expertRows([{...row,cutoff:'2023-10-01',horizon:12}],calendar)[0].status,'unavailable');});
test('comparison preserves settings and uses same target slots',async()=>{
 const request={region_code:'7',frequency:'raw',config:{tariff_features:'lags3',outlier_mode:'none'},tariff_calendar:calendar,backtests:{targets:['2024-10-01'],horizons:[1,2,12]}};
 const seen=[];const compute=async s=>{const req=JSON.parse(s);seen.push(req);return {ok:true,backtest:{rows:[{...row,forecast:req.config.tariff_features==='off'?.3:.6}]}};};
 const result=await T.compare(request,compute);
 assert.equal(seen.length,2);assert.equal(seen[0].config.tariff_features,'off');assert.equal(seen[0].tariff_calendar,undefined);assert.equal(seen[1].config.tariff_features,'lags3');assert.deepEqual(seen[0].backtests,seen[1].backtests);assert.equal(request.config.tariff_features,'lags3');assert.equal(result.paired.calendar[1].n_pairs,1);assert.equal(result.paired.calendar[1].p_value,null);
});
test('calendar fit failure is reported, baseline and expert retained',async()=>{
 const req={region_code:'7',frequency:'raw',config:{},tariff_calendar:calendar,backtests:{targets:['2024-10-01']}};
 const result=await T.compare(req,async s=>JSON.parse(s).config.tariff_features==='off'?{ok:true,backtest:{rows:[row]}}:{ok:false,error:{message:'missing history'}});
 assert.equal(result.variants.calendar.backtest.rows[0].status,'unavailable');assert.equal(result.metrics.expert[1].all.n_valid,1);assert.equal(result.paired.calendar[1].n_pairs,0);
});

test('partial lag coverage stays unclassified despite a known previous increase',()=>{const c=structuredClone(calendar);c.records.push({date:'2024-11-01',rate:0,known_at:'2023-12-20',source:'synthetic:test',kind:'assumption'});const groups=T.grouped([{...row,target_date:'2024-11-01'}],c)[1];assert.equal(groups.unknown.n_planned,1);assert.equal(groups.after_tariff.n_planned,0);});
