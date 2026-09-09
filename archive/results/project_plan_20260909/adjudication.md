# Independent adjudication — SIRENA project plan

Date: 09.09.2026. Reviewer: Codex external-review worker; final integration and independent verification belong to the parent. Frozen document and unchanged outputs are retained. This is a plan critique, not certification of model code or empirical accuracy.

## Native transport and integrity

Same request SHA256 `8ECD0775F1A360DFC2E05DAD04F32366052A5E3ACCC76286B2E647C3B8AF02F0` for all substantive calls. local-vllm started 06:09:32.706336 UTC; initial Opus 06:09:32.718340 UTC; Gemini 06:09:32.856416 UTC. Pair start difference 0.150080 s. No Pi runtime, substitutions, gateway changes, or model editing tools. Gemini backend confirmed exact requested label. SSE local identity confirms alias local-vllm, not the hidden underlying model.

Claude smoke: claude-opus-5, 1.896460 s, stop/DONE. First critique: backend claude-opus-5, 83.249298 s, length/DONE, incomplete at max_tokens=5000. Marked INFRA. One justified retry uses exactly the same request and model with max_tokens=16000; the first output is not overwritten. Retry completed in 126.714384 s with response model claude-opus-5, stop/DONE and a full answer in `claude_retry/stdout.md`. Two substantive attempts total 209.963682 s; smoke is separate. The successful retry crossed the old approximately 122 s non-stream failure point. This establishes successful native SSE transport for this request, not proof of the earlier gateway failure's internal cause.

## Paired benchmark

| Participant | Latency | Correctness /40 | Grounding /20 | Completeness /15 | Format /10 | Usefulness /15 | Penalties | Total | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Native agy Gemini 3.7 Flash (Medium) | 25.077202 s | 33 | 16 | 15 | 8 | 13 | -25 | 60 | PARTIAL |
| Native OmniRoute local-vllm | 23.898210 s | 26 | 12 | 14 | 7 | 8 | -50 | 17 | FAIL |

Gemini penalties: -15 for invented 0.02 pp aggregator stop / 0.05 bias gate that can block a valid method; -5 for declaring an unspecified real-time SA implementation and future filter choice a demonstrated defect; -5 for extrapolating the pooled 70.26% zero share into conditional behavior for most series. local-vllm penalties: -15 for incorrect month-start release exclusion; -15 for stale-index carry-forward as the proposed source-gap policy; -15 for misreading used test dates and presenting August 2025–July 2026 as new data; -5 for numerous invented thresholds without decision justification. Penalties target distinct errors; the base dimensions assess the remaining response.

Gemini is more useful on this bounded task; local-vllm is 1.179 s faster. Neither answer is safe to apply without adjudication. Quality and latency are separate; no generic route ranking is inferred.

## Verified paired findings

| Finding | Decision | Evidence and implication |
|---|---|---|
| Make real-time SA modes and evidence labels explicit | accepted as clarification | DRAFT-1 §4 requires pre-cutoff seasonality but leaves published historical vintage versus current revised SA interpretation open. Add modes and availability qualification. The reviewer does not establish actual X-13/TRAMO use. |
| Price updating must use observed pre-origin levels and forecast future levels, including year-boundary policy | accepted as explicit test, mostly already present | P2 already says future levels are forecast and unknown next-year weights cannot be inserted. Specify tests without claiming an existing error. |
| Include direct headline / 3-block controls and isolate forecast aggregation | accepted as clarification, already substantially planned | P1 includes Ridge/Huber; P2 fixes component forecasts; P3 compares 3/45 groups. Do not require every component's weighted absolute error to beat a coarser model before aggregate evaluation. |
| Evaluate event probabilities and service-block aggregate errors | accepted | P5 already requests calibration and aggregate contribution; name suitable scoring outputs. The unconditional zero share does not prove conditional medians are zero for most items. |
| Add same-information simple weekly accounting baseline | accepted | P4 already compares same stage with current bridge and monthly control. A transparent weekly-observed / monthly-residual baseline can isolate algorithmic gain from extra information. It is a proxy, not an exact known monthly price fact. |
| Treat already examined holdout as reusable diagnostic, start prospective journal | already present / reinforce | §2, P1, P7 explicitly distinguish inspected targets from new as-of evidence. The reviewers did not discover an absent split. |
| Restrict state-space / regional panel until simpler/audited variants | already present | P3/P4 defer state-space; P6 is explicitly conditional. |
| Exclude weekly release if later than day 1 of target month | rejected | P4 admits intra-month nowcasts. Correct comparison is published_at <= origin_asof, not <= first day of target month. |
| Missing weighted source should carry forward last index and block at 2% | rejected | §2 and §4 separate observed facts and explicit group forecasts; preserve source gaps and validate required parents. Neither stale source values nor 2% is justified. |
| Thresholds 0.05 pp / 5% / 6 months / 3 months / CV .15 / VIF 10 / expert gap .3 / Ridge MAE .6 | unsupported; not adopted | P7 requires preregistered operationally meaningful thresholds, not invented universal thresholds. Error increases do not demonstrate broken input data. |
| Aggregation development stops at 0.02 pp error | rejected | No source establishes this threshold. P2 explicitly separates formula correctness, reconstruction residual and forecast gain; NO_GAIN can be a valid result. |

## Opus review

Full successful output: `claude_retry/stdout.md`; do not substitute the shorter attempt-1 text. Opus verdict REVISE is reasonable for contract and acceptance-gate clarifications. The parent independently read the full response and returned integration decisions before this adjudication was finalized.

| Opus advice | Decision | Independent check and final integration boundary |
|---|---|---|
| 1. Input provenance, evidence level per forecast, target fact vintage, new run on metric restatement | accepted with clarification | §4 already joins fact and release separately and prohibits rewriting runs, but explicit typed columns close an implementable schema gap. Distinguish forecast-estimated weights from observed values; `carried_forward`/`imputed` are auditable kinds, not blanket permission to fill source gaps. |
| 2. Existing model / refreshed protocol / new mechanism ablations | accepted in substance | P3 requires code reconnaissance; without a mechanism-specific ablation, gains can be due to updated inputs or evaluation. Do not literally add a common factor to a module already containing one: first map existing behavior, then isolate the concrete factor/pooling change. |
| 3. Preregister primary metric and guardrails, disclose other horizons | accepted with scope | P7 already requires practical thresholds before evaluation. Specify hierarchy and failure decisions. A model may be promoted for h1 only; h2/h12 deterioration is reported and prevents use there, not an automatic universal veto. |
| 4. Shadow Gate-1 / replacement Gate-2, default shadow, prospective evidence | accepted | DRAFT-1 permits auxiliary status and prospective logging but does not separate the two admission gates sharply enough. Existing examined 24 targets cannot become untouched again; new as-of evidence and a prior replacement rule are needed for a superiority claim. |
| 5. RAW metric target, explicit SA provenance and inverse transformation | accepted partially; proposed relabel rejected | RAW target and inversion are useful explicit schema rules. Future-published official SA remains observed-but-unavailable at that origin, not `estimated`. Prohibit it in A/B as-of evaluation; allow separately labeled C diagnostic without real-time claim. Historical official SA genuinely published before origin may be valid; do not ban all official vintages. Train-only SA re-estimation is a distinct estimated representation. |
| 6. Separate paired forecast gain h1/h12 from reconstruction improvement | accepted with qualification | P2 already fixes group forecasts and distinguishes reconstruction; make h1/h12 reports explicit. A formula is not accepted merely because it is called Laspeyres. Mathematical correctness, mass preservation, available weights and empirical fidelity are separate gates. NO_GAIN is not a reason to claim predictive improvement and need not block later models using a fixed validated aggregation. |
| 7. Basket version and comparable coverage | accepted partially; identical mask filter rejected | Record basket/version and 100% target weight plus native/parent coverage. Requiring identical native masks can delete difficult origins and obscure a legitimate model difference, contrary to P1 failure audit. Use common information sets and full target basket, then coverage ablations/decomposition; never silently filter origins by successful native fits. |
| 8. Report number of registered specifications / grids / stages and winner rank | accepted | §8 mentions the registry; attaching search breadth and negative runs to decision.md makes it auditable. Rank alone does not establish independent validation. |
| A. Reproducibility / leak test before candidates | accepted goal; test design corrected | Comparing a current revised dataset with its cutoff prefix does not reconstruct unknown historical vintages or estimate revision bias. Use prefix invariance and perturb-future tests, fixed environment/seed and declared numerical tolerance. Do not promise cross-platform bitwise equality. Historical revision experiments require actual matched vintages. |
| B. Actual-vs-forecast future price paths in weight propagation | accepted only as oracle diagnostic | Forecasts must use available past actuals and recursive future forecasts. Actual future levels can be a separately labeled oracle diagnostic, never an available forecast or ranking result. Lost mass is an implementation error; forecast drift alone does not prove the aggregation formula invalid. h-specific use requires its own acceptance result. |
| C. Gate-1 / Gate-2 prospective matched decision | accepted | Stronger operational formulation of advice 4. No invented minimum count or success threshold is adopted. Default auxiliary outcome permits technical completion while prospective evidence accumulates. |
| Deferrals and limits | accepted, mostly already present | P6 requires panel audit, state-space waits for simple candidates, no large covariance or automatic Ensemble weights. The service zero denominator is correctly item-month, not basket weight or data-collector quality. One inspected 12-month window does not prove persistent model inferiority. |

Opus quality: correctness 35/40, grounding 18/20, completeness 15/15, format 9/10, usefulness 14/15 = 91 before penalties; -15 for the proposed future-SA relabel / identical-coverage admission policy that could misstate availability or filter hard origins, -5 for the purported revised-versus-prefix revision test and bitwise demand. Total 71/100 PARTIAL. The contract, ablation and staged-admission ideas are useful; the advice needs independent statistical and data-contract corrections. This score measures the unedited response, not the quality of the corrected final plan.

Parent decisions in this table were confirmed through explicit coordination on 09.09.2026. The frozen DRAFT-1 and all model outputs remain unchanged. No remote project files, final plan, REGISTER, WORKLOG or global model scorecard were written by this worker.
