# SIRENA M1 implementation: independent adjudication

Date: 2026-09-09. Controller/reviewer: Codex GPT-6. Frozen baseline: b72e922 plus the exact captured candidate files/diff. This review covers M1 infrastructure and aggregation contracts, not model forecast quality. All three participants received request SHA256 `37ff90cd03382b220fbd728454615df4d9b65a8ccb78c9d1d38da1762a228373`; source-set SHA256 `77bbc9f2056e315b0def98d3ddf9cbc5edbaab889f1e1ef6347e12ce9c43a93c`. Original inputs and model stdout are unchanged.

## Transport and comparison

| Participant | Confirmed route/backend | Latency | Quality | Verdict |
|---|---|---:|---:|---|
| Claude Opus | native OmniRoute cc/claude-opus-5 → claude-opus-5 | 222.688 s | 82/100 | PARTIAL |
| Gemini | native agy Gemini 3.7 Flash (Medium) | 41.653 s | 30/100 | FAIL |
| local-vllm | native OmniRoute local-vllm; underlying model undisclosed | 50.299 s | 42/100 | FAIL |

All substantive requests completed in one attempt, no route substitution and no Pi process. Both HTTP streams ended with finish_reason=stop and [DONE]; Gemini returned SUCCESS on the exact backend and no tool steps. Paired Gemini/local starts differ by 0.059 s; all three starts fall within 0.072 s. Gemini was faster; Opus produced the most useful verified review. The failed Gemini/local quality grades do not imply infrastructure failure.

The code review exceeded Windows' argv limit; native agy used its documented stream-json stdin transport. One malformed setup envelope was rejected by the CLI parser (4.313 s); a corrected transport-only smoke returned OK (10.345 s). Both are separately preserved and excluded from substantive latency/quality. GLM/Kimi were not repeated after recorded same-day HTTP503 availability failures.

## Opus findings

| Finding | Decision and evidence |
|---|---|
| Budget exhaustion needs a distinct run/failure status | **accepted, with correction.** Frozen evaluator catches its own budget TimeoutError as a model failure and returns COMPLETED_WITH_FAILURES. Reproduced by synthetic clock in verify_frozen_claims.py. Its text reason does identify budget exhaustion, so Opus's claim that it is indistinguishable is overstated. Keep all planned rows, label unexecuted budget slots separately and prevent partial runs from admission; do not use an early raise that loses the remaining schedule. |
| Zero common sample deserves explicit status/coverage gate | **accepted.** A failed adapter yields N_common=0 and null common metrics, but generic completion status. Reproduced. N_planned/N_valid/N_common already disclose the issue, so this is operational hardening rather than hidden deletion. Preserve independently valid paired losses and enforce the registered complete-pair condition; do not delete the failed model to improve the common mask. |
| Parent histories must end at cutoff | **conditional dependency check accepted; current leakage claim not established.** Actual Micro.fit slices parent series through self.cutoff (microcomponent.py:213–225) and the basket loader validates required histories. An adapter assertion is a useful boundary check against later regressions; no current stale/future-parent observation was demonstrated. |
| Explicit output-unit contract | **accepted as boundary hardening, not an established wrong-unit result.** Existing Ridge/Huber forecast paths return MoM percentages; Micro returns a weighted MoM path. Reject Opus's adaptive median-scale heuristic: extreme but valid forecasts must not be silently removed based on the training distribution. Prefer explicit units and deterministic producer/consumer tests. |
| Contributions from rejected bundles survive without status | **accepted, demonstrated defect.** An adapter returning a wrong-length path plus contributions is marked unavailable, yet its contribution is written without status. Reproduced on frozen evaluation.py. Suppress invalid contributions or carry run_id/status/reason. |
| Verify the December price base of annual expenditure weights | **accepted as a methodological assumption to document/verify.** The helper explicitly starts the relative-price chain in January, corresponding to a preceding-December level. Imported Weight_vertical semantics are outside the supplied code. No claim that the present formula is proven wrong; record reference-period evidence and limits rather than inventing it. |

Opus correctly recognized the honest C diagnostic, complete-partition normalization, journal time classification and absence of production promotion. It did not detect the precompute load-before-baseline race. The journal cannot recover an earlier source read that the caller failed to bracket.

## Gemini findings

| Finding | Decision and evidence |
|---|---|
| Micro constructed with horizon=1 must return only one point for forecast(12) | **rejected, material fabricated dependency behavior.** Actual MicrocomponentForecaster.forecast uses the explicit horizon argument at line 306 and constructs that many steps. A minimal deterministic fixture initialized at horizon=1 returned 12 finite values when asked for 12. Its proposed automatic mask reduction would weaken the registered comparison. |
| Code/source validation occurs after chmod, causing Windows cleanup failure | **rejected as stated.** In the frozen journal validation is before chmod. Rename failure after chmod is a separate conditional Windows cleanup portability edge; current target execution is Linux and that case was not demonstrated. Moving chmod after publication is not automatically better, since post-publication failures complicate atomic success. |
| Missing annual-chain month should be named in the exception | **accepted, low-priority diagnostics improvement.** The existing helper already rejects missing calendar months correctly. This is not a forecast correctness defect. |
| Git working-tree status is captured after calculation | **accepted as a metadata timing observation, not a code-integrity defect.** Actual source hashes are captured across calculation/publication. Git status may include unrelated file changes; labeling its capture time is sufficient. The proposed comment-mutation test exercises an existing hash guard, not this Git-status issue. |
| All input manifests are captured before computation | **rejected for the frozen precompute integration.** Baseline hashes were taken after initial input loading; root independently corrected this after freeze. |

Gemini's correct recognition of mass, null reasons and target-time classification does not offset the incorrect headline Micro claim or reversed cleanup sequence. It explicitly did not run tests.

## local-vllm findings

| Finding | Decision and evidence |
|---|---|
| Evaluation and journal use different code-manifest scopes | **accepted as a maintenance/provenance observation.** evaluation.py scans Python only and uses scripts.glob; journal also includes nested scripts and packaging/lock files. **Rejected consequence:** experimental CSV outputs are not subsequently passed into the prospective journal, so the claimed experiment→journal rejection chain is invented. A shared manifest helper reduces drift. |
| Context cache leaks across origins | **rejected for current code.** A fresh context/cache is constructed inside each cutoff iteration. Hypothetical future reuse is not a present P1 defect, and object id/hash does not affect these string-key lookups. |
| Protocol parameters are not checked | **rejected for the stated M1 entry point.** The identical prompt explicitly states that the current CLI validates M1_PARAMETERS; the participant ignored this scope fact. A generic evaluator cannot verify arbitrary adapters without an explicit adapter contract. |
| Missing target dates should be listed | **accepted as a low-priority message improvement.** Missing calendar targets already abort. |
| Linear memory growth is a leak | **not_applicable.** This fixed protocol has 432 rows; no leak or material exhaustion is demonstrated. Future larger experiments can stream when a real resource limit warrants it. |
| Nonfinite target facts silently reduce valid rows | **accepted as a validation gap.** The evaluator checks target-index membership, not finite values. matched_metrics excludes NaN facts from N_valid/N_common, without a distinct fact-unavailable row reason. The current loader validates the latest 12 months, so that is not sufficient proof for all 24 targets. Explicit validation of the whole target fact vector is appropriate. |

Its conclusion also overstates what prefix truncation alone proves: internally loaded inputs still need their own cutoff tests. Full df_ridge containing future rows is not itself a leak because each train is sliced before dispatch.

## Deterministic verification and integration boundary

- Frozen evaluator fixtures reproduced rejected-bundle contributions, zero-common completion classification, budget-specific text with generic run status, retained A/B pairwise losses after another model fails, and exclusion of NaN actuals. See verify_frozen_claims.py and verification.stdout.jsonl. These are synthetic contract tests, not forecast-quality results.
- Actual Micro forecast method: horizon=1 construction plus forecast(12) returned 12 finite values, rejecting Gemini's headline claim.
- Actual Micro.fit source trace slices history at cutoff; actual Ridge/Huber forecast methods specify/return MoM percentages. External reviewers were not given these implementations, so conditional checks are distinguished from demonstrated defects.
- Independently executed current integrated focused tests: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider tests/test_forecast_journal.py tests/test_registered_evaluation.py tests/test_dated_aggregation.py` — **51 passed in 0.95 s**. This was after root began integration fixes; it does not mean the frozen external-review input contained those fixes. Output is current_integration_tests.txt.
- Root reports an initial M1 result of 432 rows, 35 origins, 6 models, h=1/2/12, no failures, 64.787 s. This benchmark did not independently recompute that result and does not use it to claim model gain or score reviewers. Numeric result adjudication remains with root.
- Root independently fixed precompute's load-before-baseline race after capture and clarified overdue-month versus actual prepublication-nowcast wording. These are not discoveries credited to the external reviews. Further fixes responding to this review remain root-owned and require their own regression evidence.

## Scoring rationale and confidence

Opus base 92 (37 correctness, 19 grounding, 14 completeness, 10 compliance, 12 usefulness), minus 5 for overclaiming budget indistinguishability and 5 for the ungrounded adaptive scale-filter suggestion = **82 PARTIAL**. Core operational findings are useful and several were reproduced.

Gemini base 55 (20/12/8/9/6), minus 15 for invented Micro behavior, 5 for reversed validation/chmod order, 5 for overstating precompute race coverage = **30 FAIL**.

local-vllm base 67 (24/15/10/9/9), minus 15 for invented experiment→journal consequence, 5 for ignoring explicit CLI validation, 5 for presenting already-isolated cache reuse as a present P1 = **42 FAIL**.

Confidence is high for direct source-order checks and reproduced synthetic failures; medium for adapter/weight-base recommendations whose full domain semantics lie outside the frozen excerpt. Reviewer conflict is explicit: this Codex worker authored the journal helper. Its favorable assessment relies on test output and concrete source traces; root independently owns integration and acceptance. No peer agreement is used as a correctness criterion.

Routing: use Opus for bounded substantial contract reviews, with independent adjudication. Gemini/local were operationally fast here but require strict filtering of fabricated dependency behavior. Native SSE and native agy stdin are verified transports for this frozen request, not proof of universal route reliability.
