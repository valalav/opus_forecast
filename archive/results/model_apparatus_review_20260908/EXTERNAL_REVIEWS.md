# External review record — 2026-09-08

- User explicitly requested a fresh Claude review.
- Requested: omniroute/cc/claude-opus-5. Smoke returned claude-opus-5 (2.587 s).
- Full neutral read-only review: 281.235 s, exit0; final proxy responseModel field is keepalive.
- Claude advice adjudicated against source and direct reproduction: 82/100, useful with material qualifications.
- Mandatory Gemini/local-vllm comparison used the same bounded neutral input and separate independent CSV verification: Gemini39/100 FAIL (36.670 s), Pi42/100 FAIL (127.295 s). Both misread some columns of a wide CSV; those claims were rejected.
- GLM5.1 and Kimi2.6 exact-route probes returned HTTP503 (15.791 s and15.788 s), no advisory output; no substitute used.
- All participant outputs remain hypotheses. Accepted calculations were independently reproduced by Codex.
- Detailed paired benchmark is stored in local cb/04_TOOLING/model_benchmarks/runs/2026-09-08_sirena_model_apparatus_review.

Repository presentation copies use LF line endings. Exact original external launch logs and prompt bytes remain in the local task evidence; content and source hashes were not altered. Initial git whitespace check detected CRLF in transported JSON/CSV; normalized only task-owned presentation files.
