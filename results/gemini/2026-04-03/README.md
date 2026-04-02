# Live Gemini Results

Date: `2026-04-03`

Tracked artifact:

- [`noisy_finance_compare.json`](/Users/owlxshri/Desktop/TreeRAG/results/gemini/2026-04-03/noisy_finance_compare.json)

What this run shows:

- provider: `gemini-2.5-flash`
- fixture: [`benchmarks/comparison_cases.json`](/Users/owlxshri/Desktop/TreeRAG/benchmarks/comparison_cases.json) against [`examples/noisy_finance_report.md`](/Users/owlxshri/Desktop/TreeRAG/examples/noisy_finance_report.md)
- `tree_rag` routed into `Appendix G Debt Schedule`
- `keyword_leaf` stayed in `Executive Summary`
- `full_context` mixed the summary statement with the appendix number

Measured signals from the saved run:

- total live requests: `4`
- total duration: `12074.562 ms`
- total estimated cost: `$0.0003347`
- `tree_rag` query duration: `5318.035 ms`
- `tree_rag` estimated cost: `$0.0001544`

Important note:

- this JSON was produced before the scorer normalization pass, so its `passed` flags are stricter than the current code
- the raw output is still useful because it shows the retrieval path difference clearly: TreeRAG is the only method that drilled into the appendix leaf

Current blocker:

- additional live Gemini runs on the same key hit the free-tier daily request cap for `gemini-2.5-flash`, so this folder currently tracks the successful live artifact instead of a larger batch
