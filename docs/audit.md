# Reference Audit: `pageindex-rag`

## Scope

- Reference repo: `vixhal-baraiya/pageindex-rag`
- Local clone: `.reference/pageindex-rag`
- Audit date: 2026-03-29

## Dependency Audit

### Declared dependency surface

- `requirements.txt` contains a single unpinned dependency: `openai`
- The reference repo has no lockfile, no constraints file, and no reproducible install instructions

### Commands run

```bash
.venv/bin/pip-audit -r .reference/pageindex-rag/requirements.txt
.venv/bin/pip-audit --cache-dir .cache/pip-audit --path .venv/lib/python3.9/site-packages
```

### Results

- The direct `requirements.txt` audit could not produce a deterministic application result because the dependency is unpinned and `pip-audit` had to resolve a fresh environment.
- A best-effort resolved audit of the local environment reported issues in audit/runtime tooling packages:
  - `filelock 3.19.1`: `GHSA-w853-jp5j-5j7f`, `GHSA-qmgc-5h2g-mvrw`
  - `pip 21.2.4`: `PYSEC-2023-228`, `GHSA-4xh5-x5gv-qwph`, `GHSA-6vgw-5pg2-w6jp`
  - `pygments 2.19.2`: `GHSA-5239-wwwm-4pmq`
  - `requests 2.32.5`: `GHSA-gc5v-m9x4-r6x2`
  - `setuptools 58.0.4`: `PYSEC-2022-43012`, `PYSEC-2025-49`, `GHSA-cx63-2mw6-8hw5`
- These findings came from the audit environment, not from a pinned application lockfile in the reference repo.
- The reference repo does not provide enough version control to claim a stable "no known CVEs" result for its runtime dependency set.

## Confirmed Reference Problems

| Finding | Evidence | Status In TreeRAG |
| --- | --- | --- |
| Unpinned dependency and no lockfile | `.reference/pageindex-rag/requirements.txt` contains only `openai` | Fixed: `pyproject.toml` now pins the runtime range and dev tool ranges |
| No tests | Reference repo has no `tests/` directory | Fixed: regression, unit, and CLI tests cover the rebuild |
| No CI or lint/type gates | No workflow/config files in the reference repo | Fixed locally through `pytest`, `ruff`, `mypy`, and `compileall` gates |
| No configuration layer | Models, thresholds, and paths are hard-coded in source | Fixed: `ModelConfig`, `IndexConfig`, and `RetrievalConfig` power the API and CLI |
| No retry/timeout handling | `openai.OpenAI()` is instantiated with defaults and called directly | Fixed: `OpenAIProvider` applies explicit timeout and retry settings |
| Import-time API-key side effect | Importing `parser.py` or `retriever.py` instantiates `openai.OpenAI()` immediately | Fixed: provider creation is deferred behind `OpenAIProvider` and injectable call sites |
| Weak JSON validation | `parser.py` trusts `json.loads(response.choices[0].message.content)` | Fixed: segmentation responses are validated and malformed payloads raise `ParseError` |
| Inconsistent token parameter usage | `parser.py` uses `max_completion_tokens` while `indexer.py` uses `max_tokens` | Fixed: `OpenAIProvider` uses a single `max_completion_tokens` path |
| Silent routing fallback | `retriever.py` falls back to `node.children[0]` on bad model output | Fixed: invalid route choices raise `InvalidRouteChoiceError` |
| Leaf-only retrieval | `retrieve()` returns only the selected leaf content | Fixed: context assembly includes sibling windows and optional ancestor summaries |
| Fixed depth assumption | `parse_document()` only subdivides once | Fixed: recursive parsing continues until threshold or `max_depth` |
| No caching | Re-indexing repeats segmentation and summaries | Fixed: `FileCache` reuses segmentation and summary results |
| No metadata in stored index | Saved JSON has no source hash or build details | Fixed: stored indexes include source path, hash, timestamp, and config snapshots |
| Unsafe text I/O | Raw `open()` calls omit explicit encoding and error handling | Fixed: UTF-8 reads/writes are explicit in the rebuild |

## Comment-Driven Product Gaps

| Comment Theme | Rebuild Response |
| --- | --- |
| Nearby sibling chunks would help context | Implemented with configurable `sibling_window` |
| "Does RAG make sense for Jira docs?" | Added a Jira runbook example and integration coverage |
| "No embedding" is misleading if LLM summaries are used | README now states the system is embedding-free in retrieval strategy, not LLM-free |
| "Do we have the same in TS?" | Deferred; Python is the supported implementation in this repo |
| "Cool idea, but this is expensive and slow" | README now documents cost/latency tradeoffs honestly and adds caching to reduce repeat work |

## Verification

TreeRAG verification status at the end of the rebuild:

- `pytest tests -q`
- `ruff check src tests`
- `mypy src tests`
- `PYTHONPYCACHEPREFIX=.cache/pyc .venv/bin/python -m compileall src tests`
