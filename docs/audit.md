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
- A best-effort resolved audit of the local environment completed and reported known issues in audit/runtime tooling packages:
  - `filelock 3.19.1`: `GHSA-w853-jp5j-5j7f`, `GHSA-qmgc-5h2g-mvrw`
  - `pip 21.2.4`: `PYSEC-2023-228`, `GHSA-4xh5-x5gv-qwph`, `GHSA-6vgw-5pg2-w6jp`
  - `pygments 2.19.2`: `GHSA-5239-wwwm-4pmq`
  - `requests 2.32.5`: `GHSA-gc5v-m9x4-r6x2`
  - `setuptools 58.0.4`: `PYSEC-2022-43012`, `PYSEC-2025-49`, `GHSA-cx63-2mw6-8hw5`
- These findings came from the audit environment, not from a pinned application lockfile in the reference repo.
- The reference repo itself does not provide enough version control to claim a stable “no known CVEs” result for its runtime dependency set.

## Confirmed Reference Problems

### Packaging and operational gaps

1. Unpinned dependency and no lockfile:
   The project cannot be installed reproducibly or audited deterministically.
2. No tests:
   There is no regression coverage for parsing, serialization, routing, or answer generation.
3. No CI or lint/type gates:
   Nothing prevents API drift or broken releases.
4. No configuration layer:
   Models, thresholds, and output paths are hard-coded in source.
5. No retry, timeout, or rate-limit handling:
   Every OpenAI call is a direct single attempt.

### Code-level defects

1. Weak JSON validation in `.reference/pageindex-rag/pageindex/parser.py`:
   `json.loads(response.choices[0].message.content)` assumes valid JSON and the expected schema.
2. Inconsistent OpenAI parameter usage:
   `main.py` and `parser.py` use `max_completion_tokens`, while `indexer.py` uses `max_tokens`.
3. Brittle retrieval fallback in `.reference/pageindex-rag/pageindex/retriever.py`:
   Invalid router output silently falls back to `node.children[0]`, which can return unrelated context.
4. Leaf-only retrieval:
   `retrieve()` returns only the selected leaf content and drops nearby sibling context that could disambiguate the answer.
5. Fixed depth assumptions:
   The parser only performs one subdivision pass and produces a maximum depth of two levels.
6. No caching:
   Re-indexing repeats the same segmentation and summarization calls for unchanged content.
7. No inspection or provenance metadata:
   Saved indexes contain no document hash, model settings, or build metadata.
8. Unsafe file handling:
   `open(doc_path).read()` and raw `open(path, "w")` calls omit explicit encodings and error handling.

## Comment-Driven Product Gaps

1. Nearby sibling context should be attachable to the retrieved leaf.
2. The project should document that it is embedding-free in retrieval strategy, not LLM-free or cost-free.
3. A Jira-style example should demonstrate when tree routing helps with structured operational documents.
4. TypeScript support is a roadmap concern, not current scope.

## Rebuild Targets

- Fix the confirmed defects behind tests first.
- Replace silent routing fallbacks with explicit failures.
- Add cache-aware indexing, configurable retrieval, and a real CLI.
- Ship documentation that is accurate about scale, latency, and cost tradeoffs.
