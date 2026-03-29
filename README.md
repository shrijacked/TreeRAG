# TreeRAG

[![CI](https://github.com/shrijacked/TreeRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/shrijacked/TreeRAG/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/shrijacked/TreeRAG)](https://github.com/shrijacked/TreeRAG/releases)
[![Python](https://img.shields.io/badge/python-3.9%2B-3776AB.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-2ea44f.svg)](https://github.com/shrijacked/TreeRAG/blob/main/LICENSE)

TreeRAG is an embedding-free hierarchical retrieval system for runbook-style knowledge bases. It packages recursive tree building, cache-aware indexing, sibling-context retrieval, multi-document corpus routing, and a typed CLI/API into a production-ready Python project.

## What "Embedding-Free" Means Here

TreeRAG does not use embeddings or a vector database to navigate the index. It still uses LLM calls for segmentation, summarization, routing, and answer generation.

That means:

- retrieval is embedding-free
- indexing and querying are still model-dependent
- cost and latency are real tradeoffs, not hidden details

It is also not a strict keyword tree. Routing is summary-based and LLM-guided, so paraphrases and synonyms can still work. The bigger failure mode is weak summaries or overlapping sections that blur the right branch.

## Architecture

```mermaid
flowchart LR
    DOC["Markdown / text document"] --> PARSE["Recursive segmentation"]
    PARSE --> TREE["Typed document tree"]
    TREE --> SUM["Bottom-up summaries"]
    SUM --> CACHE["Content-hash cache"]
    CACHE --> INDEX["JSON index with metadata"]
    INDEX --> ROUTE["LLM-guided tree routing"]
    ROUTE --> CTX["Sibling + ancestor context assembly"]
    CTX --> ANSWER["Answer generation"]
```

## Features

- Typed public API: `build_index(...)` and `query_index(...)`
- Typed corpus API: `build_corpus(...)`, `load_corpus(...)`, and `query_corpus(...)`
- Benchmark APIs: `run_benchmark(...)` and `run_corpus_benchmark(...)`
- CLI commands: `treerag index`, `treerag ask`, `treerag repl`, `treerag inspect`, `treerag corpus-index`, `treerag corpus-ask`, `treerag corpus-repl`, `treerag corpus-inspect`, `treerag benchmark`, `treerag corpus-benchmark`
- Recursive parsing beyond depth two
- File-backed caches for segmentation and summaries
- Explicit routing errors instead of silent branch fallback
- Context assembly that can include nearby sibling leaves and ancestor summaries
- Corpus manifests that route questions across multiple indexed documents before leaf selection
- UTF-8 JSON index storage with metadata and parent-link restoration
- Jira-style runbook example in [`examples/jira_runbook.md`](/Users/owlxshri/Desktop/TreeRAG/examples/jira_runbook.md)

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e .[dev]
export OPENAI_API_KEY=your_api_key_here
```

## Quick Start

Build an index and ask one question:

```bash
treerag index examples/jira_runbook.md build/jira.index.json \
  --cache-dir .cache/treerag

treerag ask build/jira.index.json "How do Sev-1 escalations work?" \
  --sibling-window 1
```

Example response:

```json
{
  "answer": "Page the primary on-call immediately and escalate after five minutes.",
  "selected_leaf_title": "Escalation Policy",
  "navigation_path": [
    "root",
    "Incident Management",
    "Escalation Policy"
  ]
}
```

Build a routed corpus across multiple runbooks:

```bash
treerag corpus-index build/runbooks \
  examples/jira_runbook.md \
  examples/oncall_handbook.md \
  examples/access_management_runbook.md \
  --cache-dir .cache/treerag

treerag corpus-ask build/runbooks "Who coordinates responders during a Sev-1?" \
  --sibling-window 1
```

Stay in an interactive loop for follow-up questions:

```bash
treerag repl build/jira.index.json
treerag corpus-repl build/runbooks
```

## CLI Usage

Build an index:

```bash
treerag index examples/jira_runbook.md build/jira.index.json \
  --cache-dir .cache/treerag \
  --subsection-threshold 200 \
  --max-depth 4
```

Ask a question:

```bash
treerag ask build/jira.index.json "How do Sev-1 escalations work?" \
  --sibling-window 1
```

Keep the same index open for repeated questions:

```bash
treerag repl build/jira.index.json
```

Inspect metadata:

```bash
treerag inspect build/jira.index.json
```

Build a routed corpus from multiple documents:

```bash
treerag corpus-index build/runbooks \
  examples/jira_runbook.md \
  examples/oncall_handbook.md \
  --cache-dir .cache/treerag
```

Ask the corpus a question:

```bash
treerag corpus-ask build/runbooks "Who owns Sev-1 response?" \
  --sibling-window 1
```

Keep the same corpus open for repeated questions:

```bash
treerag corpus-repl build/runbooks
```

Inspect the corpus manifest:

```bash
treerag corpus-inspect build/runbooks
```

Run a benchmark suite:

```bash
treerag benchmark examples/jira_runbook.md benchmarks/jira_cases.json \
  --index-path .cache/treerag/jira-benchmark.index.json \
  --cache-dir .cache/treerag
```

Run a corpus benchmark suite:

```bash
treerag corpus-benchmark build/runbooks \
  benchmarks/operations_corpus_cases.json \
  examples/jira_runbook.md \
  examples/oncall_handbook.md \
  examples/access_management_runbook.md \
  --cache-dir .cache/treerag
```

Model names are configurable from the CLI:

```bash
treerag index examples/jira_runbook.md build/jira.index.json \
  --segmentation-model gpt-5.4 \
  --summarization-model gpt-5.4-mini

treerag ask build/jira.index.json "Who gets paged first?" \
  --routing-model gpt-5.4-mini \
  --answer-model gpt-5.4
```

## Python Usage

```python
from treerag import IndexConfig, ModelConfig, RetrievalConfig, build_index, query_index

index = build_index(
    "examples/jira_runbook.md",
    "build/jira.index.json",
    IndexConfig(cache_dir=".cache/treerag"),
    model_config=ModelConfig(),
)

result = query_index(
    "How do Sev-1 escalations work?",
    "build/jira.index.json",
    RetrievalConfig(sibling_window=1, include_ancestor_summaries=True),
    model_config=ModelConfig(),
)

print(result.answer)
print(result.context)
```

## Good Fits

Yes, when the document set is structured and the answer usually lives in one operational subsection. The Jira runbook example is a good fit because:

- headings naturally map to sections
- neighboring subsections provide useful context
- the runbooks are small enough that hierarchical routing is cheaper than loading everything every time

It is a worse fit for:

- tiny documents where full-context prompting is simpler
- very large multi-document corpora without a stronger corpus-selection layer
- workloads where deterministic keyword or structured search already solves the task

## Verification

Current local checks:

- `pytest tests -q`
- `ruff check src tests`
- `mypy src tests`
- `python -m compileall src tests`

GitHub Actions now runs the same gate set on pushes to `main`, pull requests, and manual workflow runs.

## Benchmarks

TreeRAG includes a lightweight benchmark harness for repeatable, question-based evals.

- Case files live in JSON and define expected leaf titles and answer substrings
- `treerag benchmark` measures index build time, total query time, and per-case results
- [`benchmarks/jira_cases.json`](/Users/owlxshri/Desktop/TreeRAG/benchmarks/jira_cases.json) gives the repo a concrete Jira-style benchmark target
- [`benchmarks/access_cases.json`](/Users/owlxshri/Desktop/TreeRAG/benchmarks/access_cases.json) covers access-control runbooks with approval and revocation flows
- [`benchmarks/paraphrase_cases.json`](/Users/owlxshri/Desktop/TreeRAG/benchmarks/paraphrase_cases.json) probes synonym and paraphrase-style questions against the same document structure
- [`benchmarks/runbook_corpus_cases.json`](/Users/owlxshri/Desktop/TreeRAG/benchmarks/runbook_corpus_cases.json) exercises corpus routing across multiple runbooks
- [`benchmarks/operations_corpus_cases.json`](/Users/owlxshri/Desktop/TreeRAG/benchmarks/operations_corpus_cases.json) expands corpus evals across incident, on-call, and access runbooks

## Corpus Layout

`treerag corpus-index` writes a small manifest plus one index per document:

```text
build/runbooks/
├── corpus.json
└── documents/
    ├── jira-runbook.index.json
    └── oncall-handbook.index.json
```

At query time, TreeRAG first routes into the right document summary, then performs normal tree navigation inside that document.
