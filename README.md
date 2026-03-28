# TreeRAG

TreeRAG is an embedding-free hierarchical retrieval system for runbook-style knowledge bases. It packages recursive tree building, cache-aware indexing, sibling-context retrieval, multi-document corpus routing, and a typed CLI/API into a production-ready Python project.

## What "Embedding-Free" Means Here

TreeRAG does not use embeddings or a vector database to navigate the index. It still uses LLM calls for segmentation, summarization, routing, and answer generation.

That means:

- retrieval is embedding-free
- indexing and querying are still model-dependent
- cost and latency are real tradeoffs, not hidden details

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
- CLI commands: `treerag index`, `treerag ask`, `treerag inspect`, `treerag corpus-index`, `treerag corpus-ask`, `treerag corpus-inspect`, `treerag benchmark`, `treerag corpus-benchmark`
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
  benchmarks/runbook_corpus_cases.json \
  examples/jira_runbook.md \
  examples/oncall_handbook.md \
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
- [`benchmarks/runbook_corpus_cases.json`](/Users/owlxshri/Desktop/TreeRAG/benchmarks/runbook_corpus_cases.json) exercises corpus routing across multiple runbooks

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
