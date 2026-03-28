# TreeRAG

TreeRAG is an embedding-free hierarchical retrieval system for single-document and runbook-style knowledge bases. It rebuilds the public `pageindex-rag` prototype as a typed Python package and CLI with reproducible tests, explicit failure modes, sibling-context retrieval, and cache-aware indexing.

## Why This Exists

The reference prototype had a strong idea and a fragile implementation:

- one unpinned dependency and no lockfile
- no tests or CI
- silent fallback to the first child on routing errors
- leaf-only retrieval with no sibling context
- no cache, no metadata, and no operational guardrails

TreeRAG keeps the core retrieval strategy and fixes those issues in a fresh codebase.

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
- CLI commands: `treerag index`, `treerag ask`, `treerag inspect`
- Recursive parsing beyond depth two
- File-backed caches for segmentation and summaries
- Explicit routing errors instead of silent branch fallback
- Context assembly that can include nearby sibling leaves and ancestor summaries
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

## What Changed From The Prototype

- Retrieval now includes nearby sibling context instead of only the selected leaf.
- Invalid routing choices raise explicit errors instead of defaulting to child `0`.
- Recursive parsing can build deeper trees when the document warrants it.
- Index builds are cache-aware, so unchanged content does not force repeated LLM work.
- Storage includes metadata like source hash and build timestamp.
- Tests, lint, type checks, and compile checks gate the implementation.

## Jira Docs: Does This Make Sense?

Yes, when the document is structured and the answer usually lives in one operational subsection. The Jira runbook example is a good fit because:

- headings naturally map to sections
- neighboring subsections provide useful context
- the document is small enough that hierarchical routing is cheaper than loading everything every time

It is a worse fit for:

- tiny documents where full-context prompting is simpler
- very large multi-document corpora without additional indexing layers
- workloads where deterministic keyword or structured search already solves the task

## Comment Responses

- Nearby sibling chunks: implemented through configurable `sibling_window`.
- Jira docs: covered with a dedicated runbook example and integration test.
- "No embeddings" concern: documented precisely as embedding-free retrieval, not LLM-free processing.
- TypeScript request: intentionally deferred; this repo ships Python-first and leaves TS as a roadmap item.

## Verification

Current local checks:

- `pytest tests -q`
- `ruff check src tests`
- `mypy src tests`
- `python -m compileall src tests`

## Provenance

This project was inspired by the public MIT-licensed repository `vixhal-baraiya/pageindex-rag`. The upstream codebase is preserved locally only as a read-only reference under `.reference/pageindex-rag`. Attribution and license text are in [`THIRD_PARTY_NOTICES.md`](/Users/owlxshri/Desktop/TreeRAG/THIRD_PARTY_NOTICES.md).
