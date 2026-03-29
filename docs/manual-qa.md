# Manual QA Checklist

Use this checklist from the repo root:
`/Users/owlxshri/Desktop/TreeRAG`

## Setup

- Create a virtualenv:

```bash
python3 -m venv .venv
```

- Install the project:

```bash
.venv/bin/pip install -e '.[dev]'
```

- Export your API key:

```bash
export OPENAI_API_KEY=your_key_here
```

## Single-Document Flow

- Build the sample index:

```bash
treerag index examples/jira_runbook.md build/jira.index.json \
  --cache-dir .cache/treerag
```

- Ask a direct question:

```bash
treerag ask build/jira.index.json "how do sev-1 escalations work?" \
  --sibling-window 1
```

- Expected result:
  - JSON output
  - `selected_leaf_title` is `Escalation Policy`
  - answer mentions the primary on-call and escalation timing

- Inspect the index:

```bash
treerag inspect build/jira.index.json
```

## Corpus Flow

- Build a routed corpus:

```bash
treerag corpus-index build/runbooks \
  examples/jira_runbook.md \
  examples/oncall_handbook.md \
  examples/access_management_runbook.md \
  --cache-dir .cache/treerag
```

- Ask a corpus question:

```bash
treerag corpus-ask build/runbooks \
  "who coordinates responders during a sev-1?" \
  --sibling-window 1
```

- Expected result:
  - `document_title` is `On-Call Handbook`
  - `selected_leaf_title` is `Incident Command`

- Inspect the corpus:

```bash
treerag corpus-inspect build/runbooks
```

## Paraphrase Check

- Ask a synonym-style question:

```bash
treerag ask build/jira.index.json \
  "who gets alerted first during a critical outage?" \
  --sibling-window 1
```

- Expected result:
  - still routes to `Escalation Policy`
  - answer still mentions the primary on-call

## Benchmarks

- Run the single-document benchmark:

```bash
treerag benchmark examples/jira_runbook.md benchmarks/jira_cases.json \
  --index-path .cache/treerag/jira-benchmark.index.json \
  --cache-dir .cache/treerag
```

- Run the paraphrase benchmark:

```bash
treerag benchmark examples/jira_runbook.md benchmarks/paraphrase_cases.json \
  --index-path .cache/treerag/jira-paraphrase.index.json \
  --cache-dir .cache/treerag
```

- Run the corpus benchmark:

```bash
treerag corpus-benchmark build/runbooks \
  benchmarks/operations_corpus_cases.json \
  examples/jira_runbook.md \
  examples/oncall_handbook.md \
  examples/access_management_runbook.md \
  --cache-dir .cache/treerag
```

- Expected result:
  - `passed_count` equals `case_count`

## Interactive UX

- Start the single-index REPL:

```bash
treerag repl build/jira.index.json
```

- Start the corpus REPL:

```bash
treerag corpus-repl build/runbooks
```

- Expected result:
  - answers are printed as JSON after each question
  - `quit` or `exit` cleanly leaves the loop

## Non-LLM Repo Checks

- Run the local verification gates:

```bash
pytest tests -q
ruff check src tests
mypy src tests
python -m build --no-isolation
python -m twine check dist/*
```

- Expected result:
  - all commands pass
