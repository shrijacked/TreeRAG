# Validation Roadmap

TreeRAG now has a side-by-side comparison benchmark so we can measure whether hierarchical retrieval is actually helping on the same document and question set.

## Evidence Flow

```mermaid
flowchart LR
    A["Document Fixture"] --> B["Benchmark Cases"]
    B --> C["TreeRAG"]
    B --> D["Keyword Leaf Baseline"]
    B --> E["Full Context Baseline"]
    C --> F["Accuracy + Latency Report"]
    D --> F
    E --> F
```

## Current Proof Surface

- Accuracy on packaged single-document evals with expected leaf titles and answer substrings
- Side-by-side comparison against simpler baselines via `treerag compare`
- Appendix-heavy and noisy-document fixtures that stress low-overlap retrieval

## Next Proof Milestones

```mermaid
flowchart TD
    A["Phase 1: Single-doc comparisons"] --> B["Phase 2: Multi-doc corpus comparisons"]
    B --> C["Phase 3: Repeated-run latency and stability"]
    C --> D["Phase 4: Cost tracking per benchmark run"]
```

- Phase 1:
  compare `tree_rag`, `keyword_leaf`, and `full_context` on hard single-document cases
- Phase 2:
  add corpus-level baselines so we can measure routing quality across multiple runbooks
- Phase 3:
  run repeated samples per case and store latency spread instead of one-off timings
- Phase 4:
  capture provider token usage when available so benchmark output includes cost signals

## Current Entry Points

- CLI:
  `treerag benchmark ...`
  `treerag compare ...`
  `treerag corpus-benchmark ...`
- Fixtures:
  [`benchmarks/comparison_cases.json`](/Users/owlxshri/Desktop/TreeRAG/benchmarks/comparison_cases.json)
  [`benchmarks/appendix_cases.json`](/Users/owlxshri/Desktop/TreeRAG/benchmarks/appendix_cases.json)
  [`benchmarks/operations_corpus_cases.json`](/Users/owlxshri/Desktop/TreeRAG/benchmarks/operations_corpus_cases.json)
