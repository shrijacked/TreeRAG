# TreeRAG

TreeRAG is a production-oriented rebuild of a public hierarchical retrieval prototype. It keeps the core idea of embedding-free tree navigation, then hardens it with typed interfaces, deterministic tests, configurable retrieval, and reusable indexing caches.

## Status

This repository is being rebuilt from scratch in this workspace. The upstream prototype is kept as a read-only reference under `.reference/pageindex-rag` and is not part of the new git history.

## Goals

- Build a real Python package and CLI instead of a single-script prototype.
- Reproduce the reference defects with tests before fixing them.
- Add sibling-context retrieval, config-driven indexing, and cache-aware LLM calls.
- Document the cost and tradeoffs of LLM-routed retrieval honestly.

## Reference And Provenance

TreeRAG is inspired by `vixhal-baraiya/pageindex-rag`, a public MIT-licensed prototype. See `THIRD_PARTY_NOTICES.md` for attribution details and the preserved upstream license text.
