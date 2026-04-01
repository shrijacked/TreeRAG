"""TreeRAG package."""

from treerag.api import build_index, query_index
from treerag.benchmark import (
    BenchmarkCase,
    BenchmarkCaseResult,
    BenchmarkReport,
    run_benchmark,
    run_corpus_benchmark,
)
from treerag.config import IndexConfig, ModelConfig, RetrievalConfig
from treerag.corpus import (
    CorpusDocument,
    CorpusIndex,
    CorpusQueryResult,
    build_corpus,
    load_corpus,
    query_corpus,
)
from treerag.models import DocumentIndex, PageNode, QueryResult, SourceReference, SourceSpan

__all__ = [
    "__version__",
    "BenchmarkCase",
    "BenchmarkCaseResult",
    "BenchmarkReport",
    "CorpusDocument",
    "CorpusIndex",
    "CorpusQueryResult",
    "DocumentIndex",
    "IndexConfig",
    "ModelConfig",
    "PageNode",
    "QueryResult",
    "RetrievalConfig",
    "SourceReference",
    "SourceSpan",
    "build_index",
    "build_corpus",
    "load_corpus",
    "query_index",
    "query_corpus",
    "run_benchmark",
    "run_corpus_benchmark",
]

__version__ = "0.2.0"
