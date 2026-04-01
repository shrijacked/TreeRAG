"""TreeRAG package."""

from treerag.api import build_index, query_index
from treerag.benchmark import (
    BenchmarkCase,
    BenchmarkCaseResult,
    BenchmarkReport,
    ComparisonMethodReport,
    ComparisonReport,
    CostEstimate,
    ModelPricing,
    run_benchmark,
    run_comparison_benchmark,
    run_corpus_benchmark,
    run_corpus_comparison_benchmark,
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
from treerag.provider import (
    GeminiProvider,
    OpenAIProvider,
    TokenUsage,
    UsageSnapshot,
    create_provider,
)

__all__ = [
    "__version__",
    "BenchmarkCase",
    "BenchmarkCaseResult",
    "BenchmarkReport",
    "CostEstimate",
    "ComparisonMethodReport",
    "ComparisonReport",
    "CorpusDocument",
    "CorpusIndex",
    "CorpusQueryResult",
    "DocumentIndex",
    "IndexConfig",
    "ModelConfig",
    "ModelPricing",
    "PageNode",
    "QueryResult",
    "RetrievalConfig",
    "SourceReference",
    "SourceSpan",
    "GeminiProvider",
    "OpenAIProvider",
    "TokenUsage",
    "UsageSnapshot",
    "build_index",
    "build_corpus",
    "create_provider",
    "load_corpus",
    "query_index",
    "query_corpus",
    "run_benchmark",
    "run_comparison_benchmark",
    "run_corpus_comparison_benchmark",
    "run_corpus_benchmark",
]

__version__ = "0.2.0"
