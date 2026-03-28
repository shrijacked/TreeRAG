"""TreeRAG package."""

from treerag.api import build_index, query_index
from treerag.benchmark import BenchmarkCase, BenchmarkCaseResult, BenchmarkReport, run_benchmark
from treerag.config import IndexConfig, ModelConfig, RetrievalConfig
from treerag.models import DocumentIndex, PageNode, QueryResult

__all__ = [
    "__version__",
    "BenchmarkCase",
    "BenchmarkCaseResult",
    "BenchmarkReport",
    "DocumentIndex",
    "IndexConfig",
    "ModelConfig",
    "PageNode",
    "QueryResult",
    "RetrievalConfig",
    "build_index",
    "query_index",
    "run_benchmark",
]

__version__ = "0.1.0"
