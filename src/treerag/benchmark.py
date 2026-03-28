"""Benchmark and evaluation helpers for TreeRAG."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from treerag.api import build_index, query_index
from treerag.config import IndexConfig, ModelConfig, RetrievalConfig
from treerag.corpus import build_corpus, query_corpus
from treerag.errors import ParseError
from treerag.provider import LLMProvider


@dataclass(frozen=True)
class BenchmarkCase:
    """A single benchmark question and its expected signals."""

    name: str
    question: str
    expected_document_title: str | None = None
    expected_leaf_title: str | None = None
    expected_answer_substring: str | None = None


@dataclass(frozen=True)
class BenchmarkCaseResult:
    """Benchmark output for one query."""

    name: str
    question: str
    document_title: str | None
    selected_leaf_title: str
    answer: str
    query_duration_ms: float
    document_match: bool
    leaf_match: bool
    answer_match: bool

    @property
    def passed(self) -> bool:
        return self.document_match and self.leaf_match and self.answer_match

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "question": self.question,
            "document_title": self.document_title,
            "selected_leaf_title": self.selected_leaf_title,
            "answer": self.answer,
            "query_duration_ms": round(self.query_duration_ms, 3),
            "document_match": self.document_match,
            "leaf_match": self.leaf_match,
            "answer_match": self.answer_match,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class BenchmarkReport:
    """Aggregate benchmark report."""

    source_path: str
    index_path: str
    build_duration_ms: float
    total_query_duration_ms: float
    total_duration_ms: float
    case_results: list[BenchmarkCaseResult]

    @property
    def case_count(self) -> int:
        return len(self.case_results)

    @property
    def passed_count(self) -> int:
        return sum(1 for result in self.case_results if result.passed)

    @property
    def failed_count(self) -> int:
        return self.case_count - self.passed_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "index_path": self.index_path,
            "build_duration_ms": round(self.build_duration_ms, 3),
            "total_query_duration_ms": round(self.total_query_duration_ms, 3),
            "total_duration_ms": round(self.total_duration_ms, 3),
            "case_count": self.case_count,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "cases": [result.to_dict() for result in self.case_results],
        }


def load_benchmark_cases(path: str | Path) -> list[BenchmarkCase]:
    """Load benchmark cases from a JSON file."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ParseError(f"Benchmark cases file does not exist: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ParseError(f"Benchmark cases file is not valid JSON: {source}") from exc

    if not isinstance(payload, dict):
        raise ParseError("Benchmark cases payload must be a JSON object.")

    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        raise ParseError('Benchmark cases payload must include a list under "cases".')

    cases: list[BenchmarkCase] = []
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise ParseError(f"Benchmark case #{index + 1} must be a JSON object.")
        name = raw_case.get("name")
        question = raw_case.get("question")
        expected_document_title = raw_case.get("expected_document_title")
        expected_leaf_title = raw_case.get("expected_leaf_title")
        expected_answer_substring = raw_case.get("expected_answer_substring")
        if not isinstance(name, str) or not name.strip():
            raise ParseError(f"Benchmark case #{index + 1} is missing a non-empty name.")
        if not isinstance(question, str) or not question.strip():
            raise ParseError(f"Benchmark case #{index + 1} is missing a non-empty question.")
        if expected_document_title is not None and not isinstance(
            expected_document_title, str
        ):
            raise ParseError(
                f"Benchmark case #{index + 1} has a non-string expected_document_title."
            )
        if expected_leaf_title is not None and not isinstance(expected_leaf_title, str):
            raise ParseError(
                f"Benchmark case #{index + 1} has a non-string expected_leaf_title."
            )
        if expected_answer_substring is not None and not isinstance(
            expected_answer_substring, str
        ):
            raise ParseError(
                f"Benchmark case #{index + 1} has a non-string expected_answer_substring."
            )
        cases.append(
            BenchmarkCase(
                name=name.strip(),
                question=question.strip(),
                expected_document_title=expected_document_title,
                expected_leaf_title=expected_leaf_title,
                expected_answer_substring=expected_answer_substring,
            )
        )
    return cases


def run_benchmark(
    input_path: str | Path,
    cases_path: str | Path,
    index_path: str | Path,
    index_config: IndexConfig,
    retrieval_config: RetrievalConfig,
    *,
    model_config: ModelConfig | None = None,
    provider: LLMProvider | None = None,
) -> BenchmarkReport:
    """Run an end-to-end TreeRAG benchmark over a document and case file."""

    start_time = perf_counter()
    cases = load_benchmark_cases(cases_path)

    build_start = perf_counter()
    build_index(
        input_path,
        index_path,
        index_config,
        model_config=model_config,
        provider=provider,
    )
    build_duration_ms = (perf_counter() - build_start) * 1000

    case_results: list[BenchmarkCaseResult] = []
    total_query_duration_ms = 0.0
    for case in cases:
        query_start = perf_counter()
        result = query_index(
            case.question,
            index_path,
            retrieval_config,
            model_config=model_config,
            provider=provider,
        )
        query_duration_ms = (perf_counter() - query_start) * 1000
        total_query_duration_ms += query_duration_ms

        leaf_match = case.expected_leaf_title is None or (
            result.selected_leaf_title == case.expected_leaf_title
        )
        answer_match = case.expected_answer_substring is None or (
            case.expected_answer_substring.lower() in result.answer.lower()
        )
        case_results.append(
            BenchmarkCaseResult(
                name=case.name,
                question=case.question,
                document_title=None,
                selected_leaf_title=result.selected_leaf_title,
                answer=result.answer,
                query_duration_ms=query_duration_ms,
                document_match=True,
                leaf_match=leaf_match,
                answer_match=answer_match,
            )
        )

    total_duration_ms = (perf_counter() - start_time) * 1000
    return BenchmarkReport(
        source_path=str(input_path),
        index_path=str(index_path),
        build_duration_ms=build_duration_ms,
        total_query_duration_ms=total_query_duration_ms,
        total_duration_ms=total_duration_ms,
        case_results=case_results,
    )


def run_corpus_benchmark(
    input_paths: list[str | Path],
    cases_path: str | Path,
    corpus_path: str | Path,
    index_config: IndexConfig,
    retrieval_config: RetrievalConfig,
    *,
    model_config: ModelConfig | None = None,
    provider: LLMProvider | None = None,
) -> BenchmarkReport:
    """Run an end-to-end benchmark against a multi-document corpus."""

    start_time = perf_counter()
    cases = load_benchmark_cases(cases_path)

    build_start = perf_counter()
    build_corpus(
        input_paths,
        corpus_path,
        index_config,
        model_config=model_config,
        provider=provider,
    )
    build_duration_ms = (perf_counter() - build_start) * 1000

    case_results: list[BenchmarkCaseResult] = []
    total_query_duration_ms = 0.0
    for case in cases:
        query_start = perf_counter()
        result = query_corpus(
            case.question,
            corpus_path,
            retrieval_config,
            model_config=model_config,
            provider=provider,
        )
        query_duration_ms = (perf_counter() - query_start) * 1000
        total_query_duration_ms += query_duration_ms

        document_match = case.expected_document_title is None or (
            result.document_title == case.expected_document_title
        )
        leaf_match = case.expected_leaf_title is None or (
            result.selected_leaf_title == case.expected_leaf_title
        )
        answer_match = case.expected_answer_substring is None or (
            case.expected_answer_substring.lower() in result.answer.lower()
        )
        case_results.append(
            BenchmarkCaseResult(
                name=case.name,
                question=case.question,
                document_title=result.document_title,
                selected_leaf_title=result.selected_leaf_title,
                answer=result.answer,
                query_duration_ms=query_duration_ms,
                document_match=document_match,
                leaf_match=leaf_match,
                answer_match=answer_match,
            )
        )

    total_duration_ms = (perf_counter() - start_time) * 1000
    return BenchmarkReport(
        source_path=str(corpus_path),
        index_path=str(_resolve_benchmark_target_path(corpus_path)),
        build_duration_ms=build_duration_ms,
        total_query_duration_ms=total_query_duration_ms,
        total_duration_ms=total_duration_ms,
        case_results=case_results,
    )


def _resolve_benchmark_target_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.suffix == ".json":
        return candidate
    return candidate / "corpus.json"
