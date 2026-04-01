"""Benchmark and evaluation helpers for TreeRAG."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from treerag.api import build_index, query_index
from treerag.config import IndexConfig, ModelConfig, RetrievalConfig
from treerag.corpus import CorpusDocument, CorpusIndex, build_corpus, load_corpus, query_corpus
from treerag.errors import ParseError
from treerag.models import PageNode
from treerag.provider import LLMProvider
from treerag.retrieval import assemble_context

TREE_RAG_METHOD = "tree_rag"
KEYWORD_LEAF_METHOD = "keyword_leaf"
KEYWORD_DOCUMENT_METHOD = "keyword_document"
FULL_CONTEXT_METHOD = "full_context"


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
    query_samples_ms: tuple[float, ...] = ()
    document_consistent: bool = True
    leaf_consistent: bool = True
    answer_consistent: bool = True

    @property
    def passed(self) -> bool:
        return self.document_match and self.leaf_match and self.answer_match

    @property
    def query_sample_count(self) -> int:
        return len(self.query_samples_ms) or 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "question": self.question,
            "document_title": self.document_title,
            "selected_leaf_title": self.selected_leaf_title,
            "answer": self.answer,
            "query_duration_ms": round(self.query_duration_ms, 3),
            "query_samples_ms": [round(duration, 3) for duration in self.query_samples_ms],
            "query_sample_count": self.query_sample_count,
            "document_match": self.document_match,
            "leaf_match": self.leaf_match,
            "answer_match": self.answer_match,
            "document_consistent": self.document_consistent,
            "leaf_consistent": self.leaf_consistent,
            "answer_consistent": self.answer_consistent,
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


@dataclass(frozen=True)
class ComparisonMethodReport:
    """Per-method results for a comparison benchmark run."""

    method: str
    total_query_duration_ms: float
    total_runs: int
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

    @property
    def average_query_duration_ms(self) -> float:
        if self.total_runs == 0:
            return 0.0
        return self.total_query_duration_ms / self.total_runs

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "case_count": self.case_count,
            "total_runs": self.total_runs,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "total_query_duration_ms": round(self.total_query_duration_ms, 3),
            "average_query_duration_ms": round(self.average_query_duration_ms, 3),
            "cases": [result.to_dict() for result in self.case_results],
        }


@dataclass(frozen=True)
class ComparisonReport:
    """Aggregate comparison benchmark output."""

    source_path: str
    index_path: str
    build_duration_ms: float
    total_duration_ms: float
    methods: list[ComparisonMethodReport]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "index_path": self.index_path,
            "build_duration_ms": round(self.build_duration_ms, 3),
            "total_duration_ms": round(self.total_duration_ms, 3),
            "methods": [method.to_dict() for method in self.methods],
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


def run_comparison_benchmark(
    input_path: str | Path,
    cases_path: str | Path,
    index_path: str | Path,
    index_config: IndexConfig,
    retrieval_config: RetrievalConfig,
    *,
    model_config: ModelConfig | None = None,
    provider: LLMProvider | None = None,
    repeat_count: int = 1,
    methods: tuple[str, ...] = (
        TREE_RAG_METHOD,
        KEYWORD_LEAF_METHOD,
        FULL_CONTEXT_METHOD,
    ),
) -> ComparisonReport:
    """Run TreeRAG against simpler baselines on the same benchmark cases."""

    if repeat_count < 1:
        raise ParseError("Comparison benchmark repeat_count must be at least 1.")

    start_time = perf_counter()
    cases = load_benchmark_cases(cases_path)
    active_model_config = model_config or ModelConfig()
    active_provider: LLMProvider
    if provider is None:
        from treerag.provider import OpenAIProvider

        active_provider = OpenAIProvider()
    else:
        active_provider = provider

    build_start = perf_counter()
    document_index = build_index(
        input_path,
        index_path,
        index_config,
        model_config=active_model_config,
        provider=active_provider,
    )
    build_duration_ms = (perf_counter() - build_start) * 1000
    source_text = Path(input_path).read_text(encoding="utf-8")

    method_reports: list[ComparisonMethodReport] = []
    for method in methods:
        case_results: list[BenchmarkCaseResult] = []
        total_query_duration_ms = 0.0
        total_runs = 0
        for case in cases:
            sample_durations_ms: list[float] = []
            observed_leaf_titles: list[str] = []
            observed_answers: list[str] = []
            selected_leaf_title = ""
            answer = ""
            leaf_match = False
            for _ in range(repeat_count):
                query_start = perf_counter()
                selected_leaf_title, answer, leaf_match = _run_single_document_method(
                    method,
                    case,
                    source_text=source_text,
                    root=document_index.root,
                    retrieval_config=retrieval_config,
                    model_config=active_model_config,
                    provider=active_provider,
                    index_path=index_path,
                )
                query_duration_ms = (perf_counter() - query_start) * 1000
                sample_durations_ms.append(query_duration_ms)
                total_query_duration_ms += query_duration_ms
                total_runs += 1
                observed_leaf_titles.append(selected_leaf_title)
                observed_answers.append(answer)

            answer_match = case.expected_answer_substring is None or (
                case.expected_answer_substring.lower() in answer.lower()
            )
            case_results.append(
                BenchmarkCaseResult(
                    name=case.name,
                    question=case.question,
                    document_title=None,
                    selected_leaf_title=selected_leaf_title,
                    answer=answer,
                    query_duration_ms=sum(sample_durations_ms) / len(sample_durations_ms),
                    document_match=True,
                    leaf_match=leaf_match,
                    answer_match=answer_match,
                    query_samples_ms=tuple(sample_durations_ms),
                    document_consistent=True,
                    leaf_consistent=len(set(observed_leaf_titles)) == 1,
                    answer_consistent=len(set(observed_answers)) == 1,
                )
            )
        method_reports.append(
            ComparisonMethodReport(
                method=method,
                total_query_duration_ms=total_query_duration_ms,
                total_runs=total_runs,
                case_results=case_results,
            )
        )

    total_duration_ms = (perf_counter() - start_time) * 1000
    return ComparisonReport(
        source_path=str(input_path),
        index_path=str(index_path),
        build_duration_ms=build_duration_ms,
        total_duration_ms=total_duration_ms,
        methods=method_reports,
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


def run_corpus_comparison_benchmark(
    input_paths: list[str | Path],
    cases_path: str | Path,
    corpus_path: str | Path,
    index_config: IndexConfig,
    retrieval_config: RetrievalConfig,
    *,
    model_config: ModelConfig | None = None,
    provider: LLMProvider | None = None,
    repeat_count: int = 1,
    methods: tuple[str, ...] = (
        TREE_RAG_METHOD,
        KEYWORD_DOCUMENT_METHOD,
        FULL_CONTEXT_METHOD,
    ),
) -> ComparisonReport:
    """Run TreeRAG against simpler corpus-level baselines on the same cases."""

    if repeat_count < 1:
        raise ParseError("Corpus comparison benchmark repeat_count must be at least 1.")

    start_time = perf_counter()
    cases = load_benchmark_cases(cases_path)
    active_model_config = model_config or ModelConfig()
    active_provider: LLMProvider
    if provider is None:
        from treerag.provider import OpenAIProvider

        active_provider = OpenAIProvider()
    else:
        active_provider = provider

    build_start = perf_counter()
    build_corpus(
        input_paths,
        corpus_path,
        index_config,
        model_config=active_model_config,
        provider=active_provider,
    )
    build_duration_ms = (perf_counter() - build_start) * 1000

    corpus_index = load_corpus(corpus_path)
    source_texts = {
        document.document_id: Path(document.source_path).read_text(encoding="utf-8")
        for document in corpus_index.documents
    }

    method_reports: list[ComparisonMethodReport] = []
    for method in methods:
        case_results: list[BenchmarkCaseResult] = []
        total_query_duration_ms = 0.0
        total_runs = 0
        for case in cases:
            sample_durations_ms: list[float] = []
            observed_document_titles: list[str] = []
            observed_leaf_titles: list[str] = []
            observed_answers: list[str] = []
            document_title = ""
            selected_leaf_title = ""
            answer = ""
            document_match = False
            leaf_match = False
            for _ in range(repeat_count):
                query_start = perf_counter()
                document_title, selected_leaf_title, answer, document_match, leaf_match = (
                    _run_corpus_method(
                        method,
                        case,
                        corpus_index=corpus_index,
                        source_texts=source_texts,
                        corpus_path=corpus_path,
                        retrieval_config=retrieval_config,
                        model_config=active_model_config,
                        provider=active_provider,
                    )
                )
                query_duration_ms = (perf_counter() - query_start) * 1000
                sample_durations_ms.append(query_duration_ms)
                total_query_duration_ms += query_duration_ms
                total_runs += 1
                observed_document_titles.append(document_title)
                observed_leaf_titles.append(selected_leaf_title)
                observed_answers.append(answer)

            answer_match = case.expected_answer_substring is None or (
                case.expected_answer_substring.lower() in answer.lower()
            )
            case_results.append(
                BenchmarkCaseResult(
                    name=case.name,
                    question=case.question,
                    document_title=document_title,
                    selected_leaf_title=selected_leaf_title,
                    answer=answer,
                    query_duration_ms=sum(sample_durations_ms) / len(sample_durations_ms),
                    document_match=document_match,
                    leaf_match=leaf_match,
                    answer_match=answer_match,
                    query_samples_ms=tuple(sample_durations_ms),
                    document_consistent=len(set(observed_document_titles)) == 1,
                    leaf_consistent=len(set(observed_leaf_titles)) == 1,
                    answer_consistent=len(set(observed_answers)) == 1,
                )
            )
        method_reports.append(
            ComparisonMethodReport(
                method=method,
                total_query_duration_ms=total_query_duration_ms,
                total_runs=total_runs,
                case_results=case_results,
            )
        )

    total_duration_ms = (perf_counter() - start_time) * 1000
    return ComparisonReport(
        source_path=str(corpus_path),
        index_path=str(_resolve_benchmark_target_path(corpus_path)),
        build_duration_ms=build_duration_ms,
        total_duration_ms=total_duration_ms,
        methods=method_reports,
    )


def _resolve_benchmark_target_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.suffix == ".json":
        return candidate
    return candidate / "corpus.json"


def _run_single_document_method(
    method: str,
    case: BenchmarkCase,
    *,
    source_text: str,
    root: PageNode,
    retrieval_config: RetrievalConfig,
    model_config: ModelConfig,
    provider: LLMProvider,
    index_path: str | Path,
) -> tuple[str, str, bool]:
    if method == TREE_RAG_METHOD:
        result = query_index(
            case.question,
            index_path,
            retrieval_config,
            model_config=model_config,
            provider=provider,
        )
        leaf_match = case.expected_leaf_title is None or (
            result.selected_leaf_title == case.expected_leaf_title
        )
        return result.selected_leaf_title, result.answer, leaf_match

    if method == KEYWORD_LEAF_METHOD:
        leaf = _choose_keyword_leaf(case.question, root)
        context, _ = assemble_context(leaf, config=retrieval_config)
        answer = provider.answer(case.question, context=context, model_config=model_config)
        leaf_match = case.expected_leaf_title is None or leaf.title == case.expected_leaf_title
        return leaf.title, answer, leaf_match

    if method == FULL_CONTEXT_METHOD:
        answer = provider.answer(case.question, context=source_text, model_config=model_config)
        return "(full document)", answer, True

    raise ParseError(f"Unsupported comparison method: {method}")


def _run_corpus_method(
    method: str,
    case: BenchmarkCase,
    *,
    corpus_index: CorpusIndex,
    source_texts: dict[str, str],
    corpus_path: str | Path,
    retrieval_config: RetrievalConfig,
    model_config: ModelConfig,
    provider: LLMProvider,
) -> tuple[str, str, str, bool, bool]:
    if method == TREE_RAG_METHOD:
        result = query_corpus(
            case.question,
            corpus_path,
            retrieval_config,
            model_config=model_config,
            provider=provider,
        )
        document_match = case.expected_document_title is None or (
            result.document_title == case.expected_document_title
        )
        leaf_match = case.expected_leaf_title is None or (
            result.selected_leaf_title == case.expected_leaf_title
        )
        return (
            result.document_title,
            result.selected_leaf_title,
            result.answer,
            document_match,
            leaf_match,
        )

    if method == KEYWORD_DOCUMENT_METHOD:
        document = _choose_keyword_document(case.question, corpus_index)
        query_result = query_index(
            case.question,
            document.index_path,
            retrieval_config,
            model_config=model_config,
            provider=provider,
        )
        document_match = case.expected_document_title is None or (
            document.title == case.expected_document_title
        )
        leaf_match = case.expected_leaf_title is None or (
            query_result.selected_leaf_title == case.expected_leaf_title
        )
        return (
            document.title,
            query_result.selected_leaf_title,
            query_result.answer,
            document_match,
            leaf_match,
        )

    if method == FULL_CONTEXT_METHOD:
        answer = provider.answer(
            case.question,
            context=_full_corpus_context(corpus_index, source_texts),
            model_config=model_config,
        )
        return "(full corpus)", "(full corpus)", answer, True, True

    raise ParseError(f"Unsupported corpus comparison method: {method}")


def _choose_keyword_leaf(question: str, root: PageNode) -> PageNode:
    leaves = [node for node in root.walk() if node.is_leaf() and node.content.strip()]
    if not leaves:
        raise ParseError("Comparison benchmark could not find any leaf content to score.")

    question_tokens = set(_tokenize(question))
    best_leaf = leaves[0]
    best_score = -1
    for leaf in leaves:
        haystack_tokens = set(_tokenize(f"{leaf.title} {leaf.content}"))
        score = len(question_tokens & haystack_tokens)
        if score > best_score:
            best_leaf = leaf
            best_score = score
    return best_leaf


def _choose_keyword_document(question: str, corpus_index: CorpusIndex) -> CorpusDocument:
    if not corpus_index.documents:
        raise ParseError("Corpus comparison benchmark requires at least one indexed document.")

    question_tokens = set(_tokenize(question))
    best_document = corpus_index.documents[0]
    best_score = -1
    for document in corpus_index.documents:
        haystack_tokens = set(_tokenize(f"{document.title} {document.summary}"))
        score = len(question_tokens & haystack_tokens)
        if score > best_score:
            best_document = document
            best_score = score
    return best_document


def _full_corpus_context(corpus_index: CorpusIndex, source_texts: dict[str, str]) -> str:
    blocks: list[str] = []
    for document in corpus_index.documents:
        blocks.append(f"Document: {document.title}")
        blocks.append(source_texts[document.document_id].strip())
    return "\n\n".join(blocks)


def _tokenize(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9]+", text.lower())
