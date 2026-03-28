"""Command-line interface for TreeRAG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from treerag.api import build_index, query_index
from treerag.benchmark import run_benchmark, run_corpus_benchmark
from treerag.config import IndexConfig, ModelConfig, RetrievalConfig
from treerag.corpus import build_corpus, load_corpus, query_corpus
from treerag.provider import LLMProvider
from treerag.storage import load_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="treerag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Build a TreeRAG index.")
    index_parser.add_argument("input_path")
    index_parser.add_argument("output_path")
    index_parser.add_argument("--subsection-threshold", type=int, default=300)
    index_parser.add_argument("--max-depth", type=int, default=4)
    index_parser.add_argument("--cache-dir", default=".cache/treerag")
    index_parser.add_argument("--disable-cache", action="store_true")
    index_parser.add_argument("--segmentation-model", default=ModelConfig().segmentation_model)
    index_parser.add_argument("--summarization-model", default=ModelConfig().summarization_model)

    ask_parser = subparsers.add_parser("ask", help="Answer a question using a saved index.")
    ask_parser.add_argument("index_path")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--sibling-window", type=int, default=1)
    ask_parser.add_argument("--exclude-ancestors", action="store_true")
    ask_parser.add_argument("--routing-model", default=ModelConfig().routing_model)
    ask_parser.add_argument("--answer-model", default=ModelConfig().answer_model)

    repl_parser = subparsers.add_parser(
        "repl",
        help="Start an interactive question loop for a saved index.",
    )
    repl_parser.add_argument("index_path")
    repl_parser.add_argument("--sibling-window", type=int, default=1)
    repl_parser.add_argument("--exclude-ancestors", action="store_true")
    repl_parser.add_argument("--routing-model", default=ModelConfig().routing_model)
    repl_parser.add_argument("--answer-model", default=ModelConfig().answer_model)

    inspect_parser = subparsers.add_parser("inspect", help="Print index metadata.")
    inspect_parser.add_argument("index_path")

    corpus_index_parser = subparsers.add_parser(
        "corpus-index",
        help="Build a multi-document corpus manifest and per-document indexes.",
    )
    corpus_index_parser.add_argument("output_path")
    corpus_index_parser.add_argument("input_paths", nargs="+")
    corpus_index_parser.add_argument("--subsection-threshold", type=int, default=300)
    corpus_index_parser.add_argument("--max-depth", type=int, default=4)
    corpus_index_parser.add_argument("--cache-dir", default=".cache/treerag")
    corpus_index_parser.add_argument("--disable-cache", action="store_true")
    corpus_index_parser.add_argument(
        "--segmentation-model",
        default=ModelConfig().segmentation_model,
    )
    corpus_index_parser.add_argument(
        "--summarization-model",
        default=ModelConfig().summarization_model,
    )

    corpus_ask_parser = subparsers.add_parser(
        "corpus-ask",
        help="Answer a question by routing through a corpus manifest.",
    )
    corpus_ask_parser.add_argument("corpus_path")
    corpus_ask_parser.add_argument("question")
    corpus_ask_parser.add_argument("--sibling-window", type=int, default=1)
    corpus_ask_parser.add_argument("--exclude-ancestors", action="store_true")
    corpus_ask_parser.add_argument("--routing-model", default=ModelConfig().routing_model)
    corpus_ask_parser.add_argument("--answer-model", default=ModelConfig().answer_model)

    corpus_repl_parser = subparsers.add_parser(
        "corpus-repl",
        help="Start an interactive question loop for a saved corpus manifest.",
    )
    corpus_repl_parser.add_argument("corpus_path")
    corpus_repl_parser.add_argument("--sibling-window", type=int, default=1)
    corpus_repl_parser.add_argument("--exclude-ancestors", action="store_true")
    corpus_repl_parser.add_argument("--routing-model", default=ModelConfig().routing_model)
    corpus_repl_parser.add_argument("--answer-model", default=ModelConfig().answer_model)

    corpus_inspect_parser = subparsers.add_parser(
        "corpus-inspect",
        help="Print corpus manifest metadata.",
    )
    corpus_inspect_parser.add_argument("corpus_path")

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Build an index and run benchmark cases against it.",
    )
    benchmark_parser.add_argument("input_path")
    benchmark_parser.add_argument("cases_path")
    benchmark_parser.add_argument(
        "--index-path",
        default=".cache/treerag/benchmark.index.json",
    )
    benchmark_parser.add_argument("--subsection-threshold", type=int, default=300)
    benchmark_parser.add_argument("--max-depth", type=int, default=4)
    benchmark_parser.add_argument("--cache-dir", default=".cache/treerag")
    benchmark_parser.add_argument("--disable-cache", action="store_true")
    benchmark_parser.add_argument("--sibling-window", type=int, default=1)
    benchmark_parser.add_argument("--exclude-ancestors", action="store_true")
    benchmark_parser.add_argument("--segmentation-model", default=ModelConfig().segmentation_model)
    benchmark_parser.add_argument(
        "--summarization-model",
        default=ModelConfig().summarization_model,
    )
    benchmark_parser.add_argument("--routing-model", default=ModelConfig().routing_model)
    benchmark_parser.add_argument("--answer-model", default=ModelConfig().answer_model)

    corpus_benchmark_parser = subparsers.add_parser(
        "corpus-benchmark",
        help="Build a corpus and run benchmark cases against it.",
    )
    corpus_benchmark_parser.add_argument("corpus_path")
    corpus_benchmark_parser.add_argument("cases_path")
    corpus_benchmark_parser.add_argument("input_paths", nargs="+")
    corpus_benchmark_parser.add_argument("--subsection-threshold", type=int, default=300)
    corpus_benchmark_parser.add_argument("--max-depth", type=int, default=4)
    corpus_benchmark_parser.add_argument("--cache-dir", default=".cache/treerag")
    corpus_benchmark_parser.add_argument("--disable-cache", action="store_true")
    corpus_benchmark_parser.add_argument("--sibling-window", type=int, default=1)
    corpus_benchmark_parser.add_argument("--exclude-ancestors", action="store_true")
    corpus_benchmark_parser.add_argument(
        "--segmentation-model",
        default=ModelConfig().segmentation_model,
    )
    corpus_benchmark_parser.add_argument(
        "--summarization-model",
        default=ModelConfig().summarization_model,
    )
    corpus_benchmark_parser.add_argument("--routing-model", default=ModelConfig().routing_model)
    corpus_benchmark_parser.add_argument("--answer-model", default=ModelConfig().answer_model)

    return parser


def main(argv: Sequence[str] | None = None, *, provider: LLMProvider | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "index":
        index_config = IndexConfig(
            subsection_word_threshold=args.subsection_threshold,
            max_depth=args.max_depth,
            cache_dir=Path(args.cache_dir),
            use_cache=not args.disable_cache,
        )
        model_config = ModelConfig(
            segmentation_model=args.segmentation_model,
            summarization_model=args.summarization_model,
        )
        document_index = build_index(
            args.input_path,
            args.output_path,
            index_config,
            model_config=model_config,
            provider=provider,
        )
        print(
            json.dumps(
                {
                    "source_path": document_index.source_path,
                    "output_path": args.output_path,
                    "root_title": document_index.root.title,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "ask":
        retrieval_config = _retrieval_config_from_args(args)
        model_config = _query_model_config_from_args(args)
        result = query_index(
            args.question,
            args.index_path,
            retrieval_config,
            model_config=model_config,
            provider=provider,
        )
        print(json.dumps(_index_query_output(result), indent=2))
        return 0

    if args.command == "repl":
        retrieval_config = _retrieval_config_from_args(args)
        model_config = _query_model_config_from_args(args)
        return _interactive_query_loop(
            lambda question: _index_query_output(
                query_index(
                    question,
                    args.index_path,
                    retrieval_config,
                    model_config=model_config,
                    provider=provider,
                )
            )
        )

    if args.command == "inspect":
        document_index = load_index(Path(args.index_path))
        print(
            json.dumps(
                {
                    "source_path": document_index.source_path,
                    "source_hash": document_index.source_hash,
                    "created_at": document_index.created_at,
                    "root_title": document_index.root.title,
                    "child_count": len(document_index.root.children),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "corpus-index":
        index_config = IndexConfig(
            subsection_word_threshold=args.subsection_threshold,
            max_depth=args.max_depth,
            cache_dir=Path(args.cache_dir),
            use_cache=not args.disable_cache,
        )
        model_config = ModelConfig(
            segmentation_model=args.segmentation_model,
            summarization_model=args.summarization_model,
        )
        corpus_index = build_corpus(
            args.input_paths,
            args.output_path,
            index_config,
            model_config=model_config,
            provider=provider,
        )
        print(
            json.dumps(
                {
                    "corpus_path": str(Path(args.output_path) / "corpus.json")
                    if Path(args.output_path).suffix != ".json"
                    else args.output_path,
                    "document_count": len(corpus_index.documents),
                    "document_titles": [document.title for document in corpus_index.documents],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "corpus-ask":
        retrieval_config = _retrieval_config_from_args(args)
        model_config = _query_model_config_from_args(args)
        corpus_result = query_corpus(
            args.question,
            args.corpus_path,
            retrieval_config,
            model_config=model_config,
            provider=provider,
        )
        print(json.dumps(_corpus_query_output(corpus_result), indent=2))
        return 0

    if args.command == "corpus-repl":
        retrieval_config = _retrieval_config_from_args(args)
        model_config = _query_model_config_from_args(args)
        return _interactive_query_loop(
            lambda question: _corpus_query_output(
                query_corpus(
                    question,
                    args.corpus_path,
                    retrieval_config,
                    model_config=model_config,
                    provider=provider,
                )
            )
        )

    if args.command == "corpus-inspect":
        corpus_index = load_corpus(Path(args.corpus_path))
        print(
            json.dumps(
                {
                    "created_at": corpus_index.created_at,
                    "document_count": len(corpus_index.documents),
                    "document_titles": [document.title for document in corpus_index.documents],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "benchmark":
        index_config = IndexConfig(
            subsection_word_threshold=args.subsection_threshold,
            max_depth=args.max_depth,
            cache_dir=Path(args.cache_dir),
            use_cache=not args.disable_cache,
        )
        retrieval_config = RetrievalConfig(
            sibling_window=args.sibling_window,
            include_ancestor_summaries=not args.exclude_ancestors,
        )
        model_config = ModelConfig(
            segmentation_model=args.segmentation_model,
            summarization_model=args.summarization_model,
            routing_model=args.routing_model,
            answer_model=args.answer_model,
        )
        report = run_benchmark(
            args.input_path,
            args.cases_path,
            args.index_path,
            index_config,
            retrieval_config,
            model_config=model_config,
            provider=provider,
        )
        print(json.dumps(report.to_dict(), indent=2))
        return 0

    if args.command == "corpus-benchmark":
        index_config = IndexConfig(
            subsection_word_threshold=args.subsection_threshold,
            max_depth=args.max_depth,
            cache_dir=Path(args.cache_dir),
            use_cache=not args.disable_cache,
        )
        retrieval_config = RetrievalConfig(
            sibling_window=args.sibling_window,
            include_ancestor_summaries=not args.exclude_ancestors,
        )
        model_config = ModelConfig(
            segmentation_model=args.segmentation_model,
            summarization_model=args.summarization_model,
            routing_model=args.routing_model,
            answer_model=args.answer_model,
        )
        report = run_corpus_benchmark(
            args.input_paths,
            args.cases_path,
            args.corpus_path,
            index_config,
            retrieval_config,
            model_config=model_config,
            provider=provider,
        )
        print(json.dumps(report.to_dict(), indent=2))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def _retrieval_config_from_args(args: argparse.Namespace) -> RetrievalConfig:
    return RetrievalConfig(
        sibling_window=args.sibling_window,
        include_ancestor_summaries=not args.exclude_ancestors,
    )


def _query_model_config_from_args(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        routing_model=args.routing_model,
        answer_model=args.answer_model,
    )


def _index_query_output(result: Any) -> dict[str, Any]:
    return {
        "answer": result.answer,
        "selected_leaf_title": result.selected_leaf_title,
        "navigation_path": result.navigation_path,
        "included_sections": result.included_sections,
    }


def _corpus_query_output(result: Any) -> dict[str, Any]:
    return {
        "document_id": result.document_id,
        "document_title": result.document_title,
        "selected_leaf_title": result.selected_leaf_title,
        "navigation_path": result.navigation_path,
        "included_sections": result.included_sections,
        "answer": result.answer,
    }


def _interactive_query_loop(render_query: Callable[[str], dict[str, Any]]) -> int:
    while True:
        try:
            question = input("treerag> ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            return 0

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            return 0

        print(json.dumps(render_query(question), indent=2))
