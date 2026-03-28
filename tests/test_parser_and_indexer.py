from __future__ import annotations

from pathlib import Path

from tests.support.fake_provider import FakeProvider
from treerag.cache import FileCache
from treerag.config import IndexConfig, ModelConfig
from treerag.indexer import build_summaries
from treerag.models import Section
from treerag.parser import parse_document


def test_parse_document_recurses_beyond_two_levels() -> None:
    provider = FakeProvider(
        segment_responses=[
            [Section(title="Top", content=" ".join(["sub"] * 301))],
            [Section(title="Mid", content=" ".join(["deep"] * 301))],
            [Section(title="Leaf", content="leaf content")],
        ]
    )

    root = parse_document(
        "ignored",
        provider=provider,
        index_config=IndexConfig(subsection_word_threshold=300, max_depth=4),
        model_config=ModelConfig(),
    )

    assert root.children[0].depth == 1
    assert root.children[0].children[0].depth == 2
    assert root.children[0].children[0].children[0].depth == 3
    assert root.children[0].children[0].children[0].content == "leaf content"


def test_build_summaries_reuses_cache_for_identical_tree(tmp_path: Path) -> None:
    provider = FakeProvider(
        summary_responses=["leaf summary", "root summary"],
    )
    cache = FileCache(tmp_path / "cache")
    root = parse_document(
        "ignored",
        provider=FakeProvider(segment_responses=[[Section(title="Leaf", content="leaf content")]]),
        index_config=IndexConfig(),
        model_config=ModelConfig(),
        cache=cache,
    )

    build_summaries(
        root,
        provider=provider,
        index_config=IndexConfig(),
        model_config=ModelConfig(),
        cache=cache,
    )
    build_summaries(
        root,
        provider=provider,
        index_config=IndexConfig(),
        model_config=ModelConfig(),
        cache=cache,
    )

    assert provider.summary_calls == 2
