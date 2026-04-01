from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from treerag.models import DocumentIndex, PageNode, SourceSpan
from treerag.storage import (
    MalformedIndexError,
    MissingIndexError,
    load,
    save,
)


def build_sample_index() -> DocumentIndex:
    root = PageNode(
        node_id="root",
        title="root",
        content="",
        summary="document summary",
        depth=0,
        source_span=SourceSpan(start_char=0, end_char=64, start_line=1, end_line=8),
    )
    parent = PageNode(
        node_id="root.0",
        title="Parent",
        content="parent content",
        summary="parent summary",
        depth=1,
        source_span=SourceSpan(start_char=10, end_char=40, start_line=2, end_line=5),
    )
    leaf_one = PageNode(
        node_id="root.0.0",
        title="Leaf One",
        content="leaf one content",
        summary="leaf one summary",
        depth=2,
        source_span=SourceSpan(start_char=10, end_char=24, start_line=3, end_line=3),
    )
    leaf_two = PageNode(
        node_id="root.0.1",
        title="Leaf Two",
        content="leaf two content",
        summary="leaf two summary",
        depth=2,
        source_span=SourceSpan(start_char=26, end_char=40, start_line=5, end_line=5),
    )
    parent.set_children([leaf_one, leaf_two])
    root.set_children([parent])

    return DocumentIndex(
        root=root,
        source_path="/docs/source.md",
        created_at=datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc).isoformat(),
        source_hash="abc123",
        model_config={"answer_model": "gpt-5.4"},
        index_config={"topic": "jira", "unicode": "café"},
    )


def test_storage_round_trip_preserves_tree_and_metadata(tmp_path: Path) -> None:
    original = build_sample_index()
    index_path = tmp_path / "index.json"

    save(original, index_path)
    restored = load(index_path)

    assert restored.source_path == original.source_path
    assert restored.source_hash == original.source_hash
    assert restored.created_at == original.created_at
    assert restored.model_config == original.model_config
    assert restored.index_config == original.index_config
    assert restored.root.title == "root"
    assert restored.root.children[0].parent is restored.root
    assert restored.root.children[0].children[0].parent is restored.root.children[0]
    assert restored.root.children[0].children[1].title == "Leaf Two"
    assert restored.root.children[0].children[0].content == "leaf one content"
    assert restored.root.children[0].children[1].summary == "leaf two summary"
    assert restored.root.source_span == SourceSpan(
        start_char=0,
        end_char=64,
        start_line=1,
        end_line=8,
    )
    assert restored.root.children[0].children[1].source_span == SourceSpan(
        start_char=26,
        end_char=40,
        start_line=5,
        end_line=5,
    )


def test_load_missing_index_raises_clear_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.json"

    with pytest.raises(MissingIndexError, match="does not exist"):
        load(missing_path)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("not json", "not valid JSON"),
        ('{"schema_version": 1, "root": {}}', "missing a string 'source_path'"),
        (
            (
                '{"schema_version": 1, "source_path": "x", "source_hash": "y", '
                '"created_at": "z", "model_config": [], "index_config": {}, '
                '"root": {"node_id": "root", "title": "root", "content": "", '
                '"summary": "", "depth": 0, "children": []}}'
            ),
            "missing an object 'model_config'",
        ),
        (
            (
                '{"schema_version": 99, "source_path": "x", "source_hash": "y", '
                '"created_at": "z", "model_config": {}, "index_config": {}, '
                '"root": {"node_id": "root", "title": "root", "content": "", '
                '"summary": "", "depth": 0, "children": []}}'
            ),
            "Unsupported index schema_version 99; expected one of",
        ),
    ],
)
def test_load_malformed_payloads_raise_clear_errors(
    tmp_path: Path, payload: str, message: str
) -> None:
    index_path = tmp_path / "bad.json"
    index_path.write_text(payload, encoding="utf-8")

    with pytest.raises(MalformedIndexError, match=message):
        load(index_path)
