from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from treerag.storage import (
    DocumentIndex,
    IndexMetadata,
    IndexNode,
    MalformedIndexError,
    MissingIndexError,
    load,
    save,
)


def build_sample_index() -> DocumentIndex:
    root = IndexNode(
        title="root",
        summary="document summary",
        depth=0,
    )
    parent = IndexNode(
        title="Parent",
        content="parent content",
        summary="parent summary",
        depth=1,
        parent=root,
    )
    leaf_one = IndexNode(
        title="Leaf One",
        content="leaf one content",
        summary="leaf one summary",
        depth=2,
        parent=parent,
    )
    leaf_two = IndexNode(
        title="Leaf Two",
        content="leaf two content",
        summary="leaf two summary",
        depth=2,
        parent=parent,
    )
    parent.children.extend([leaf_one, leaf_two])
    root.children.append(parent)

    metadata = IndexMetadata(
        source_path="/docs/source.md",
        created_at=datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc).isoformat(),
        generator="TreeRAG",
        version="0.1.0",
        document_hash="abc123",
        extra={"topic": "jira", "unicode": "café"},
    )
    return DocumentIndex(root=root, metadata=metadata)


def test_storage_round_trip_preserves_tree_and_metadata(tmp_path: Path) -> None:
    original = build_sample_index()
    index_path = tmp_path / "index.json"

    save(original, index_path)
    restored = load(index_path)

    assert restored.schema_version == original.schema_version
    assert restored.metadata == original.metadata
    assert restored.root.title == "root"
    assert restored.root.children[0].parent is restored.root
    assert restored.root.children[0].children[0].parent is restored.root.children[0]
    assert restored.root.children[0].children[1].title == "Leaf Two"
    assert restored.root.children[0].children[0].content == "leaf one content"
    assert restored.root.children[0].children[1].summary == "leaf two summary"


def test_load_missing_index_raises_clear_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.json"

    with pytest.raises(MissingIndexError, match="does not exist"):
        load(missing_path)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("not json", "not valid JSON"),
        ('{"schema_version": 1, "metadata": {}, "root": {}}', "missing a string 'source_path'"),
        (
            (
                '{"schema_version": 1, "metadata": {"source_path": "x", '
                '"created_at": "y", "generator": "TreeRAG", "version": "0.1.0", '
                '"document_hash": "", "extra": []}, "root": {"title": "root", '
                '"content": "", "summary": "", "depth": 0, "children": []}}'
            ),
            "metadata.extra must be a JSON object",
        ),
        (
            (
                '{"schema_version": 99, "metadata": {"source_path": "x", '
                '"created_at": "y", "generator": "TreeRAG", "version": "0.1.0", '
                '"document_hash": "", "extra": {}}, "root": {"title": "root", '
                '"content": "", "summary": "", "depth": 0, "children": []}}'
            ),
            "Unsupported index schema_version 99",
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
