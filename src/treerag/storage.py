"""JSON serialization for TreeRAG indexes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from treerag.errors import StorageError
from treerag.models import DocumentIndex, PageNode, SourceSpan

SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = {1, 2}


class MissingIndexError(FileNotFoundError, StorageError):
    """Raised when an expected index file does not exist."""


class MalformedIndexError(StorageError):
    """Raised when an index payload is unreadable or structurally invalid."""


def save(index: DocumentIndex, path: str | Path) -> None:
    save_index(index, path)


def save_index(index: DocumentIndex, path: str | Path) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_path": index.source_path,
        "source_hash": index.source_hash,
        "created_at": index.created_at,
        "model_config": index.model_config,
        "index_config": index.index_config,
        "root": _node_to_dict(index.root),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load(path: str | Path) -> DocumentIndex:
    return load_index(path)


def load_index(path: str | Path) -> DocumentIndex:
    source = Path(path)
    if not source.exists():
        raise MissingIndexError(f"Index file does not exist: {source}")

    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise MalformedIndexError(f"Index file is not valid UTF-8: {source}") from exc
    except json.JSONDecodeError as exc:
        raise MalformedIndexError(f"Index file is not valid JSON: {source}") from exc
    except OSError as exc:
        raise StorageError(f"Unable to read index file {source}: {exc}") from exc

    if not isinstance(payload, dict):
        raise MalformedIndexError("Index payload must be a JSON object.")

    schema_version = payload.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise MalformedIndexError(
            "Unsupported index schema_version "
            f"{schema_version}; expected one of {sorted(SUPPORTED_SCHEMA_VERSIONS)}."
        )

    source_path = _require_str(payload, "source_path", "index")
    source_hash = _require_str(payload, "source_hash", "index")
    created_at = _require_str(payload, "created_at", "index")
    model_config = _require_mapping(payload, "model_config", "index")
    index_config = _require_mapping(payload, "index_config", "index")
    root_payload = _require_mapping(payload, "root", "index")

    root = _node_from_dict(root_payload, parent=None)
    return DocumentIndex(
        root=root,
        source_path=source_path,
        source_hash=source_hash,
        created_at=created_at,
        model_config=dict(model_config),
        index_config=dict(index_config),
    )


def _node_to_dict(node: PageNode) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "title": node.title,
        "content": node.content,
        "summary": node.summary,
        "depth": node.depth,
        "source_span": _span_to_dict(node.source_span),
        "children": [_node_to_dict(child) for child in node.children],
    }


def _node_from_dict(payload: Mapping[str, Any], parent: PageNode | None) -> PageNode:
    node = PageNode(
        node_id=_require_str(payload, "node_id", "node"),
        title=_require_str(payload, "title", "node"),
        content=_require_str(payload, "content", "node"),
        summary=_require_str(payload, "summary", "node"),
        depth=_require_int(payload, "depth", "node"),
        source_span=_optional_span(payload.get("source_span"), "node"),
    )
    node.parent = parent
    children_payload = payload.get("children")
    if not isinstance(children_payload, list):
        raise MalformedIndexError("node is missing a list 'children' field")
    children = [
        _node_from_dict(_coerce_mapping(child, "child node"), parent=node)
        for child in children_payload
    ]
    node.set_children(children)
    return node


def _require_str(payload: Mapping[str, Any], key: str, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise MalformedIndexError(f"{context} is missing a string '{key}' field")
    return value


def _require_int(payload: Mapping[str, Any], key: str, context: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise MalformedIndexError(f"{context} is missing an integer '{key}' field")
    return value


def _require_mapping(payload: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise MalformedIndexError(f"{context} is missing an object '{key}' field")
    return value


def _coerce_mapping(payload: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise MalformedIndexError(f"{context} must be a JSON object")
    return payload


def _span_to_dict(span: SourceSpan | None) -> dict[str, int] | None:
    if span is None:
        return None
    return {
        "start_char": span.start_char,
        "end_char": span.end_char,
        "start_line": span.start_line,
        "end_line": span.end_line,
    }


def _optional_span(payload: Any, context: str) -> SourceSpan | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise MalformedIndexError(f"{context} has an invalid 'source_span' field")
    return SourceSpan(
        start_char=_require_int(payload, "start_char", "source span"),
        end_char=_require_int(payload, "end_char", "source span"),
        start_line=_require_int(payload, "start_line", "source span"),
        end_line=_require_int(payload, "end_line", "source span"),
    )
