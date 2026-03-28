from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Union, overload

JsonValue = Any

SCHEMA_VERSION = 1


class StorageError(RuntimeError):
    """Base error for TreeRAG storage problems."""


class MissingIndexError(StorageError):
    """Raised when an index file does not exist."""


class MalformedIndexError(StorageError):
    """Raised when an index file cannot be parsed or validated."""


@dataclass
class IndexMetadata:
    """Metadata stored alongside a serialized document index."""

    source_path: str
    created_at: str
    generator: str = "TreeRAG"
    version: str = "0.1.0"
    document_hash: str = ""
    extra: dict[str, JsonValue] = field(default_factory=dict)


@dataclass
class IndexNode:
    """A tree node used by the serialized document index."""

    title: str
    content: str = ""
    summary: str = ""
    depth: int = 0
    children: list["IndexNode"] = field(default_factory=list)
    parent: Optional["IndexNode"] = field(default=None, repr=False, compare=False)

    def is_leaf(self) -> bool:
        return not self.children


@dataclass
class DocumentIndex:
    """Full serialized index payload."""

    root: IndexNode
    metadata: IndexMetadata
    schema_version: int = SCHEMA_VERSION


IndexLike = Union[DocumentIndex, IndexNode]


def _to_json_value(value: Any) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_to_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_to_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    raise TypeError(f"Value {value!r} is not JSON serializable")


def _metadata_to_dict(metadata: IndexMetadata) -> dict[str, JsonValue]:
    payload = asdict(metadata)
    payload["extra"] = {key: _to_json_value(value) for key, value in metadata.extra.items()}
    return payload


def _node_to_dict(node: IndexNode) -> dict[str, JsonValue]:
    return {
        "title": node.title,
        "content": node.content,
        "summary": node.summary,
        "depth": node.depth,
        "children": [_node_to_dict(child) for child in node.children],
    }


def _require_mapping(value: Any, context: str) -> MutableMapping[str, Any]:
    if not isinstance(value, MutableMapping):
        raise MalformedIndexError(f"{context} must be a JSON object")
    return value


def _require_str(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise MalformedIndexError(f"{context} is missing a string '{key}' field")
    return value


def _require_int(mapping: Mapping[str, Any], key: str, context: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int):
        raise MalformedIndexError(f"{context} is missing an integer '{key}' field")
    return value


def _require_list(mapping: Mapping[str, Any], key: str, context: str) -> list[Any]:
    value = mapping.get(key)
    if not isinstance(value, list):
        raise MalformedIndexError(f"{context} is missing a list '{key}' field")
    return value


def _node_from_dict(payload: Mapping[str, Any], parent: Optional[IndexNode] = None) -> IndexNode:
    title = _require_str(payload, "title", "node")
    content = _require_str(payload, "content", "node")
    summary = _require_str(payload, "summary", "node")
    depth = _require_int(payload, "depth", "node")
    children_payload = _require_list(payload, "children", "node")

    node = IndexNode(
        title=title,
        content=content,
        summary=summary,
        depth=depth,
        parent=parent,
    )
    for child_payload in children_payload:
        child_mapping = _require_mapping(child_payload, "child node")
        child = _node_from_dict(child_mapping, parent=node)
        node.children.append(child)
    return node


def _metadata_from_dict(payload: Mapping[str, Any]) -> IndexMetadata:
    source_path = _require_str(payload, "source_path", "metadata")
    created_at = _require_str(payload, "created_at", "metadata")
    generator = _require_str(payload, "generator", "metadata")
    version = _require_str(payload, "version", "metadata")
    document_hash = _require_str(payload, "document_hash", "metadata")
    extra_payload = payload.get("extra", {})
    if not isinstance(extra_payload, MutableMapping):
        raise MalformedIndexError("metadata.extra must be a JSON object")

    extra: dict[str, JsonValue] = {}
    for key, value in extra_payload.items():
        extra[str(key)] = _to_json_value(value)

    return IndexMetadata(
        source_path=source_path,
        created_at=created_at,
        generator=generator,
        version=version,
        document_hash=document_hash,
        extra=extra,
    )


def _index_to_dict(index: DocumentIndex) -> dict[str, JsonValue]:
    return {
        "schema_version": index.schema_version,
        "metadata": _metadata_to_dict(index.metadata),
        "root": _node_to_dict(index.root),
    }


def _index_from_dict(payload: Mapping[str, Any]) -> DocumentIndex:
    schema_version = _require_int(payload, "schema_version", "index")
    if schema_version != SCHEMA_VERSION:
        raise MalformedIndexError(
            f"Unsupported index schema_version {schema_version}; expected {SCHEMA_VERSION}"
        )

    metadata_payload = _require_mapping(payload.get("metadata"), "metadata")
    root_payload = _require_mapping(payload.get("root"), "root")
    metadata = _metadata_from_dict(metadata_payload)
    root = _node_from_dict(root_payload, parent=None)
    return DocumentIndex(root=root, metadata=metadata, schema_version=schema_version)


def _coerce_index(index: IndexLike, metadata: Optional[IndexMetadata] = None) -> DocumentIndex:
    if isinstance(index, DocumentIndex):
        return index
    if metadata is None:
        raise ValueError("metadata is required when saving a bare IndexNode")
    return DocumentIndex(root=index, metadata=metadata)


@overload
def save(index: DocumentIndex, path: str | Path) -> None: ...


@overload
def save(index: IndexNode, path: str | Path, metadata: IndexMetadata) -> None: ...


def save(index: IndexLike, path: str | Path, metadata: Optional[IndexMetadata] = None) -> None:
    """Serialize an index to JSON using UTF-8 encoding."""

    document_index = _coerce_index(index, metadata)
    payload = _index_to_dict(document_index)
    destination = Path(path)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_index(index: DocumentIndex, path: str | Path) -> None:
    save(index, path)


def load(path: str | Path) -> DocumentIndex:
    """Load a serialized index from JSON using UTF-8 encoding."""

    source = Path(path)
    if not source.exists():
        raise MissingIndexError(f"Index file does not exist: {source}")
    try:
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise MissingIndexError(f"Index file does not exist: {source}") from exc
    except UnicodeDecodeError as exc:
        raise MalformedIndexError(f"Index file is not valid UTF-8: {source}") from exc
    except json.JSONDecodeError as exc:
        raise MalformedIndexError(f"Index file is not valid JSON: {source}") from exc
    except OSError as exc:
        raise StorageError(f"Unable to read index file {source}: {exc}") from exc

    try:
        index_payload = _require_mapping(payload, "index")
        return _index_from_dict(index_payload)
    except StorageError:
        raise
    except (TypeError, ValueError, KeyError) as exc:
        raise MalformedIndexError(f"Index file has an invalid payload: {source}") from exc


def load_index(path: str | Path) -> DocumentIndex:
    return load(path)


__all__ = [
    "DocumentIndex",
    "IndexMetadata",
    "IndexNode",
    "MalformedIndexError",
    "MissingIndexError",
    "SCHEMA_VERSION",
    "StorageError",
    "load",
    "load_index",
    "save",
    "save_index",
]
