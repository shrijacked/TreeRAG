"""Multi-document corpus indexing and querying."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from treerag.api import build_index, query_index
from treerag.config import IndexConfig, ModelConfig, RetrievalConfig
from treerag.errors import ParseError, RoutingError, StorageError
from treerag.provider import LLMProvider, OpenAIProvider, RouteChoice

CORPUS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CorpusDocument:
    """A document entry inside a corpus manifest."""

    document_id: str
    title: str
    summary: str
    source_path: str
    source_hash: str
    index_path: str


@dataclass(frozen=True)
class CorpusIndex:
    """A saved corpus manifest with per-document routing summaries."""

    documents: list[CorpusDocument]
    created_at: str
    model_config: dict[str, Any]
    index_config: dict[str, Any]


@dataclass(frozen=True)
class CorpusQueryResult:
    """Answer and routing metadata returned by corpus querying."""

    document_id: str
    document_title: str
    document_index_path: str
    answer: str
    context: str
    selected_leaf_id: str
    selected_leaf_title: str
    navigation_path: list[str]
    included_sections: list[str]


def build_corpus(
    input_paths: list[str | Path],
    output_path: str | Path,
    config: IndexConfig,
    *,
    model_config: ModelConfig | None = None,
    provider: LLMProvider | None = None,
) -> CorpusIndex:
    """Build a corpus manifest and per-document indexes."""

    if not input_paths:
        raise ParseError("Corpus indexing requires at least one input document.")

    active_model_config = model_config or ModelConfig()
    active_provider = provider or OpenAIProvider()
    manifest_path = _resolve_corpus_manifest_path(output_path)
    indexes_dir = manifest_path.parent / "documents"
    indexes_dir.mkdir(parents=True, exist_ok=True)

    used_ids: set[str] = set()
    documents: list[CorpusDocument] = []
    for input_path in input_paths:
        source_path = Path(input_path)
        source_text = source_path.read_text(encoding="utf-8")
        document_id = _allocate_document_id(source_path, used_ids)
        index_path = indexes_dir / f"{document_id}.index.json"
        document_index = build_index(
            source_path,
            index_path,
            config,
            model_config=active_model_config,
            provider=active_provider,
        )
        documents.append(
            CorpusDocument(
                document_id=document_id,
                title=_document_title(document_index.root, source_text, source_path),
                summary=document_index.root.summary,
                source_path=str(source_path.resolve()),
                source_hash=document_index.source_hash,
                index_path=str(index_path.resolve()),
            )
        )

    corpus_index = CorpusIndex(
        documents=documents,
        created_at=datetime.now(timezone.utc).isoformat(),
        model_config=active_model_config.to_dict(),
        index_config=config.to_dict(),
    )
    save_corpus(corpus_index, manifest_path)
    return corpus_index


def query_corpus(
    question: str,
    corpus_path: str | Path,
    config: RetrievalConfig,
    *,
    model_config: ModelConfig | None = None,
    provider: LLMProvider | None = None,
) -> CorpusQueryResult:
    """Route a query to the best document in the corpus, then to the best leaf."""

    corpus_index = load_corpus(corpus_path)
    if not corpus_index.documents:
        raise ParseError("Corpus manifest does not contain any indexed documents.")

    active_model_config = model_config or ModelConfig()
    active_provider = provider or OpenAIProvider()
    document = _select_document(
        question,
        corpus_index,
        provider=active_provider,
        model_config=active_model_config,
    )
    query_result = query_index(
        question,
        document.index_path,
        config,
        model_config=active_model_config,
        provider=active_provider,
    )
    return CorpusQueryResult(
        document_id=document.document_id,
        document_title=document.title,
        document_index_path=document.index_path,
        answer=query_result.answer,
        context=query_result.context,
        selected_leaf_id=query_result.selected_leaf_id,
        selected_leaf_title=query_result.selected_leaf_title,
        navigation_path=query_result.navigation_path,
        included_sections=query_result.included_sections,
    )


def save_corpus(corpus_index: CorpusIndex, path: str | Path) -> None:
    """Persist a corpus manifest to disk."""

    manifest_path = _resolve_corpus_manifest_path(path)
    payload = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "created_at": corpus_index.created_at,
        "model_config": corpus_index.model_config,
        "index_config": corpus_index.index_config,
        "documents": [
            {
                "document_id": document.document_id,
                "title": document.title,
                "summary": document.summary,
                "source_path": document.source_path,
                "source_hash": document.source_hash,
                "index_path": document.index_path,
            }
            for document in corpus_index.documents
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_corpus(path: str | Path) -> CorpusIndex:
    """Load a corpus manifest from a directory or explicit JSON file path."""

    manifest_path = _resolve_corpus_manifest_path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ParseError(f"Corpus manifest does not exist: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ParseError(f"Corpus manifest is not valid JSON: {manifest_path}") from exc
    except UnicodeDecodeError as exc:
        raise ParseError(f"Corpus manifest is not valid UTF-8: {manifest_path}") from exc
    except OSError as exc:
        raise StorageError(f"Unable to read corpus manifest {manifest_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ParseError("Corpus manifest must be a JSON object.")

    schema_version = payload.get("schema_version")
    if schema_version != CORPUS_SCHEMA_VERSION:
        raise ParseError(
            f"Unsupported corpus schema_version {schema_version}; expected "
            f"{CORPUS_SCHEMA_VERSION}."
        )

    created_at = _require_str(payload, "created_at", "corpus")
    model_config = _require_mapping(payload, "model_config", "corpus")
    index_config = _require_mapping(payload, "index_config", "corpus")
    raw_documents = payload.get("documents")
    if not isinstance(raw_documents, list):
        raise ParseError("Corpus manifest is missing a list 'documents' field.")

    documents: list[CorpusDocument] = []
    for raw_document in raw_documents:
        if not isinstance(raw_document, Mapping):
            raise ParseError("Each corpus document must be a JSON object.")
        documents.append(
            CorpusDocument(
                document_id=_require_str(raw_document, "document_id", "corpus document"),
                title=_require_str(raw_document, "title", "corpus document"),
                summary=_require_str(raw_document, "summary", "corpus document"),
                source_path=_require_str(raw_document, "source_path", "corpus document"),
                source_hash=_require_str(raw_document, "source_hash", "corpus document"),
                index_path=_require_str(raw_document, "index_path", "corpus document"),
            )
        )

    return CorpusIndex(
        documents=documents,
        created_at=created_at,
        model_config=dict(model_config),
        index_config=dict(index_config),
    )


def _select_document(
    question: str,
    corpus_index: CorpusIndex,
    *,
    provider: LLMProvider,
    model_config: ModelConfig,
) -> CorpusDocument:
    if len(corpus_index.documents) == 1:
        return corpus_index.documents[0]

    choices = [
        RouteChoice(title=document.title, summary=document.summary)
        for document in corpus_index.documents
    ]
    selected_index = provider.route(
        question,
        node_title="corpus",
        choices=choices,
        model_config=model_config,
    )
    if selected_index < 0 or selected_index >= len(corpus_index.documents):
        raise RoutingError(
            f"Invalid corpus route choice {selected_index!r} for {len(corpus_index.documents)} "
            "documents."
        )
    return corpus_index.documents[selected_index]


def _document_title(root: Any, source_text: str, source_path: Path) -> str:
    markdown_title = _markdown_title(source_text)
    if markdown_title is not None:
        return markdown_title
    if getattr(root, "children", None):
        first_child = root.children[0]
        if isinstance(first_child.title, str) and first_child.title.strip():
            return first_child.title
    return _humanize_stem(source_path.stem)


def _resolve_corpus_manifest_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.suffix == ".json":
        return candidate
    return candidate / "corpus.json"


def _allocate_document_id(source_path: Path, used_ids: set[str]) -> str:
    base_id = _slugify(source_path.stem)
    candidate = base_id
    counter = 2
    while candidate in used_ids:
        candidate = f"{base_id}-{counter}"
        counter += 1
    used_ids.add(candidate)
    return candidate


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "document"


def _humanize_stem(value: str) -> str:
    words = [word for word in re.split(r"[_-]+", value) if word]
    if not words:
        return "Document"
    return " ".join(word.capitalize() for word in words)


def _markdown_title(source_text: str) -> str | None:
    for line in source_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            if title:
                return title
        if stripped:
            break
    return None


def _require_str(payload: Mapping[str, Any], key: str, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ParseError(f"{context} is missing a string '{key}' field.")
    return value


def _require_mapping(payload: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ParseError(f"{context} is missing an object '{key}' field.")
    return value
