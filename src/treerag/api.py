"""Public entrypoints for TreeRAG."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from treerag.cache import FileCache
from treerag.config import IndexConfig, ModelConfig, RetrievalConfig
from treerag.indexer import build_summaries
from treerag.models import DocumentIndex, QueryResult
from treerag.parser import parse_document
from treerag.provider import LLMProvider, OpenAIProvider
from treerag.retrieval import assemble_context, retrieve_leaf
from treerag.storage import load_index, save_index


def build_index(
    input_path: str | Path,
    output_path: str | Path,
    config: IndexConfig,
    *,
    model_config: ModelConfig | None = None,
    provider: LLMProvider | None = None,
) -> DocumentIndex:
    source_path = Path(input_path)
    output = Path(output_path)
    text = source_path.read_text(encoding="utf-8")
    active_model_config = model_config or ModelConfig()
    active_provider = provider or OpenAIProvider()
    cache = FileCache(config.cache_dir)

    root = parse_document(
        text,
        provider=active_provider,
        index_config=config,
        model_config=active_model_config,
        cache=cache,
    )
    build_summaries(
        root,
        provider=active_provider,
        index_config=config,
        model_config=active_model_config,
        cache=cache,
    )

    document_index = DocumentIndex(
        root=root,
        source_path=str(source_path),
        source_hash=_hash_text(text),
        created_at=datetime.now(timezone.utc).isoformat(),
        model_config=active_model_config.to_dict(),
        index_config=config.to_dict(),
    )
    save_index(document_index, output)
    return document_index


def query_index(
    question: str,
    index_path: str | Path,
    config: RetrievalConfig,
    *,
    model_config: ModelConfig | None = None,
    provider: LLMProvider | None = None,
) -> QueryResult:
    active_model_config = model_config or ModelConfig()
    active_provider = provider or OpenAIProvider()

    document_index = load_index(Path(index_path))
    leaf = retrieve_leaf(
        question,
        root=document_index.root,
        provider=active_provider,
        model_config=active_model_config,
    )
    context, included_nodes = assemble_context(leaf, config=config)
    answer = active_provider.answer(question, context=context, model_config=active_model_config)
    return QueryResult(
        answer=answer,
        context=context,
        selected_leaf_id=leaf.node_id,
        selected_leaf_title=leaf.title,
        navigation_path=leaf.path_titles(),
        included_sections=[node.title for node in included_nodes],
    )


def _hash_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()
