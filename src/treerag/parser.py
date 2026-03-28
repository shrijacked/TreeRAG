"""Recursive document segmentation."""

from __future__ import annotations

import hashlib

from treerag.cache import FileCache
from treerag.config import IndexConfig, ModelConfig
from treerag.models import PageNode, Section
from treerag.provider import LLMProvider


def parse_document(
    text: str,
    *,
    provider: LLMProvider,
    index_config: IndexConfig,
    model_config: ModelConfig,
    cache: FileCache | None = None,
) -> PageNode:
    root = PageNode(node_id="root", title="root", content="", summary="", depth=0)
    children = _segment_to_children(
        text=text,
        provider=provider,
        index_config=index_config,
        model_config=model_config,
        cache=cache,
        depth=1,
        parent_id="root",
    )
    if not children:
        root.content = text
        return root
    root.set_children(children)
    return root


def _segment_to_children(
    *,
    text: str,
    provider: LLMProvider,
    index_config: IndexConfig,
    model_config: ModelConfig,
    cache: FileCache | None,
    depth: int,
    parent_id: str,
) -> list[PageNode]:
    sections = _segment_text(
        text=text,
        provider=provider,
        index_config=index_config,
        model_config=model_config,
        cache=cache,
    )
    children: list[PageNode] = []
    for index, section in enumerate(sections):
        node_id = f"{parent_id}.{index}"
        child = PageNode(
            node_id=node_id,
            title=section.title,
            content=section.content,
            summary="",
            depth=depth,
        )
        should_expand = (
            depth < index_config.max_depth
            and len(section.content.split()) > index_config.subsection_word_threshold
        )
        if should_expand:
            grandchildren = _segment_to_children(
                text=section.content,
                provider=provider,
                index_config=index_config,
                model_config=model_config,
                cache=cache,
                depth=depth + 1,
                parent_id=node_id,
            )
            if grandchildren:
                child.content = ""
                child.set_children(grandchildren)
        children.append(child)
    return children


def _segment_text(
    *,
    text: str,
    provider: LLMProvider,
    index_config: IndexConfig,
    model_config: ModelConfig,
    cache: FileCache | None,
) -> list[Section]:
    cache_key = _hash_inputs(
        "segment",
        text,
        model_config.segmentation_model,
        str(index_config.segment_char_limit),
        str(model_config.segmentation_max_completion_tokens),
    )
    if cache is not None and index_config.use_cache:
        cached = cache.get("segment", cache_key)
        if cached is not None:
            from treerag.models import Section

            return [Section(**entry) for entry in cached]

    sections = provider.segment(
        text,
        model_config=model_config,
        char_limit=index_config.segment_char_limit,
    )
    if cache is not None and index_config.use_cache:
        cache.set("segment", cache_key, [section.__dict__ for section in sections])
    return sections


def _hash_inputs(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
