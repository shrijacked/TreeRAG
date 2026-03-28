"""Summary generation over parsed document trees."""

from __future__ import annotations

import hashlib

from treerag.cache import FileCache
from treerag.config import IndexConfig, ModelConfig
from treerag.models import PageNode
from treerag.provider import LLMProvider


def build_summaries(
    node: PageNode,
    *,
    provider: LLMProvider,
    index_config: IndexConfig,
    model_config: ModelConfig,
    cache: FileCache | None = None,
) -> None:
    for child in node.children:
        build_summaries(
            child,
            provider=provider,
            index_config=index_config,
            model_config=model_config,
            cache=cache,
        )

    if node.is_leaf():
        if node.content.strip():
            node.summary = _summarize(
                node.content,
                section_name=node.title,
                provider=provider,
                index_config=index_config,
                model_config=model_config,
                cache=cache,
            )
        else:
            node.summary = "(empty section)"
        return

    child_summaries = "\n\n".join(f"[{child.title}]: {child.summary}" for child in node.children)
    node.summary = _summarize(
        child_summaries,
        section_name=node.title,
        provider=provider,
        index_config=index_config,
        model_config=model_config,
        cache=cache,
    )


def _summarize(
    text: str,
    *,
    section_name: str,
    provider: LLMProvider,
    index_config: IndexConfig,
    model_config: ModelConfig,
    cache: FileCache | None,
) -> str:
    cache_key = hashlib.sha256(
        "||".join(
            [
                section_name,
                text,
                model_config.summarization_model,
                str(index_config.summary_char_limit),
                str(model_config.summarization_max_completion_tokens),
            ]
        ).encode("utf-8")
    ).hexdigest()
    if cache is not None and index_config.use_cache:
        cached = cache.get("summary", cache_key)
        if isinstance(cached, str):
            return cached

    summary = provider.summarize(
        text,
        section_name=section_name,
        model_config=model_config,
        char_limit=index_config.summary_char_limit,
    )
    if cache is not None and index_config.use_cache:
        cache.set("summary", cache_key, summary)
    return summary
