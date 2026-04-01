"""Recursive document segmentation."""

from __future__ import annotations

import hashlib
import re

from treerag.cache import FileCache
from treerag.config import IndexConfig, ModelConfig
from treerag.models import PageNode, Section, SourceSpan
from treerag.provider import LLMProvider


def parse_document(
    text: str,
    *,
    provider: LLMProvider,
    index_config: IndexConfig,
    model_config: ModelConfig,
    cache: FileCache | None = None,
) -> PageNode:
    root = PageNode(
        node_id="root",
        title="root",
        content="",
        summary="",
        depth=0,
        source_span=_span_for_offsets(text, 0, len(text)),
    )
    children = _segment_to_children(
        text=text,
        document_text=text,
        provider=provider,
        index_config=index_config,
        model_config=model_config,
        cache=cache,
        depth=1,
        parent_id="root",
        absolute_start=0,
    )
    if not children:
        root.content = text
        return root
    root.set_children(children)
    return root


def _segment_to_children(
    *,
    text: str,
    document_text: str,
    provider: LLMProvider,
    index_config: IndexConfig,
    model_config: ModelConfig,
    cache: FileCache | None,
    depth: int,
    parent_id: str,
    absolute_start: int | None,
) -> list[PageNode]:
    sections = _segment_text(
        text=text,
        provider=provider,
        index_config=index_config,
        model_config=model_config,
        cache=cache,
    )
    children: list[PageNode] = []
    located_sections = _locate_sections(
        parent_text=text,
        document_text=document_text,
        sections=sections,
        absolute_start=absolute_start,
    )
    for index, (section, source_span) in enumerate(located_sections):
        node_id = f"{parent_id}.{index}"
        child = PageNode(
            node_id=node_id,
            title=section.title,
            content=section.content,
            summary="",
            depth=depth,
            source_span=source_span,
        )
        should_expand = (
            depth < index_config.max_depth
            and len(section.content.split()) > index_config.subsection_word_threshold
        )
        if should_expand:
            grandchildren = _segment_to_children(
                text=section.content,
                document_text=document_text,
                provider=provider,
                index_config=index_config,
                model_config=model_config,
                cache=cache,
                depth=depth + 1,
                parent_id=node_id,
                absolute_start=None if source_span is None else source_span.start_char,
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


def _locate_sections(
    *,
    parent_text: str,
    document_text: str,
    sections: list[Section],
    absolute_start: int | None,
) -> list[tuple[Section, SourceSpan | None]]:
    if absolute_start is None:
        return [(section, None) for section in sections]

    cursor = 0
    located: list[tuple[Section, SourceSpan | None]] = []
    for section in sections:
        match = _find_section_match(parent_text, section.content, cursor)
        if match is None:
            located.append((section, None))
            continue

        start_offset, end_offset = match
        start_offset = _expand_to_heading(parent_text, start_offset, section.title)
        cursor = max(cursor, end_offset)
        span = _span_for_offsets(
            document_text,
            absolute_start + start_offset,
            absolute_start + end_offset,
        )
        located.append((section, span))
    return located


def _find_section_match(text: str, content: str, start: int) -> tuple[int, int] | None:
    if not content:
        return None

    exact_index = text.find(content, start)
    if exact_index != -1:
        return exact_index, exact_index + len(content)

    stripped = content.strip()
    if not stripped:
        return None

    tokens = stripped.split()
    if not tokens:
        return None
    pattern = re.compile(r"\s+".join(re.escape(token) for token in tokens), re.DOTALL)
    match = pattern.search(text, pos=start)
    if match is None:
        return None
    return match.start(), match.end()


def _expand_to_heading(text: str, content_start: int, title: str) -> int:
    line_start = text.rfind("\n", 0, content_start) + 1
    candidate_start = line_start

    for _ in range(3):
        previous_break = text.rfind("\n", 0, max(candidate_start - 1, 0))
        previous_start = 0 if previous_break == -1 else previous_break + 1
        previous_line = text[previous_start:candidate_start].rstrip("\n")
        stripped = previous_line.strip()
        if not stripped:
            candidate_start = previous_start
            continue
        if stripped.lstrip("#").strip() == title.strip():
            return previous_start
        break

    return content_start


def _span_for_offsets(text: str, start_char: int, end_char: int) -> SourceSpan:
    safe_start = max(0, min(start_char, len(text)))
    safe_end = max(safe_start, min(end_char, len(text)))
    end_line_offset = safe_start if safe_end == safe_start else safe_end - 1
    return SourceSpan(
        start_char=safe_start,
        end_char=safe_end,
        start_line=_line_number(text, safe_start),
        end_line=_line_number(text, end_line_offset),
    )


def _line_number(text: str, offset: int) -> int:
    if not text:
        return 1
    safe_offset = max(0, min(offset, len(text)))
    return text.count("\n", 0, safe_offset) + 1
