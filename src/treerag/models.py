"""Core data structures for TreeRAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class Section:
    """A segmented section returned by the provider."""

    title: str
    content: str


@dataclass(frozen=True)
class SourceSpan:
    """Absolute source offsets and human-friendly line numbers for a node."""

    start_char: int
    end_char: int
    start_line: int
    end_line: int


@dataclass(frozen=True)
class SourceReference:
    """A traceable source citation for a retrieved node."""

    node_id: str
    title: str
    start_char: int
    end_char: int
    start_line: int
    end_line: int


@dataclass
class PageNode:
    """A single node in the document tree."""

    node_id: str
    title: str
    content: str
    summary: str
    depth: int
    source_span: SourceSpan | None = None
    children: list["PageNode"] = field(default_factory=list)
    parent: "PageNode | None" = field(default=None, repr=False, compare=False)

    def is_leaf(self) -> bool:
        return not self.children

    def set_children(self, children: list["PageNode"]) -> None:
        self.children = children
        for child in children:
            child.parent = self

    def walk(self) -> Iterator["PageNode"]:
        yield self
        for child in self.children:
            yield from child.walk()

    def path_titles(self) -> list[str]:
        titles: list[str] = []
        current: PageNode | None = self
        while current is not None:
            titles.append(current.title)
            current = current.parent
        return list(reversed(titles))


@dataclass
class DocumentIndex:
    """A saved document index with metadata."""

    root: PageNode
    source_path: str
    source_hash: str
    created_at: str
    model_config: dict[str, Any]
    index_config: dict[str, Any]


@dataclass
class QueryResult:
    """Answer and retrieval metadata returned by query execution."""

    answer: str
    context: str
    source_path: str
    selected_leaf_id: str
    selected_leaf_title: str
    selected_source_span: SourceSpan | None
    navigation_path: list[str]
    included_sections: list[str]
    source_references: list[SourceReference]
