"""Core data structures for TreeRAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class Section:
    """A segmented section returned by the provider."""

    title: str
    content: str


@dataclass
class PageNode:
    """A single node in the document tree."""

    node_id: str
    title: str
    content: str
    summary: str
    depth: int
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
    selected_leaf_id: str
    selected_leaf_title: str
    navigation_path: list[str]
    included_sections: list[str]
