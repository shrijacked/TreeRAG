from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence


class RetrievalError(Exception):
    """Base error for retrieval failures."""


class InvalidRouteChoiceError(RetrievalError):
    """Raised when a router returns a choice that cannot be applied safely."""

    def __init__(self, choice: object, child_count: int, node_title: str) -> None:
        message = (
            f"Invalid route choice {choice!r} for node {node_title!r} with {child_count} children"
        )
        super().__init__(message)
        self.choice = choice
        self.child_count = child_count
        self.node_title = node_title


@dataclass
class TreeNode:
    title: str
    content: str = ""
    summary: str = ""
    children: tuple["TreeNode", ...] = field(default_factory=tuple)
    parent: "TreeNode | None" = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.children = tuple(self.children)
        for child in self.children:
            child.parent = self

    def is_leaf(self) -> bool:
        return len(self.children) == 0


@dataclass(frozen=True)
class RetrievalConfig:
    sibling_window: int = 1
    include_ancestor_summaries: bool = True
    ancestor_summary_depth: int = 1

    def __post_init__(self) -> None:
        if self.sibling_window < 0:
            raise ValueError("sibling_window must be >= 0")
        if self.ancestor_summary_depth < 0:
            raise ValueError("ancestor_summary_depth must be >= 0")


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    leaf: TreeNode
    route_path: tuple[TreeNode, ...]
    context: str
    ancestor_nodes: tuple[TreeNode, ...] = field(default_factory=tuple)
    sibling_nodes: tuple[TreeNode, ...] = field(default_factory=tuple)


class RouteSelector(Protocol):
    def choose_child_index(
        self,
        query: str,
        node: TreeNode,
        children: Sequence[TreeNode],
    ) -> int:
        """Return the 0-based child index to follow."""


def _validate_child_choice(choice: object, child_count: int, node: TreeNode) -> int:
    if isinstance(choice, bool) or not isinstance(choice, int):
        raise InvalidRouteChoiceError(choice, child_count, node.title)
    if choice < 0 or choice >= child_count:
        raise InvalidRouteChoiceError(choice, child_count, node.title)
    return choice


def _find_child_index(node: TreeNode, child: TreeNode) -> int:
    for index, candidate in enumerate(node.children):
        if candidate is child:
            return index
    raise RetrievalError(f"Selected node {child.title!r} is not a child of {node.title!r}")


def _selected_ancestors(path: Sequence[TreeNode], config: RetrievalConfig) -> tuple[TreeNode, ...]:
    if not config.include_ancestor_summaries or config.ancestor_summary_depth == 0:
        return ()
    ancestors = path[:-1]
    if config.ancestor_summary_depth >= len(ancestors):
        return tuple(ancestors)
    return tuple(ancestors[-config.ancestor_summary_depth :])


def _selected_siblings(path: Sequence[TreeNode], config: RetrievalConfig) -> tuple[TreeNode, ...]:
    leaf = path[-1]
    parent = leaf.parent
    if parent is None:
        return (leaf,)

    index = _find_child_index(parent, leaf)
    start = max(0, index - config.sibling_window)
    stop = min(len(parent.children), index + config.sibling_window + 1)
    return tuple(parent.children[start:stop])


def _format_context(
    path: Sequence[TreeNode],
    config: RetrievalConfig,
) -> tuple[str, tuple[TreeNode, ...], tuple[TreeNode, ...]]:
    leaf = path[-1]
    ancestor_nodes = _selected_ancestors(path, config)
    sibling_nodes = _selected_siblings(path, config)

    blocks: list[str] = []

    if ancestor_nodes:
        blocks.append("Ancestor summaries:")
        for depth, ancestor in enumerate(ancestor_nodes, start=1):
            summary = ancestor.summary.strip() or "(no summary available)"
            blocks.append(f"{depth}. {ancestor.title}: {summary}")

    blocks.append(f"Selected leaf: {leaf.title}")
    leaf_text = leaf.content.strip() or leaf.summary.strip() or "(empty leaf)"
    blocks.append(leaf_text)

    if sibling_nodes:
        blocks.append("Sibling context:")
        for sibling in sibling_nodes:
            label = "selected" if sibling is leaf else "neighbor"
            sibling_text = sibling.content.strip() or sibling.summary.strip() or "(empty section)"
            blocks.append(f"- {sibling.title} [{label}]: {sibling_text}")

    return "\n".join(blocks), ancestor_nodes, sibling_nodes


def retrieve(
    query: str,
    root: TreeNode,
    router: RouteSelector,
    config: RetrievalConfig | None = None,
) -> RetrievalResult:
    active_config = config or RetrievalConfig()
    node = root
    route_path = [root]

    while node.children:
        choice = router.choose_child_index(query, node, node.children)
        index = _validate_child_choice(choice, len(node.children), node)
        node = node.children[index]
        route_path.append(node)

    context, ancestor_nodes, sibling_nodes = _format_context(route_path, active_config)
    return RetrievalResult(
        query=query,
        leaf=node,
        route_path=tuple(route_path),
        context=context,
        ancestor_nodes=ancestor_nodes,
        sibling_nodes=sibling_nodes,
    )
