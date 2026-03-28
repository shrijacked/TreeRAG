"""Hierarchical retrieval and context assembly."""

from __future__ import annotations

from dataclasses import dataclass, field

from treerag.config import ModelConfig, RetrievalConfig
from treerag.errors import RoutingError
from treerag.models import PageNode
from treerag.provider import LLMProvider, RouteChoice


class InvalidRouteChoiceError(RoutingError):
    """Raised when a provider returns a route choice that cannot be applied safely."""

    def __init__(self, choice: object, child_count: int, node_title: str) -> None:
        super().__init__(
            f"Invalid route choice {choice!r} for node {node_title!r} with {child_count} children."
        )
        self.choice = choice
        self.child_count = child_count
        self.node_title = node_title


@dataclass(frozen=True)
class RetrievalResult:
    """Structured retrieval metadata returned by tree navigation."""

    leaf: PageNode
    route_path: tuple[PageNode, ...]
    context: str
    included_nodes: tuple[PageNode, ...] = field(default_factory=tuple)


def retrieve(
    question: str,
    *,
    root: PageNode,
    provider: LLMProvider,
    model_config: ModelConfig,
    config: RetrievalConfig | None = None,
) -> RetrievalResult:
    active_config = config or RetrievalConfig()
    node = root
    route_path = [root]

    while node.children:
        choices = [RouteChoice(title=child.title, summary=child.summary) for child in node.children]
        index = provider.route(
            question,
            node_title=node.title,
            choices=choices,
            model_config=model_config,
        )
        child_index = _validate_choice(index, len(node.children), node.title)
        node = node.children[child_index]
        route_path.append(node)

    context, included_nodes = assemble_context(node, config=active_config)
    return RetrievalResult(
        leaf=node,
        route_path=tuple(route_path),
        context=context,
        included_nodes=tuple(included_nodes),
    )


def retrieve_leaf(
    question: str,
    *,
    root: PageNode,
    provider: LLMProvider,
    model_config: ModelConfig,
) -> PageNode:
    return retrieve(
        question,
        root=root,
        provider=provider,
        model_config=model_config,
        config=RetrievalConfig(),
    ).leaf


def assemble_context(leaf: PageNode, *, config: RetrievalConfig) -> tuple[str, list[PageNode]]:
    ancestor_nodes = _ancestor_nodes(leaf) if config.include_ancestor_summaries else []
    sibling_nodes = _sibling_window(leaf, config.sibling_window)

    blocks: list[str] = []
    included_nodes: list[PageNode] = []

    if ancestor_nodes:
        blocks.append("Ancestor summaries:")
        for index, ancestor in enumerate(ancestor_nodes, start=1):
            summary = ancestor.summary.strip() or "(no summary available)"
            blocks.append(f"{index}. {ancestor.title}: {summary}")
            included_nodes.append(ancestor)

    blocks.append(f"Selected leaf: {leaf.title}")
    leaf_text = leaf.content.strip() or leaf.summary.strip() or "(empty leaf)"
    blocks.append(leaf_text)

    if sibling_nodes:
        blocks.append("Sibling context:")
        for sibling in sibling_nodes:
            label = "selected" if sibling is leaf else "neighbor"
            sibling_text = sibling.content.strip() or sibling.summary.strip() or "(empty section)"
            blocks.append(f"- {sibling.title} [{label}]: {sibling_text}")
            if sibling not in included_nodes:
                included_nodes.append(sibling)

    return "\n".join(blocks), included_nodes


def _validate_choice(choice: object, child_count: int, node_title: str) -> int:
    if isinstance(choice, bool) or not isinstance(choice, int):
        raise InvalidRouteChoiceError(choice, child_count, node_title)
    if choice < 0 or choice >= child_count:
        raise InvalidRouteChoiceError(choice, child_count, node_title)
    return choice


def _ancestor_nodes(node: PageNode) -> list[PageNode]:
    ancestors: list[PageNode] = []
    current = node.parent
    while current is not None:
        ancestors.append(current)
        current = current.parent
    ancestors.reverse()
    return ancestors


def _sibling_window(node: PageNode, window: int) -> list[PageNode]:
    parent = node.parent
    if parent is None:
        return [node]

    index = _find_child_index(parent, node)
    start = max(0, index - window)
    stop = min(len(parent.children), index + window + 1)
    return parent.children[start:stop]


def _find_child_index(parent: PageNode, child: PageNode) -> int:
    for index, candidate in enumerate(parent.children):
        if candidate is child:
            return index
    raise RoutingError(f"Node {child.node_id!r} is not attached to parent {parent.node_id!r}.")
