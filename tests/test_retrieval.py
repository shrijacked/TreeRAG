from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pytest

from treerag.retrieval import (
    InvalidRouteChoiceError,
    RetrievalConfig,
    RetrievalResult,
    TreeNode,
    retrieve,
)


@dataclass
class FakeRouter:
    choices: list[int]

    def choose_child_index(self, query: str, node: TreeNode, children: Sequence[TreeNode]) -> int:
        del query, node, children
        return self.choices.pop(0)


def _leaf(title: str, content: str, summary: str) -> TreeNode:
    return TreeNode(title=title, content=content, summary=summary)


def test_retrieve_includes_sibling_window_and_ancestor_summaries() -> None:
    overview = TreeNode(title="root", summary="Root overview")
    section = TreeNode(title="Policies", summary="Policies summary", parent=overview)
    overview.children = (section,)

    first = _leaf("General", "general details", "general summary")
    selected = _leaf("Escalations", "escalation steps", "selected summary")
    third = _leaf("Exceptions", "exception notes", "exceptions summary")
    section.children = (first, selected, third)
    first.parent = section
    selected.parent = section
    third.parent = section

    result = retrieve(
        "How do escalations work?",
        overview,
        FakeRouter([0, 1]),
        RetrievalConfig(
            sibling_window=1,
            include_ancestor_summaries=True,
            ancestor_summary_depth=2,
        ),
    )

    assert isinstance(result, RetrievalResult)
    assert result.leaf is selected
    assert "Root overview" in result.context
    assert "Policies summary" in result.context
    assert "general details" in result.context
    assert "escalation steps" in result.context
    assert "exception notes" in result.context
    assert "Selected leaf" in result.context


def test_invalid_route_choice_raises_custom_error() -> None:
    overview = TreeNode(title="root", summary="Root overview")
    section = TreeNode(title="Policies", summary="Policies summary", parent=overview)
    overview.children = (section,)
    section.children = (
        _leaf("General", "general details", "general summary"),
        _leaf("Escalations", "escalation steps", "selected summary"),
    )
    for child in section.children:
        child.parent = section

    with pytest.raises(InvalidRouteChoiceError):
        retrieve(
            "How do escalations work?",
            overview,
            FakeRouter([9]),
            RetrievalConfig(sibling_window=1, include_ancestor_summaries=False),
        )
