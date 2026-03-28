from __future__ import annotations

from dataclasses import dataclass

import pytest

from tests.support.fake_provider import FakeProvider
from treerag.config import ModelConfig, RetrievalConfig
from treerag.models import PageNode
from treerag.retrieval import (
    InvalidRouteChoiceError,
    RetrievalResult,
    retrieve,
)


@dataclass
class FakeRoutingProvider(FakeProvider):
    pass


def _leaf(node_id: str, title: str, content: str, summary: str, depth: int) -> PageNode:
    return PageNode(node_id=node_id, title=title, content=content, summary=summary, depth=depth)


def test_retrieve_includes_sibling_window_and_ancestor_summaries() -> None:
    overview = PageNode(node_id="root", title="root", content="", summary="Root overview", depth=0)
    section = PageNode(
        node_id="root.0",
        title="Policies",
        content="",
        summary="Policies summary",
        depth=1,
    )
    overview.set_children([section])

    first = _leaf("root.0.0", "General", "general details", "general summary", 2)
    selected = _leaf("root.0.1", "Escalations", "escalation steps", "selected summary", 2)
    third = _leaf("root.0.2", "Exceptions", "exception notes", "exceptions summary", 2)
    section.set_children([first, selected, third])
    provider = FakeRoutingProvider(route_responses=[0, 1])

    result = retrieve(
        "How do escalations work?",
        root=overview,
        provider=provider,
        model_config=ModelConfig(),
        config=RetrievalConfig(sibling_window=1, include_ancestor_summaries=True),
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
    overview = PageNode(node_id="root", title="root", content="", summary="Root overview", depth=0)
    section = PageNode(
        node_id="root.0",
        title="Policies",
        content="",
        summary="Policies summary",
        depth=1,
    )
    overview.set_children([section])
    section.set_children(
        [
            _leaf("root.0.0", "General", "general details", "general summary", 2),
            _leaf("root.0.1", "Escalations", "escalation steps", "selected summary", 2),
        ]
    )
    provider = FakeRoutingProvider(route_responses=[9])

    with pytest.raises(InvalidRouteChoiceError):
        retrieve(
            "How do escalations work?",
            root=overview,
            provider=provider,
            model_config=ModelConfig(),
            config=RetrievalConfig(sibling_window=1, include_ancestor_summaries=False),
        )
