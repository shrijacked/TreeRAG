from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


REFERENCE_ROOT = Path(__file__).resolve().parents[2] / ".reference" / "pageindex-rag"


def _load_reference_module(module_name: str) -> Any:
    previous_api_key = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = "test-key"
    sys.path.insert(0, str(REFERENCE_ROOT))
    try:
        for loaded_name in list(sys.modules):
            if loaded_name == "pageindex" or loaded_name.startswith("pageindex."):
                sys.modules.pop(loaded_name, None)
        return importlib.import_module(module_name)
    finally:
        sys.path.pop(0)
        if previous_api_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = previous_api_key


def _fake_chat_response(content: str) -> SimpleNamespace:
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _fake_client(content: str, capture: list[dict[str, Any]] | None = None) -> SimpleNamespace:
    def create(**kwargs: Any) -> SimpleNamespace:
        if capture is not None:
            capture.append(kwargs)
        return _fake_chat_response(content)

    completions = SimpleNamespace(create=create)
    chat = SimpleNamespace(completions=completions)
    return SimpleNamespace(chat=chat)


def test_reference_router_defaults_to_first_child_on_invalid_response() -> None:
    node_module = _load_reference_module("pageindex.node")
    retriever_module = _load_reference_module("pageindex.retriever")

    first = node_module.PageNode(
        title="First",
        content="first leaf",
        summary="first summary",
        depth=1,
    )
    second = node_module.PageNode(
        title="Second",
        content="second leaf",
        summary="second summary",
        depth=1,
    )
    root = node_module.PageNode(
        title="root",
        content="",
        summary="",
        depth=0,
        children=[first, second],
    )

    retriever_module.client = _fake_client("not-a-number")

    picked = retriever_module._pick_child("where is the answer?", root)

    assert picked is first


def test_reference_retrieve_returns_only_selected_leaf_content() -> None:
    node_module = _load_reference_module("pageindex.node")
    retriever_module = _load_reference_module("pageindex.retriever")

    first = node_module.PageNode(
        title="First",
        content="before context",
        summary="first summary",
        depth=1,
    )
    second = node_module.PageNode(
        title="Second",
        content="answer context",
        summary="second summary",
        depth=1,
    )
    third = node_module.PageNode(
        title="Third",
        content="after context",
        summary="third summary",
        depth=1,
    )
    root = node_module.PageNode(
        title="root",
        content="",
        summary="",
        depth=0,
        children=[first, second, third],
    )

    retriever_module._pick_child = lambda query, node: second

    selected = retriever_module.retrieve("where is the answer?", root)

    assert selected == "answer context"
    assert "before context" not in selected
    assert "after context" not in selected


def test_reference_parser_crashes_on_malformed_json() -> None:
    parser_module = _load_reference_module("pageindex.parser")
    parser_module.client = _fake_client("{bad json")

    with pytest.raises(json.JSONDecodeError):
        parser_module._segment("Section 1")


def test_reference_parameter_usage_is_inconsistent_between_parser_and_indexer() -> None:
    parser_module = _load_reference_module("pageindex.parser")
    indexer_module = _load_reference_module("pageindex.indexer")

    parser_calls: list[dict[str, Any]] = []
    indexer_calls: list[dict[str, Any]] = []

    parser_module.client = _fake_client('{"sections": []}', parser_calls)
    indexer_module.client = _fake_client("summary text", indexer_calls)

    parser_module._segment("Section 1")
    node_module = _load_reference_module("pageindex.node")
    leaf = node_module.PageNode(title="Leaf", content="Some text", summary="", depth=1)
    indexer_module.build_summaries(leaf)

    assert "max_completion_tokens" in parser_calls[0]
    assert "max_tokens" in indexer_calls[0]


def test_reference_parser_never_recurses_beyond_depth_two(monkeypatch: pytest.MonkeyPatch) -> None:
    parser_module = _load_reference_module("pageindex.parser")

    calls = [
        [
            {
                "title": "Top",
                "content": " ".join(["subsection"] * 301),
            }
        ],
        [
            {
                "title": "Child A",
                "content": " ".join(["grandchild"] * 301),
            },
            {
                "title": "Child B",
                "content": "leaf content",
            },
        ],
    ]

    def fake_segment(_: str) -> list[dict[str, str]]:
        return calls.pop(0)

    monkeypatch.setattr(parser_module, "_segment", fake_segment)

    root = parser_module.parse_document("ignored")

    assert root.children[0].depth == 1
    assert root.children[0].children[0].depth == 2
    assert root.children[0].children[0].content
    assert root.children[0].children[0].children == []
