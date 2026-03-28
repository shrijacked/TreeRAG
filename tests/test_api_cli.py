from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.support.fake_provider import FakeProvider
from treerag.api import build_index, query_index
from treerag.cli import main
from treerag.config import IndexConfig, ModelConfig, RetrievalConfig
from treerag.models import Section
from treerag.storage import MissingIndexError


def _write_document(tmp_path: Path) -> Path:
    document_path = tmp_path / "jira_runbook.md"
    document_path.write_text(
        (
            "# Jira Incident Runbook\n\n"
            "This document explains severity levels, escalation policy, and notification rules."
        ),
        encoding="utf-8",
    )
    return document_path


def test_build_and_query_index_supports_jira_style_runbook(tmp_path: Path) -> None:
    document_path = _write_document(tmp_path)
    index_path = tmp_path / "jira.index.json"
    provider = FakeProvider(
        segment_responses=[
            [Section(title="Incident Management", content=" ".join(["policy"] * 301))],
            [
                Section(
                    title="Severity Levels",
                    content="Update the status page within five minutes.",
                ),
                Section(
                    title="Escalation Policy",
                    content=(
                        "Page the primary on-call immediately and escalate "
                        "after five minutes."
                    ),
                ),
                Section(
                    title="Notification Rules",
                    content=(
                        "Notify support leadership once engineering acknowledges "
                        "the incident."
                    ),
                ),
            ],
        ],
        summary_responses=[
            "Status updates happen quickly for critical incidents.",
            "Escalations start with the primary on-call and step up after five minutes.",
            "Leadership notifications happen after engineering acknowledgment.",
            "Incident management covers severity, escalation, and communication.",
            "The runbook explains how to coordinate a Sev-1 incident.",
        ],
        route_responses=[0, 1],
        answer_responses=[
            "Page the primary on-call immediately and escalate after five minutes."
        ],
    )

    document_index = build_index(
        document_path,
        index_path,
        IndexConfig(cache_dir=tmp_path / ".cache"),
        model_config=ModelConfig(),
        provider=provider,
    )
    result = query_index(
        "How do Sev-1 escalations work?",
        index_path,
        RetrievalConfig(sibling_window=1, include_ancestor_summaries=True),
        model_config=ModelConfig(),
        provider=provider,
    )

    assert document_index.root.children[0].children[1].title == "Escalation Policy"
    assert result.answer == "Page the primary on-call immediately and escalate after five minutes."
    assert "Severity Levels" in result.context
    assert "Notification Rules" in result.context
    assert result.navigation_path == ["root", "Incident Management", "Escalation Policy"]


def test_query_index_raises_for_missing_index(tmp_path: Path) -> None:
    with pytest.raises(MissingIndexError):
        query_index(
            "What is the escalation path?",
            tmp_path / "missing.json",
            RetrievalConfig(),
            model_config=ModelConfig(),
            provider=FakeProvider(answer_responses=["unused"]),
        )


def test_cli_index_ask_and_inspect_commands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    document_path = _write_document(tmp_path)
    index_path = tmp_path / "jira.index.json"
    provider = FakeProvider(
        segment_responses=[
            [Section(title="Incident Management", content=" ".join(["policy"] * 301))],
            [
                Section(title="Severity Levels", content="Status page update."),
                Section(title="Escalation Policy", content="Page the primary on-call."),
            ],
        ],
        summary_responses=[
            "Status updates happen quickly.",
            "Escalation policy pages the primary on-call.",
            "Incident management covers the operational runbook.",
            "The runbook explains incident response.",
        ],
        route_responses=[0, 1],
        answer_responses=["Page the primary on-call."],
    )

    assert main(
        ["index", str(document_path), str(index_path), "--cache-dir", str(tmp_path / ".cache")],
        provider=provider,
    ) == 0
    index_output = json.loads(capsys.readouterr().out)
    assert index_output["output_path"] == str(index_path)

    assert main(
        ["ask", str(index_path), "Who gets paged first?", "--sibling-window", "1"],
        provider=provider,
    ) == 0
    ask_output = json.loads(capsys.readouterr().out)
    assert ask_output["answer"] == "Page the primary on-call."
    assert ask_output["selected_leaf_title"] == "Escalation Policy"

    assert main(["inspect", str(index_path)]) == 0
    inspect_output = json.loads(capsys.readouterr().out)
    assert inspect_output["child_count"] == 1
