from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.support.fake_provider import FakeProvider
from treerag.api import build_index, query_index
from treerag.cli import main
from treerag.config import IndexConfig, ModelConfig, RetrievalConfig
from treerag.corpus import build_corpus
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


def _write_traceable_document(tmp_path: Path) -> Path:
    document_path = tmp_path / "traceable_runbook.md"
    document_path.write_text(
        (
            "# Jira Incident Runbook\n\n"
            "## Severity Levels\n"
            "Update the status page within five minutes.\n\n"
            "## Escalation Policy\n"
            "Page the primary on-call immediately and escalate after five minutes.\n\n"
            "## Notification Rules\n"
            "Notify support leadership after engineering acknowledges the incident.\n"
        ),
        encoding="utf-8",
    )
    return document_path


def _write_corpus_documents(tmp_path: Path) -> tuple[Path, Path]:
    access_path = tmp_path / "access_requests.md"
    access_path.write_text(
        "# Access Requests\n\nGrant access with manager approval.",
        encoding="utf-8",
    )
    incidents_path = tmp_path / "incident_response.md"
    incidents_path.write_text(
        "# Incident Response\n\nPage the incident commander immediately.",
        encoding="utf-8",
    )
    return access_path, incidents_path


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


def test_paraphrased_query_still_routes_to_the_same_section(tmp_path: Path) -> None:
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

    build_index(
        document_path,
        index_path,
        IndexConfig(cache_dir=tmp_path / ".cache"),
        model_config=ModelConfig(),
        provider=provider,
    )
    result = query_index(
        "who gets alerted first during a critical outage?",
        index_path,
        RetrievalConfig(sibling_window=1, include_ancestor_summaries=True),
        model_config=ModelConfig(),
        provider=provider,
    )

    assert result.selected_leaf_title == "Escalation Policy"
    assert "primary on-call" in result.answer


def test_query_index_returns_traceable_source_references(tmp_path: Path) -> None:
    document_path = _write_traceable_document(tmp_path)
    index_path = tmp_path / "traceable.index.json"
    provider = FakeProvider(
        segment_responses=[
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
                        "Notify support leadership after engineering acknowledges "
                        "the incident."
                    ),
                ),
            ]
        ],
        summary_responses=[
            "Status updates happen quickly for critical incidents.",
            "Escalations start with the primary on-call and step up after five minutes.",
            "Leadership notifications happen after engineering acknowledgment.",
            "The runbook explains how to coordinate a Sev-1 incident.",
        ],
        route_responses=[1],
        answer_responses=[
            "Page the primary on-call immediately and escalate after five minutes."
        ],
    )

    build_index(
        document_path,
        index_path,
        IndexConfig(cache_dir=tmp_path / ".cache"),
        model_config=ModelConfig(),
        provider=provider,
    )
    result = query_index(
        "How do Sev-1 escalations work?",
        index_path,
        RetrievalConfig(sibling_window=1, include_ancestor_summaries=False),
        model_config=ModelConfig(),
        provider=provider,
    )

    assert result.source_path == str(document_path)
    assert result.selected_source_span is not None
    assert result.selected_source_span.start_line == 6
    assert result.selected_source_span.end_line == 7
    assert [
        (reference.title, reference.start_line, reference.end_line)
        for reference in result.source_references
    ] == [
        ("Severity Levels", 3, 4),
        ("Escalation Policy", 6, 7),
        ("Notification Rules", 9, 10),
    ]


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


def test_cli_ask_outputs_source_references(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    document_path = _write_traceable_document(tmp_path)
    index_path = tmp_path / "traceable.index.json"
    provider = FakeProvider(
        segment_responses=[
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
                        "Notify support leadership after engineering acknowledges "
                        "the incident."
                    ),
                ),
            ]
        ],
        summary_responses=[
            "Status updates happen quickly for critical incidents.",
            "Escalations start with the primary on-call and step up after five minutes.",
            "Leadership notifications happen after engineering acknowledgment.",
            "The runbook explains how to coordinate a Sev-1 incident.",
        ],
        route_responses=[1],
        answer_responses=[
            "Page the primary on-call immediately and escalate after five minutes."
        ],
    )

    assert main(
        ["index", str(document_path), str(index_path), "--cache-dir", str(tmp_path / ".cache")],
        provider=provider,
    ) == 0
    capsys.readouterr()

    assert main(
        ["ask", str(index_path), "How do Sev-1 escalations work?", "--sibling-window", "1"],
        provider=provider,
    ) == 0
    ask_output = json.loads(capsys.readouterr().out)

    assert ask_output["source_path"] == str(document_path)
    assert ask_output["selected_source_span"]["start_line"] == 6
    assert ask_output["selected_source_span"]["end_line"] == 7
    assert ask_output["source_references"] == [
        {"title": "Severity Levels", "start_line": 3, "end_line": 4},
        {"title": "Escalation Policy", "start_line": 6, "end_line": 7},
        {"title": "Notification Rules", "start_line": 9, "end_line": 10},
    ]


def test_cli_repl_answers_until_quit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
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

    build_index(
        document_path,
        index_path,
        IndexConfig(cache_dir=tmp_path / ".cache"),
        model_config=ModelConfig(),
        provider=provider,
    )

    questions = iter(["Who gets paged first?", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(questions))

    assert main(["repl", str(index_path), "--sibling-window", "1"], provider=provider) == 0
    output = capsys.readouterr().out
    assert "Escalation Policy" in output
    assert "Page the primary on-call." in output


def test_cli_corpus_repl_answers_until_exit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    access_path, incidents_path = _write_corpus_documents(tmp_path)
    corpus_dir = tmp_path / "corpus"
    provider = FakeProvider(
        segment_responses=[
            [Section(title="Access Requests", content="Grant access with manager approval.")],
            [
                Section(
                    title="Incident Response",
                    content="Page the incident commander immediately.",
                )
            ],
        ],
        summary_responses=[
            "Grant access with manager approval.",
            "This document explains access request handling.",
            "Page the incident commander immediately.",
            "This document explains incident escalation.",
        ],
        route_responses=[1, 0],
        answer_responses=["Page the incident commander immediately."],
    )

    build_corpus(
        [access_path, incidents_path],
        corpus_dir,
        IndexConfig(cache_dir=tmp_path / ".cache"),
        model_config=ModelConfig(),
        provider=provider,
    )

    questions = iter(["How do incidents escalate?", "exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(questions))

    assert main(["corpus-repl", str(corpus_dir)], provider=provider) == 0
    output = capsys.readouterr().out
    assert "Incident Response" in output
    assert "Page the incident commander immediately." in output
