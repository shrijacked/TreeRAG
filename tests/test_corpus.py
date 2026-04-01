from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.support.fake_provider import FakeProvider
from treerag.cli import main
from treerag.config import IndexConfig, ModelConfig, RetrievalConfig
from treerag.corpus import build_corpus, load_corpus, query_corpus
from treerag.errors import ParseError
from treerag.models import Section


def _write_documents(tmp_path: Path) -> tuple[Path, Path]:
    access_path = tmp_path / "access_requests.md"
    access_path.write_text(
        (
            "# Access Requests\n\n"
            "This runbook explains how to grant and revoke internal access."
        ),
        encoding="utf-8",
    )
    incidents_path = tmp_path / "incident_response.md"
    incidents_path.write_text(
        (
            "# Incident Response\n\n"
            "This runbook explains how to escalate incidents and notify responders."
        ),
        encoding="utf-8",
    )
    return access_path, incidents_path


def _incident_section() -> Section:
    return Section(
        title="Incident Response",
        content="Page the incident commander immediately.",
    )


def _write_traceable_corpus_documents(tmp_path: Path) -> tuple[Path, Path]:
    access_path = tmp_path / "access_requests_traceable.md"
    access_path.write_text(
        (
            "# Access Requests\n\n"
            "## Approvals\n"
            "Grant access only after manager approval.\n"
        ),
        encoding="utf-8",
    )
    incidents_path = tmp_path / "incident_response_traceable.md"
    incidents_path.write_text(
        (
            "# Incident Response\n\n"
            "## Incident Command\n"
            "Page the incident commander immediately.\n\n"
            "## Comms\n"
            "Notify the response channel after the page goes out.\n"
        ),
        encoding="utf-8",
    )
    return access_path, incidents_path


def test_build_and_query_corpus_routes_to_the_right_document(tmp_path: Path) -> None:
    access_path, incidents_path = _write_documents(tmp_path)
    corpus_dir = tmp_path / "corpus"
    provider = FakeProvider(
        segment_responses=[
            [Section(title="Access Requests", content="Grant access with manager approval.")],
            [_incident_section()],
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

    corpus = build_corpus(
        [access_path, incidents_path],
        corpus_dir,
        IndexConfig(cache_dir=tmp_path / ".cache"),
        model_config=ModelConfig(),
        provider=provider,
    )
    result = query_corpus(
        "How do incidents escalate?",
        corpus_dir,
        RetrievalConfig(),
        model_config=ModelConfig(),
        provider=provider,
    )

    assert len(corpus.documents) == 2
    assert corpus.documents[1].title == "Incident Response"
    assert result.document_title == "Incident Response"
    assert result.selected_leaf_title == "Incident Response"
    assert result.answer == "Page the incident commander immediately."


def test_query_corpus_preserves_traceable_source_references(tmp_path: Path) -> None:
    access_path, incidents_path = _write_traceable_corpus_documents(tmp_path)
    corpus_dir = tmp_path / "traceable-corpus"
    provider = FakeProvider(
        segment_responses=[
            [
                Section(
                    title="Approvals",
                    content="Grant access only after manager approval.",
                )
            ],
            [
                Section(
                    title="Incident Command",
                    content="Page the incident commander immediately.",
                ),
                Section(
                    title="Comms",
                    content="Notify the response channel after the page goes out.",
                ),
            ],
        ],
        summary_responses=[
            "Grant access only after manager approval.",
            "Access requests are controlled through manager approval.",
            "Page the incident commander immediately.",
            "Notify the response channel after the page goes out.",
            "Incident response explains command and communication flow.",
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
    result = query_corpus(
        "Who coordinates the response first?",
        corpus_dir,
        RetrievalConfig(sibling_window=1, include_ancestor_summaries=False),
        model_config=ModelConfig(),
        provider=provider,
    )

    assert result.source_path == str(incidents_path.resolve())
    assert result.selected_source_span is not None
    assert result.selected_source_span.start_line == 3
    assert result.selected_source_span.end_line == 4
    assert [
        (reference.title, reference.start_line, reference.end_line)
        for reference in result.source_references
    ] == [
        ("Incident Command", 3, 4),
        ("Comms", 6, 7),
    ]


def test_load_corpus_accepts_directory_or_manifest_path(tmp_path: Path) -> None:
    access_path, incidents_path = _write_documents(tmp_path)
    corpus_dir = tmp_path / "corpus"
    provider = FakeProvider(
        segment_responses=[
            [Section(title="Access Requests", content="Grant access with manager approval.")],
            [_incident_section()],
        ],
        summary_responses=[
            "Grant access with manager approval.",
            "This document explains access request handling.",
            "Page the incident commander immediately.",
            "This document explains incident escalation.",
        ],
    )

    build_corpus(
        [access_path, incidents_path],
        corpus_dir,
        IndexConfig(cache_dir=tmp_path / ".cache"),
        model_config=ModelConfig(),
        provider=provider,
    )

    loaded_from_dir = load_corpus(corpus_dir)
    loaded_from_file = load_corpus(corpus_dir / "corpus.json")

    assert [doc.title for doc in loaded_from_dir.documents] == [
        "Access Requests",
        "Incident Response",
    ]
    assert loaded_from_file.documents[0].source_path.endswith("access_requests.md")


def test_build_corpus_prefers_document_heading_for_document_title(tmp_path: Path) -> None:
    jira_path = tmp_path / "jira_runbook.md"
    jira_path.write_text(
        (
            "# Jira Incident Runbook\n\n"
            "## Severity Levels\n\n"
            "Update the status page within five minutes.\n\n"
            "## Escalation Policy\n\n"
            "Page the primary on-call immediately."
        ),
        encoding="utf-8",
    )
    handbook_path = tmp_path / "oncall_handbook.md"
    handbook_path.write_text(
        (
            "# On-Call Handbook\n\n"
            "## Incident Command\n\n"
            "The incident commander coordinates responders."
        ),
        encoding="utf-8",
    )
    corpus_dir = tmp_path / "corpus"
    provider = FakeProvider(
        segment_responses=[
            [
                Section(
                    title="Severity Levels",
                    content="Update the status page within five minutes.",
                ),
                Section(
                    title="Escalation Policy",
                    content="Page the primary on-call immediately.",
                ),
            ],
            [
                Section(
                    title="Incident Command",
                    content="The incident commander coordinates responders.",
                )
            ],
        ],
        summary_responses=[
            "Status page updates happen within five minutes.",
            "Page the primary on-call immediately.",
            "The runbook explains how to respond to incidents.",
            "The incident commander coordinates responders.",
            "The handbook explains on-call ownership.",
        ],
    )

    corpus = build_corpus(
        [jira_path, handbook_path],
        corpus_dir,
        IndexConfig(cache_dir=tmp_path / ".cache"),
        model_config=ModelConfig(),
        provider=provider,
    )

    assert [document.title for document in corpus.documents] == [
        "Jira Incident Runbook",
        "On-Call Handbook",
    ]


def test_query_corpus_raises_for_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(ParseError):
        query_corpus(
            "How do incidents escalate?",
            tmp_path / "missing-corpus",
            RetrievalConfig(),
            model_config=ModelConfig(),
            provider=FakeProvider(answer_responses=["unused"]),
        )


def test_cli_corpus_commands_output_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    access_path, incidents_path = _write_documents(tmp_path)
    corpus_dir = tmp_path / "corpus"
    provider = FakeProvider(
        segment_responses=[
            [Section(title="Access Requests", content="Grant access with manager approval.")],
            [_incident_section()],
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

    assert (
        main(
            [
                "corpus-index",
                str(corpus_dir),
                str(access_path),
                str(incidents_path),
                "--cache-dir",
                str(tmp_path / ".cache"),
            ],
            provider=provider,
        )
        == 0
    )
    index_output = json.loads(capsys.readouterr().out)
    assert index_output["document_count"] == 2

    assert (
        main(
            ["corpus-ask", str(corpus_dir), "How do incidents escalate?"],
            provider=provider,
        )
        == 0
    )
    ask_output = json.loads(capsys.readouterr().out)
    assert ask_output["document_title"] == "Incident Response"
    assert ask_output["answer"] == "Page the incident commander immediately."

    assert main(["corpus-inspect", str(corpus_dir)]) == 0
    inspect_output = json.loads(capsys.readouterr().out)
    assert inspect_output["document_count"] == 2
    assert inspect_output["document_titles"] == ["Access Requests", "Incident Response"]
