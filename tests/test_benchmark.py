from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.support.fake_provider import FakeProvider
from treerag.benchmark import load_benchmark_cases, run_benchmark
from treerag.cli import main
from treerag.config import IndexConfig, ModelConfig, RetrievalConfig
from treerag.models import Section


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


def _write_cases(tmp_path: Path) -> Path:
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "name": "escalation_case",
                        "question": "How do Sev-1 escalations work?",
                        "expected_leaf_title": "Escalation Policy",
                        "expected_answer_substring": "primary on-call",
                    },
                    {
                        "name": "notification_case",
                        "question": "When should support leadership be notified?",
                        "expected_leaf_title": "Notification Rules",
                        "expected_answer_substring": "engineering acknowledges",
                    },
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return cases_path


def test_load_benchmark_cases_parses_json_fixture(tmp_path: Path) -> None:
    cases_path = _write_cases(tmp_path)

    cases = load_benchmark_cases(cases_path)

    assert [case.name for case in cases] == ["escalation_case", "notification_case"]
    assert cases[0].expected_leaf_title == "Escalation Policy"


def test_run_benchmark_reports_pass_and_failure_counts(tmp_path: Path) -> None:
    document_path = _write_document(tmp_path)
    cases_path = _write_cases(tmp_path)
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
                    content="Page the primary on-call immediately and escalate after five minutes.",
                ),
                Section(
                    title="Notification Rules",
                    content="Notify support leadership once engineering acknowledges the incident.",
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
        route_responses=[0, 1, 0, 2],
        answer_responses=[
            "Page the primary on-call immediately and escalate after five minutes.",
            "Support leadership should be notified after engineering acknowledges the incident.",
        ],
    )

    report = run_benchmark(
        document_path,
        cases_path,
        index_path,
        IndexConfig(cache_dir=tmp_path / ".cache"),
        RetrievalConfig(sibling_window=1, include_ancestor_summaries=True),
        model_config=ModelConfig(),
        provider=provider,
    )

    assert report.case_count == 2
    assert report.passed_count == 2
    assert report.failed_count == 0
    assert report.build_duration_ms >= 0
    assert report.total_query_duration_ms >= 0
    assert all(case.passed for case in report.case_results)


def test_benchmark_cli_outputs_summary_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    document_path = _write_document(tmp_path)
    cases_path = _write_cases(tmp_path)
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
                    content="Page the primary on-call immediately and escalate after five minutes.",
                ),
                Section(
                    title="Notification Rules",
                    content="Notify support leadership once engineering acknowledges the incident.",
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
        route_responses=[0, 1, 0, 2],
        answer_responses=[
            "Page the primary on-call immediately and escalate after five minutes.",
            "Support leadership should be notified after engineering acknowledges the incident.",
        ],
    )

    assert (
        main(
            [
                "benchmark",
                str(document_path),
                str(cases_path),
                "--index-path",
                str(index_path),
                "--cache-dir",
                str(tmp_path / ".cache"),
            ],
            provider=provider,
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["passed_count"] == 2
    assert output["failed_count"] == 0
    assert output["case_count"] == 2
