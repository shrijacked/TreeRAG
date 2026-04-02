from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.support.fake_provider import FakeProvider
from treerag.benchmark import (
    load_benchmark_cases,
    run_benchmark,
    run_comparison_benchmark,
    run_corpus_benchmark,
    run_corpus_comparison_benchmark,
)
from treerag.cli import main
from treerag.config import IndexConfig, ModelConfig, RetrievalConfig
from treerag.models import Section
from treerag.provider import TokenUsage


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


def _write_corpus_cases(tmp_path: Path) -> Path:
    cases_path = tmp_path / "corpus-cases.json"
    cases_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "name": "incident_escalation",
                        "question": "How do incidents escalate?",
                        "expected_document_title": "Incident Response",
                        "expected_leaf_title": "Incident Response",
                        "expected_answer_substring": "incident commander",
                    },
                    {
                        "name": "access_grants",
                        "question": "How do we grant access?",
                        "expected_document_title": "Access Requests",
                        "expected_leaf_title": "Access Requests",
                        "expected_answer_substring": "manager approval",
                    },
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return cases_path


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


def _write_corpus_comparison_documents(tmp_path: Path) -> tuple[Path, Path]:
    updates_path = tmp_path / "critical_outage_updates.md"
    updates_path.write_text(
        (
            "# Critical Outage Updates\n\n"
            "During a critical outage, responders share status updates in Slack."
        ),
        encoding="utf-8",
    )
    handbook_path = tmp_path / "oncall_handbook.md"
    handbook_path.write_text(
        (
            "# On-Call Handbook\n\n"
            "The incident commander runs the room and coordinates responders during a Sev-1."
        ),
        encoding="utf-8",
    )
    return updates_path, handbook_path


def _incident_response_section() -> Section:
    return Section(
        title="Incident Response",
        content="Page the incident commander immediately.",
    )


def _write_comparison_document(tmp_path: Path) -> Path:
    document_path = tmp_path / "finance_review.md"
    document_path.write_text(
        (
            "# Q3 Finance Review\n\n"
            "## Executive Summary\n"
            "Debt trends are discussed later in the report.\n\n"
            "## Liquidity Overview\n"
            "In Q3, management said leverage was improving and debt schedules were being "
            "simplified.\n\n"
            "## Appendix G - Debt Schedule\n"
            "At September 30, short-term borrowings were $61 million versus $84 million at "
            "June 30.\n"
        ),
        encoding="utf-8",
    )
    return document_path


def _write_comparison_cases(tmp_path: Path) -> Path:
    cases_path = tmp_path / "comparison-cases.json"
    cases_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "name": "debt_trends",
                        "question": "what were the debt trends in q3?",
                        "expected_leaf_title": "Appendix G - Debt Schedule",
                        "expected_answer_substring": "$61 million",
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return cases_path


def _write_corpus_comparison_cases(tmp_path: Path) -> Path:
    cases_path = tmp_path / "corpus-comparison-cases.json"
    cases_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "name": "incident_command_routing",
                        "question": "who runs the room during a critical outage?",
                        "expected_document_title": "On-Call Handbook",
                        "expected_leaf_title": "On-Call Handbook",
                        "expected_answer_substring": "incident commander",
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return cases_path


@dataclass
class ContextAwareComparisonProvider(FakeProvider):
    def answer(self, question: str, *, context: str, model_config: ModelConfig) -> str:
        self.answer_calls += 1
        lowered = context.lower()
        if "$61 million" in context:
            return "Short-term borrowings fell to $61 million from $84 million."
        if "leverage was improving" in lowered:
            return "Management said leverage was improving."
        return "I could not find the debt schedule."


@dataclass
class ContextAwareCorpusComparisonProvider(FakeProvider):
    def answer(self, question: str, *, context: str, model_config: ModelConfig) -> str:
        self.answer_calls += 1
        lowered = context.lower()
        if "incident commander" in lowered:
            return "The incident commander runs the room during a critical outage."
        if "status updates in slack" in lowered:
            return "Responders share status updates in Slack."
        return "I could not identify who runs the room."


def test_load_benchmark_cases_parses_json_fixture(tmp_path: Path) -> None:
    cases_path = _write_cases(tmp_path)

    cases = load_benchmark_cases(cases_path)

    assert [case.name for case in cases] == ["escalation_case", "notification_case"]
    assert cases[0].expected_leaf_title == "Escalation Policy"


def test_load_benchmark_cases_keeps_expected_document_title(tmp_path: Path) -> None:
    cases_path = _write_corpus_cases(tmp_path)

    cases = load_benchmark_cases(cases_path)

    assert cases[0].expected_document_title == "Incident Response"


def test_packaged_benchmark_fixtures_load_from_repo() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    jira_cases = load_benchmark_cases(repo_root / "benchmarks" / "jira_cases.json")
    access_cases = load_benchmark_cases(repo_root / "benchmarks" / "access_cases.json")
    appendix_cases = load_benchmark_cases(repo_root / "benchmarks" / "appendix_cases.json")
    comparison_cases = load_benchmark_cases(repo_root / "benchmarks" / "comparison_cases.json")
    corpus_comparison_cases = load_benchmark_cases(
        repo_root / "benchmarks" / "corpus_comparison_cases.json"
    )
    paraphrase_cases = load_benchmark_cases(
        repo_root / "benchmarks" / "paraphrase_cases.json"
    )
    corpus_cases = load_benchmark_cases(
        repo_root / "benchmarks" / "operations_corpus_cases.json"
    )

    assert [case.name for case in jira_cases] == ["sev1_escalation", "sev1_notifications"]
    assert [case.name for case in access_cases] == [
        "emergency_access",
        "access_revocation",
    ]
    assert [case.name for case in appendix_cases] == [
        "q3_debt_trends",
        "borrowing_direction",
        "covenant_threshold_change",
    ]
    assert [case.expected_leaf_title for case in appendix_cases] == [
        "Appendix G - Debt Schedule",
        "Appendix G - Debt Schedule",
        "Appendix H - Covenant Notes",
    ]
    assert [case.name for case in comparison_cases] == ["q3_debt_trends_compare"]
    assert [case.name for case in corpus_comparison_cases] == ["incident_command_compare"]
    assert [case.name for case in paraphrase_cases] == [
        "critical_outage_alerting",
        "customer_comms_timing",
    ]
    assert [case.expected_document_title for case in corpus_cases] == [
        "Jira Incident Runbook",
        "On-Call Handbook",
        "Access Management Runbook",
    ]


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


def test_run_benchmark_reports_usage_and_estimated_cost(tmp_path: Path) -> None:
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
        segment_usages=[
            TokenUsage(requests=1, input_tokens=100, output_tokens=20, total_tokens=120)
        ],
        summary_usages=[
            TokenUsage(requests=1, input_tokens=80, output_tokens=10, total_tokens=90),
            TokenUsage(requests=1, input_tokens=80, output_tokens=10, total_tokens=90),
            TokenUsage(requests=1, input_tokens=80, output_tokens=10, total_tokens=90),
            TokenUsage(requests=1, input_tokens=80, output_tokens=10, total_tokens=90),
            TokenUsage(requests=1, input_tokens=80, output_tokens=10, total_tokens=90),
        ],
        route_usages=[
            TokenUsage(requests=1, input_tokens=30, output_tokens=1, total_tokens=31),
            TokenUsage(requests=1, input_tokens=30, output_tokens=1, total_tokens=31),
            TokenUsage(requests=1, input_tokens=30, output_tokens=1, total_tokens=31),
            TokenUsage(requests=1, input_tokens=30, output_tokens=1, total_tokens=31),
        ],
        answer_usages=[
            TokenUsage(requests=1, input_tokens=60, output_tokens=20, total_tokens=80),
            TokenUsage(requests=1, input_tokens=60, output_tokens=20, total_tokens=80),
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

    assert report.total_usage is not None
    assert report.total_cost_estimate is not None
    assert report.total_usage.total.requests == 13
    assert report.total_cost_estimate.total_cost_usd is not None
    assert report.total_cost_estimate.total_cost_usd > 0
    assert report.case_results[0].usage is not None
    assert report.case_results[0].cost_estimate is not None


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


def test_run_comparison_benchmark_reports_method_level_results(tmp_path: Path) -> None:
    document_path = _write_comparison_document(tmp_path)
    cases_path = _write_comparison_cases(tmp_path)
    index_path = tmp_path / "finance.index.json"
    provider = ContextAwareComparisonProvider(
        segment_responses=[
            [
                Section(
                    title="Executive Summary",
                    content="Debt trends are discussed later in the report.",
                ),
                Section(
                    title="Liquidity Overview",
                    content=(
                        "In Q3, management said leverage was improving and debt schedules "
                        "were being simplified."
                    ),
                ),
                Section(
                    title="Appendix G - Debt Schedule",
                    content=(
                        "At September 30, short-term borrowings were $61 million versus "
                        "$84 million at June 30."
                    ),
                ),
            ]
        ],
        summary_responses=[
            "Debt trends are discussed later in the report.",
            "Management said leverage was improving in Q3.",
            "Short-term borrowings fell to $61 million from $84 million.",
            "The report covers summary, liquidity, and debt schedule details.",
        ],
        route_responses=[2],
    )

    report = run_comparison_benchmark(
        document_path,
        cases_path,
        index_path,
        IndexConfig(cache_dir=tmp_path / ".cache", subsection_word_threshold=999),
        RetrievalConfig(sibling_window=0, include_ancestor_summaries=False),
        model_config=ModelConfig(),
        provider=provider,
    )

    methods = {method.method: method for method in report.methods}
    assert set(methods) == {"tree_rag", "keyword_leaf", "full_context"}
    assert methods["tree_rag"].passed_count == 1
    assert methods["keyword_leaf"].failed_count == 1
    assert methods["full_context"].passed_count == 1


def test_run_comparison_benchmark_normalizes_leaf_title_matches(tmp_path: Path) -> None:
    document_path = _write_comparison_document(tmp_path)
    cases_path = _write_comparison_cases(tmp_path)
    index_path = tmp_path / "finance.index.json"
    provider = FakeProvider(
        segment_responses=[
            [
                Section(
                    title="Executive Summary",
                    content="Debt trends are discussed later in the report.",
                ),
                Section(
                    title="Liquidity Overview",
                    content=(
                        "In Q3, management said leverage was improving and debt schedules "
                        "were being simplified."
                    ),
                ),
                Section(
                    title="Appendix G Debt Schedule",
                    content=(
                        "At September 30, short-term borrowings were $61 million versus "
                        "$84 million at June 30."
                    ),
                ),
            ]
        ],
        summary_responses=[
            "Debt trends are discussed later in the report.",
            "Management said leverage was improving in Q3.",
            "Short-term borrowings fell to $61 million from $84 million.",
            "The report covers summary, liquidity, and debt schedule details.",
        ],
        route_responses=[2],
        answer_responses=["Short-term borrowings fell to $61 million from $84 million."],
    )

    report = run_comparison_benchmark(
        document_path,
        cases_path,
        index_path,
        IndexConfig(cache_dir=tmp_path / ".cache", subsection_word_threshold=999),
        RetrievalConfig(sibling_window=0, include_ancestor_summaries=False),
        model_config=ModelConfig(),
        provider=provider,
        methods=("tree_rag",),
    )

    case_result = report.methods[0].case_results[0]
    assert report.methods[0].passed_count == 1
    assert case_result.leaf_match is True


def test_run_comparison_benchmark_normalizes_answer_substring_matches(
    tmp_path: Path,
) -> None:
    document_path = _write_comparison_document(tmp_path)
    cases_path = _write_comparison_cases(tmp_path)
    index_path = tmp_path / "finance.index.json"
    provider = FakeProvider(
        segment_responses=[
            [
                Section(
                    title="Executive Summary",
                    content="Debt trends are discussed later in the report.",
                ),
                Section(
                    title="Liquidity Overview",
                    content=(
                        "In Q3, management said leverage was improving and debt schedules "
                        "were being simplified."
                    ),
                ),
                Section(
                    title="Appendix G - Debt Schedule",
                    content=(
                        "At September 30, short-term borrowings were $61 million versus "
                        "$84 million at June 30."
                    ),
                ),
            ]
        ],
        summary_responses=[
            "Debt trends are discussed later in the report.",
            "Management said leverage was improving in Q3.",
            "Short-term borrowings fell to $61 million from $84 million.",
            "The report covers summary, liquidity, and debt schedule details.",
        ],
        route_responses=[2],
        answer_responses=["Short term borrowings fell to 61 million from 84 million."],
    )

    report = run_comparison_benchmark(
        document_path,
        cases_path,
        index_path,
        IndexConfig(cache_dir=tmp_path / ".cache", subsection_word_threshold=999),
        RetrievalConfig(sibling_window=0, include_ancestor_summaries=False),
        model_config=ModelConfig(),
        provider=provider,
        methods=("tree_rag",),
    )

    case_result = report.methods[0].case_results[0]
    assert report.methods[0].passed_count == 1
    assert case_result.answer_match is True


def test_run_comparison_benchmark_reports_method_costs(tmp_path: Path) -> None:
    document_path = _write_comparison_document(tmp_path)
    cases_path = _write_comparison_cases(tmp_path)
    index_path = tmp_path / "finance.index.json"
    provider = ContextAwareComparisonProvider(
        segment_responses=[
            [
                Section(
                    title="Executive Summary",
                    content="Debt trends are discussed later in the report.",
                ),
                Section(
                    title="Liquidity Overview",
                    content=(
                        "In Q3, management said leverage was improving and debt schedules "
                        "were being simplified."
                    ),
                ),
                Section(
                    title="Appendix G - Debt Schedule",
                    content=(
                        "At September 30, short-term borrowings were $61 million versus "
                        "$84 million at June 30."
                    ),
                ),
            ]
        ],
        summary_responses=[
            "Debt trends are discussed later in the report.",
            "Management said leverage was improving in Q3.",
            "Short-term borrowings fell to $61 million from $84 million.",
            "The report covers summary, liquidity, and debt schedule details.",
        ],
        route_responses=[2],
        segment_usages=[
            TokenUsage(requests=1, input_tokens=90, output_tokens=15, total_tokens=105)
        ],
        summary_usages=[
            TokenUsage(requests=1, input_tokens=70, output_tokens=8, total_tokens=78),
            TokenUsage(requests=1, input_tokens=70, output_tokens=8, total_tokens=78),
            TokenUsage(requests=1, input_tokens=70, output_tokens=8, total_tokens=78),
            TokenUsage(requests=1, input_tokens=70, output_tokens=8, total_tokens=78),
        ],
        route_usages=[TokenUsage(requests=1, input_tokens=25, output_tokens=1, total_tokens=26)],
        answer_usages=[
            TokenUsage(requests=1, input_tokens=50, output_tokens=10, total_tokens=60),
            TokenUsage(requests=1, input_tokens=40, output_tokens=8, total_tokens=48),
            TokenUsage(requests=1, input_tokens=120, output_tokens=12, total_tokens=132),
        ],
    )

    report = run_comparison_benchmark(
        document_path,
        cases_path,
        index_path,
        IndexConfig(cache_dir=tmp_path / ".cache", subsection_word_threshold=999),
        RetrievalConfig(sibling_window=0, include_ancestor_summaries=False),
        model_config=ModelConfig(),
        provider=provider,
    )

    method = {entry.method: entry for entry in report.methods}["tree_rag"]
    assert method.usage is not None
    assert method.cost_estimate is not None
    assert method.cost_estimate.total_cost_usd is not None
    assert method.case_results[0].usage is not None
    assert report.total_usage is not None


def test_cost_estimate_covers_snapshot_and_preview_model_names(tmp_path: Path) -> None:
    document_path = _write_comparison_document(tmp_path)
    cases_path = _write_comparison_cases(tmp_path)
    index_path = tmp_path / "finance.index.json"
    provider = ContextAwareComparisonProvider(
        segment_responses=[
            [
                Section(
                    title="Executive Summary",
                    content="Debt trends are discussed later in the report.",
                ),
                Section(
                    title="Liquidity Overview",
                    content=(
                        "In Q3, management said leverage was improving and debt schedules "
                        "were being simplified."
                    ),
                ),
                Section(
                    title="Appendix G - Debt Schedule",
                    content=(
                        "At September 30, short-term borrowings were $61 million versus "
                        "$84 million at June 30."
                    ),
                ),
            ]
        ],
        summary_responses=[
            "Debt trends are discussed later in the report.",
            "Management said leverage was improving in Q3.",
            "Short-term borrowings fell to $61 million from $84 million.",
            "The report covers summary, liquidity, and debt schedule details.",
        ],
        route_responses=[2],
        segment_usages=[
            TokenUsage(requests=1, input_tokens=90, output_tokens=15, total_tokens=105)
        ],
        summary_usages=[
            TokenUsage(requests=1, input_tokens=70, output_tokens=8, total_tokens=78),
            TokenUsage(requests=1, input_tokens=70, output_tokens=8, total_tokens=78),
            TokenUsage(requests=1, input_tokens=70, output_tokens=8, total_tokens=78),
            TokenUsage(requests=1, input_tokens=70, output_tokens=8, total_tokens=78),
        ],
        route_usages=[TokenUsage(requests=1, input_tokens=25, output_tokens=1, total_tokens=26)],
        answer_usages=[
            TokenUsage(requests=1, input_tokens=50, output_tokens=10, total_tokens=60),
            TokenUsage(requests=1, input_tokens=40, output_tokens=8, total_tokens=48),
            TokenUsage(requests=1, input_tokens=120, output_tokens=12, total_tokens=132),
        ],
    )

    report = run_comparison_benchmark(
        document_path,
        cases_path,
        index_path,
        IndexConfig(cache_dir=tmp_path / ".cache", subsection_word_threshold=999),
        RetrievalConfig(sibling_window=0, include_ancestor_summaries=False),
        model_config=ModelConfig(
            segmentation_model="gpt-5.4-nano-2026-03-17",
            summarization_model="gpt-5.4-mini-2026-03-17",
            routing_model="gemini-2.5-flash-lite-preview-09-2025",
            answer_model="gemini-2.5-flash-002",
        ),
        provider=provider,
    )

    assert report.total_cost_estimate is not None
    assert report.total_cost_estimate.pricing_complete is True
    assert report.total_cost_estimate.missing_models == ()


def test_compare_cli_outputs_method_summary_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    document_path = _write_comparison_document(tmp_path)
    cases_path = _write_comparison_cases(tmp_path)
    index_path = tmp_path / "finance.index.json"
    provider = ContextAwareComparisonProvider(
        segment_responses=[
            [
                Section(
                    title="Executive Summary",
                    content="Debt trends are discussed later in the report.",
                ),
                Section(
                    title="Liquidity Overview",
                    content=(
                        "In Q3, management said leverage was improving and debt schedules "
                        "were being simplified."
                    ),
                ),
                Section(
                    title="Appendix G - Debt Schedule",
                    content=(
                        "At September 30, short-term borrowings were $61 million versus "
                        "$84 million at June 30."
                    ),
                ),
            ]
        ],
        summary_responses=[
            "Debt trends are discussed later in the report.",
            "Management said leverage was improving in Q3.",
            "Short-term borrowings fell to $61 million from $84 million.",
            "The report covers summary, liquidity, and debt schedule details.",
        ],
        route_responses=[2],
    )

    assert (
        main(
            [
                "compare",
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
    methods = {method["method"]: method for method in output["methods"]}
    assert methods["tree_rag"]["passed_count"] == 1
    assert methods["keyword_leaf"]["failed_count"] == 1


def test_run_comparison_benchmark_repeat_count_records_samples(tmp_path: Path) -> None:
    document_path = _write_comparison_document(tmp_path)
    cases_path = _write_comparison_cases(tmp_path)
    index_path = tmp_path / "finance.index.json"
    provider = ContextAwareComparisonProvider(
        segment_responses=[
            [
                Section(
                    title="Executive Summary",
                    content="Debt trends are discussed later in the report.",
                ),
                Section(
                    title="Liquidity Overview",
                    content=(
                        "In Q3, management said leverage was improving and debt schedules "
                        "were being simplified."
                    ),
                ),
                Section(
                    title="Appendix G - Debt Schedule",
                    content=(
                        "At September 30, short-term borrowings were $61 million versus "
                        "$84 million at June 30."
                    ),
                ),
            ]
        ],
        summary_responses=[
            "Debt trends are discussed later in the report.",
            "Management said leverage was improving in Q3.",
            "Short-term borrowings fell to $61 million from $84 million.",
            "The report covers summary, liquidity, and debt schedule details.",
        ],
        route_responses=[2, 2, 2],
    )

    report = run_comparison_benchmark(
        document_path,
        cases_path,
        index_path,
        IndexConfig(cache_dir=tmp_path / ".cache", subsection_word_threshold=999),
        RetrievalConfig(sibling_window=0, include_ancestor_summaries=False),
        model_config=ModelConfig(),
        provider=provider,
        repeat_count=3,
    )

    method = {entry.method: entry for entry in report.methods}["tree_rag"]
    case_result = method.case_results[0]
    assert method.total_runs == 3
    assert len(case_result.query_samples_ms) == 3
    assert case_result.document_consistent is True
    assert case_result.leaf_consistent is True
    assert case_result.answer_consistent is True


def test_run_corpus_benchmark_reports_document_matches(tmp_path: Path) -> None:
    access_path, incidents_path = _write_corpus_documents(tmp_path)
    cases_path = _write_corpus_cases(tmp_path)
    corpus_path = tmp_path / "runbooks"
    provider = FakeProvider(
        segment_responses=[
            [Section(title="Access Requests", content="Grant access with manager approval.")],
            [_incident_response_section()],
        ],
        summary_responses=[
            "Grant access with manager approval.",
            "This document explains access request handling.",
            "Page the incident commander immediately.",
            "This document explains incident escalation.",
        ],
        route_responses=[1, 0, 0, 0],
        answer_responses=[
            "Page the incident commander immediately.",
            "Grant access with manager approval.",
        ],
    )

    report = run_corpus_benchmark(
        [access_path, incidents_path],
        cases_path,
        corpus_path,
        IndexConfig(cache_dir=tmp_path / ".cache"),
        RetrievalConfig(sibling_window=1, include_ancestor_summaries=True),
        model_config=ModelConfig(),
        provider=provider,
    )

    assert report.case_count == 2
    assert report.passed_count == 2
    assert report.failed_count == 0
    assert report.case_results[0].document_title == "Incident Response"
    assert report.case_results[0].document_match is True


def test_run_corpus_comparison_benchmark_reports_method_level_results(
    tmp_path: Path,
) -> None:
    updates_path, handbook_path = _write_corpus_comparison_documents(tmp_path)
    cases_path = _write_corpus_comparison_cases(tmp_path)
    corpus_path = tmp_path / "ops-runbooks"
    provider = ContextAwareCorpusComparisonProvider(
        segment_responses=[
            [
                Section(
                    title="Critical Outage Updates",
                    content="During a critical outage, responders share status updates in Slack.",
                )
            ],
            [
                Section(
                    title="On-Call Handbook",
                    content=(
                        "The incident commander runs the room and coordinates responders "
                        "during a Sev-1."
                    ),
                )
            ],
        ],
        summary_responses=[
            "Responders share status updates in Slack during critical outages.",
            "This document explains outage status updates.",
            "The incident commander runs the room during a Sev-1.",
            "This document explains incident command responsibilities.",
        ],
        route_responses=[1, 0, 0],
    )

    report = run_corpus_comparison_benchmark(
        [updates_path, handbook_path],
        cases_path,
        corpus_path,
        IndexConfig(cache_dir=tmp_path / ".cache", subsection_word_threshold=999),
        RetrievalConfig(sibling_window=0, include_ancestor_summaries=False),
        model_config=ModelConfig(),
        provider=provider,
    )

    methods = {method.method: method for method in report.methods}
    assert set(methods) == {"tree_rag", "keyword_document", "full_context"}
    assert methods["tree_rag"].passed_count == 1
    assert methods["keyword_document"].failed_count == 1
    assert methods["full_context"].passed_count == 1


def test_corpus_benchmark_cli_outputs_summary_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    access_path, incidents_path = _write_corpus_documents(tmp_path)
    cases_path = _write_corpus_cases(tmp_path)
    corpus_path = tmp_path / "runbooks"
    provider = FakeProvider(
        segment_responses=[
            [Section(title="Access Requests", content="Grant access with manager approval.")],
            [_incident_response_section()],
        ],
        summary_responses=[
            "Grant access with manager approval.",
            "This document explains access request handling.",
            "Page the incident commander immediately.",
            "This document explains incident escalation.",
        ],
        route_responses=[1, 0, 0, 0],
        answer_responses=[
            "Page the incident commander immediately.",
            "Grant access with manager approval.",
        ],
    )

    assert (
        main(
            [
                "corpus-benchmark",
                str(corpus_path),
                str(cases_path),
                str(access_path),
                str(incidents_path),
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


def test_corpus_compare_cli_outputs_method_summary_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    updates_path, handbook_path = _write_corpus_comparison_documents(tmp_path)
    cases_path = _write_corpus_comparison_cases(tmp_path)
    corpus_path = tmp_path / "ops-runbooks"
    provider = ContextAwareCorpusComparisonProvider(
        segment_responses=[
            [
                Section(
                    title="Critical Outage Updates",
                    content="During a critical outage, responders share status updates in Slack.",
                )
            ],
            [
                Section(
                    title="On-Call Handbook",
                    content=(
                        "The incident commander runs the room and coordinates responders "
                        "during a Sev-1."
                    ),
                )
            ],
        ],
        summary_responses=[
            "Responders share status updates in Slack during critical outages.",
            "This document explains outage status updates.",
            "The incident commander runs the room during a Sev-1.",
            "This document explains incident command responsibilities.",
        ],
        route_responses=[1, 0, 0],
    )

    assert (
        main(
            [
                "corpus-compare",
                str(corpus_path),
                str(cases_path),
                str(updates_path),
                str(handbook_path),
                "--cache-dir",
                str(tmp_path / ".cache"),
            ],
            provider=provider,
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    methods = {method["method"]: method for method in output["methods"]}
    assert methods["tree_rag"]["passed_count"] == 1
    assert methods["keyword_document"]["failed_count"] == 1


def test_corpus_compare_cli_repeat_count_records_total_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    updates_path, handbook_path = _write_corpus_comparison_documents(tmp_path)
    cases_path = _write_corpus_comparison_cases(tmp_path)
    corpus_path = tmp_path / "ops-runbooks"
    provider = ContextAwareCorpusComparisonProvider(
        segment_responses=[
            [
                Section(
                    title="Critical Outage Updates",
                    content="During a critical outage, responders share status updates in Slack.",
                )
            ],
            [
                Section(
                    title="On-Call Handbook",
                    content=(
                        "The incident commander runs the room and coordinates responders "
                        "during a Sev-1."
                    ),
                )
            ],
        ],
        summary_responses=[
            "Responders share status updates in Slack during critical outages.",
            "This document explains outage status updates.",
            "The incident commander runs the room during a Sev-1.",
            "This document explains incident command responsibilities.",
        ],
        route_responses=[1, 0, 1, 0, 0, 0],
    )

    assert (
        main(
            [
                "corpus-compare",
                str(corpus_path),
                str(cases_path),
                str(updates_path),
                str(handbook_path),
                "--cache-dir",
                str(tmp_path / ".cache"),
                "--repeat",
                "2",
            ],
            provider=provider,
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    method = {entry["method"]: entry for entry in output["methods"]}["tree_rag"]
    assert method["total_runs"] == 2
    assert len(method["cases"][0]["query_samples_ms"]) == 2
