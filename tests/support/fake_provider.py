from __future__ import annotations

from dataclasses import dataclass, field

from treerag.config import ModelConfig
from treerag.models import Section
from treerag.provider import LLMProvider, RouteChoice, TokenUsage, UsageSnapshot


@dataclass
class FakeProvider(LLMProvider):
    segment_responses: list[list[Section]] = field(default_factory=list)
    summary_responses: list[str] = field(default_factory=list)
    route_responses: list[int] = field(default_factory=list)
    answer_responses: list[str] = field(default_factory=list)
    segment_usages: list[TokenUsage] = field(default_factory=list)
    summary_usages: list[TokenUsage] = field(default_factory=list)
    route_usages: list[TokenUsage] = field(default_factory=list)
    answer_usages: list[TokenUsage] = field(default_factory=list)
    segment_calls: int = 0
    summary_calls: int = 0
    route_calls: int = 0
    answer_calls: int = 0
    _usage_by_model: dict[str, TokenUsage] = field(default_factory=dict)

    def segment(self, text: str, *, model_config: ModelConfig, char_limit: int) -> list[Section]:
        self.segment_calls += 1
        self._record_usage(
            model_config.segmentation_model,
            self.segment_usages.pop(0) if self.segment_usages else TokenUsage(requests=1),
        )
        if not self.segment_responses:
            raise AssertionError("No fake segment response was queued.")
        return self.segment_responses.pop(0)

    def summarize(
        self,
        text: str,
        *,
        section_name: str,
        model_config: ModelConfig,
        char_limit: int,
    ) -> str:
        self.summary_calls += 1
        self._record_usage(
            model_config.summarization_model,
            self.summary_usages.pop(0) if self.summary_usages else TokenUsage(requests=1),
        )
        if not self.summary_responses:
            raise AssertionError("No fake summary response was queued.")
        return self.summary_responses.pop(0)

    def route(
        self,
        question: str,
        *,
        node_title: str,
        choices: list[RouteChoice],
        model_config: ModelConfig,
    ) -> int:
        self.route_calls += 1
        self._record_usage(
            model_config.routing_model,
            self.route_usages.pop(0) if self.route_usages else TokenUsage(requests=1),
        )
        if not self.route_responses:
            raise AssertionError("No fake route response was queued.")
        return self.route_responses.pop(0)

    def answer(self, question: str, *, context: str, model_config: ModelConfig) -> str:
        self.answer_calls += 1
        self._record_usage(
            model_config.answer_model,
            self.answer_usages.pop(0) if self.answer_usages else TokenUsage(requests=1),
        )
        if not self.answer_responses:
            raise AssertionError("No fake answer response was queued.")
        return self.answer_responses.pop(0)

    def usage_snapshot(self) -> UsageSnapshot:
        return UsageSnapshot(by_model=dict(self._usage_by_model))

    def _record_usage(self, model: str, usage: TokenUsage) -> None:
        current = self._usage_by_model.get(model, TokenUsage())
        self._usage_by_model[model] = current.add(usage)
