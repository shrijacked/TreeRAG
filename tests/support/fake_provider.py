from __future__ import annotations

from dataclasses import dataclass, field

from treerag.config import ModelConfig
from treerag.models import Section
from treerag.provider import LLMProvider, RouteChoice


@dataclass
class FakeProvider(LLMProvider):
    segment_responses: list[list[Section]] = field(default_factory=list)
    summary_responses: list[str] = field(default_factory=list)
    route_responses: list[int] = field(default_factory=list)
    answer_responses: list[str] = field(default_factory=list)
    segment_calls: int = 0
    summary_calls: int = 0
    route_calls: int = 0
    answer_calls: int = 0

    def segment(self, text: str, *, model_config: ModelConfig, char_limit: int) -> list[Section]:
        self.segment_calls += 1
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
        if not self.route_responses:
            raise AssertionError("No fake route response was queued.")
        return self.route_responses.pop(0)

    def answer(self, question: str, *, context: str, model_config: ModelConfig) -> str:
        self.answer_calls += 1
        if not self.answer_responses:
            raise AssertionError("No fake answer response was queued.")
        return self.answer_responses.pop(0)
