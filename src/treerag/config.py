"""Configuration types for indexing and retrieval."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    segmentation_model: str = "gpt-5.4"
    summarization_model: str = "gpt-5.4-mini"
    routing_model: str = "gpt-5.4-mini"
    answer_model: str = "gpt-5.4"
    segmentation_max_completion_tokens: int = 3000
    summarization_max_completion_tokens: int = 200
    routing_max_completion_tokens: int = 20
    answer_max_completion_tokens: int = 500

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IndexConfig:
    subsection_word_threshold: int = 300
    max_depth: int = 4
    cache_dir: Path = Path(".cache/treerag")
    use_cache: bool = True
    segment_char_limit: int = 8000
    summary_char_limit: int = 3000

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["cache_dir"] = str(self.cache_dir)
        return payload


@dataclass(frozen=True)
class RetrievalConfig:
    sibling_window: int = 1
    include_ancestor_summaries: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
