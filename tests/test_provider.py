from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from treerag.config import ModelConfig
from treerag.errors import ParseError, ProviderError, RoutingError
from treerag.provider import (
    GeminiProvider,
    OpenAIProvider,
    RouteChoice,
    TokenUsage,
    create_provider,
)


def _fake_client(content: str, capture: list[dict[str, Any]] | None = None) -> SimpleNamespace:
    def create(**kwargs: Any) -> SimpleNamespace:
        if capture is not None:
            capture.append(kwargs)
        message = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])

    completions = SimpleNamespace(create=create)
    chat = SimpleNamespace(completions=completions)
    return SimpleNamespace(chat=chat)


def _fake_gemini_client(
    content: str, capture: list[dict[str, Any]] | None = None
) -> SimpleNamespace:
    def generate_content(**kwargs: Any) -> SimpleNamespace:
        if capture is not None:
            capture.append(kwargs)
        return SimpleNamespace(text=content)

    models = SimpleNamespace(generate_content=generate_content)
    return SimpleNamespace(models=models)


def test_segment_raises_parse_error_on_invalid_json() -> None:
    provider = OpenAIProvider(client=_fake_client("{bad json"))

    with pytest.raises(ParseError):
        provider.segment("text", model_config=ModelConfig(), char_limit=8000)


def test_route_raises_routing_error_on_invalid_choice() -> None:
    provider = OpenAIProvider(client=_fake_client("not-a-number"))

    with pytest.raises(RoutingError):
        provider.route(
            "where is the answer?",
            node_title="root",
            choices=[RouteChoice(title="A", summary="alpha")],
            model_config=ModelConfig(),
        )


def test_provider_uses_consistent_completion_token_parameter() -> None:
    capture: list[dict[str, Any]] = []
    provider = OpenAIProvider(client=_fake_client('{"sections": []}', capture))
    model_config = ModelConfig(segmentation_max_completion_tokens=111)

    provider.segment("text", model_config=model_config, char_limit=8000)

    assert capture[0]["max_completion_tokens"] == 111
    assert "max_tokens" not in capture[0]


def test_gemini_segment_raises_parse_error_on_invalid_json() -> None:
    provider = GeminiProvider(client=_fake_gemini_client("{bad json"))

    with pytest.raises(ParseError):
        provider.segment("text", model_config=ModelConfig(), char_limit=8000)


def test_gemini_provider_uses_json_response_config() -> None:
    capture: list[dict[str, Any]] = []
    provider = GeminiProvider(client=_fake_gemini_client('{"sections": []}', capture))
    model_config = ModelConfig(segmentation_max_completion_tokens=111)

    provider.segment("text", model_config=model_config, char_limit=8000)

    assert capture[0]["model"] == model_config.segmentation_model
    assert capture[0]["config"]["max_output_tokens"] == 111
    assert capture[0]["config"]["response_mime_type"] == "application/json"
    assert capture[0]["config"]["response_json_schema"]["type"] == "object"


def test_gemini_provider_disables_thinking_for_deterministic_output() -> None:
    capture: list[dict[str, Any]] = []
    provider = GeminiProvider(client=_fake_gemini_client("hello", capture))

    provider.answer("question", context="context", model_config=ModelConfig())

    assert capture[0]["config"]["thinking_config"]["thinking_budget"] == 0


def test_gemini_route_reads_text_from_candidate_parts() -> None:
    response = SimpleNamespace(
        text=None,
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text="1")])
            )
        ],
    )
    provider = GeminiProvider(
        client=SimpleNamespace(
            models=SimpleNamespace(generate_content=lambda **_: response)
        )
    )

    choice = provider.route(
        "where is the answer?",
        node_title="root",
        choices=[RouteChoice(title="A", summary="alpha")],
        model_config=ModelConfig(),
    )

    assert choice == 0


def test_create_provider_supports_gemini() -> None:
    provider = create_provider("gemini", client=_fake_gemini_client("hello"))

    assert isinstance(provider, GeminiProvider)


def test_create_provider_rejects_unknown_name() -> None:
    with pytest.raises(ProviderError):
        create_provider("mystery")


def test_openai_provider_records_usage_snapshot() -> None:
    usage = SimpleNamespace(
        prompt_tokens=120,
        completion_tokens=30,
        total_tokens=150,
        prompt_tokens_details=SimpleNamespace(cached_tokens=20),
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))],
        usage=usage,
    )
    provider = OpenAIProvider(
        client=SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_: response)
            )
        )
    )

    provider.answer("question", context="context", model_config=ModelConfig())
    snapshot = provider.usage_snapshot()

    assert snapshot.total == TokenUsage(
        requests=1,
        input_tokens=120,
        output_tokens=30,
        total_tokens=150,
        cached_input_tokens=20,
    )
    assert snapshot.by_model[ModelConfig().answer_model].cached_input_tokens == 20


def test_gemini_provider_records_usage_snapshot() -> None:
    usage_metadata = SimpleNamespace(
        prompt_token_count=90,
        candidates_token_count=15,
        total_token_count=105,
        cached_content_token_count=10,
    )
    response = SimpleNamespace(text="hello", usage_metadata=usage_metadata)
    provider = GeminiProvider(
        client=SimpleNamespace(
            models=SimpleNamespace(generate_content=lambda **_: response)
        )
    )

    provider.answer("question", context="context", model_config=ModelConfig())
    snapshot = provider.usage_snapshot()

    assert snapshot.total == TokenUsage(
        requests=1,
        input_tokens=90,
        output_tokens=15,
        total_tokens=105,
        cached_input_tokens=10,
    )
    assert snapshot.by_model[ModelConfig().answer_model].output_tokens == 15
