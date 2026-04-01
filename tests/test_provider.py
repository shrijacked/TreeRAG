from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from treerag.config import ModelConfig
from treerag.errors import ParseError, ProviderError, RoutingError
from treerag.provider import GeminiProvider, OpenAIProvider, RouteChoice, create_provider


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


def test_create_provider_supports_gemini() -> None:
    provider = create_provider("gemini", client=_fake_gemini_client("hello"))

    assert isinstance(provider, GeminiProvider)


def test_create_provider_rejects_unknown_name() -> None:
    with pytest.raises(ProviderError):
        create_provider("mystery")
