"""Provider abstractions for segmentation, summaries, routing, and answers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from openai import OpenAI, OpenAIError

from treerag.config import ModelConfig
from treerag.errors import ParseError, ProviderError, RoutingError
from treerag.models import Section

try:
    from google import genai  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised through create_provider failure paths
    genai = None


@dataclass(frozen=True)
class RouteChoice:
    title: str
    summary: str


class LLMProvider(Protocol):
    """Protocol for pluggable model backends."""

    def segment(self, text: str, *, model_config: ModelConfig, char_limit: int) -> list[Section]:
        ...

    def summarize(
        self,
        text: str,
        *,
        section_name: str,
        model_config: ModelConfig,
        char_limit: int,
    ) -> str:
        ...

    def route(
        self,
        question: str,
        *,
        node_title: str,
        choices: list[RouteChoice],
        model_config: ModelConfig,
    ) -> int:
        ...

    def answer(self, question: str, *, context: str, model_config: ModelConfig) -> str:
        ...


class OpenAIProvider:
    """OpenAI-backed implementation of the provider protocol."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self.client = client or OpenAI(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    def segment(self, text: str, *, model_config: ModelConfig, char_limit: int) -> list[Section]:
        prompt = (
            "Split the following text into logical sections.\n"
            'Return a JSON object with a "sections" key. Each item must have:\n'
            '- "title": a short title (5 words or less)\n'
            '- "content": the text belonging to this section\n\n'
            f"Text:\n{text[:char_limit]}"
        )
        payload = self._chat_completion(
            model=model_config.segmentation_model,
            prompt=prompt,
            max_completion_tokens=model_config.segmentation_max_completion_tokens,
            response_format={"type": "json_object"},
        )
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ParseError("Provider returned malformed JSON for segmentation.") from exc

        sections = parsed.get("sections")
        if not isinstance(sections, list):
            raise ParseError('Segmentation response must include a list under "sections".')

        normalized: list[Section] = []
        for section in sections:
            if not isinstance(section, dict):
                raise ParseError("Each segmented section must be an object.")
            title = section.get("title")
            content = section.get("content")
            if not isinstance(title, str) or not title.strip():
                raise ParseError("Segmented sections must include a non-empty title.")
            if not isinstance(content, str):
                raise ParseError("Segmented sections must include string content.")
            normalized.append(Section(title=title.strip(), content=content))
        return normalized

    def summarize(
        self,
        text: str,
        *,
        section_name: str,
        model_config: ModelConfig,
        char_limit: int,
    ) -> str:
        hint = f"This is the section titled: {section_name}.\n" if section_name else ""
        prompt = (
            f"{hint}Summarize the following in 2-3 sentences. "
            "Be specific and factual. Do not add anything not in the text.\n\n"
            f"{text[:char_limit]}"
        )
        return self._chat_completion(
            model=model_config.summarization_model,
            prompt=prompt,
            max_completion_tokens=model_config.summarization_max_completion_tokens,
        ).strip()

    def route(
        self,
        question: str,
        *,
        node_title: str,
        choices: list[RouteChoice],
        model_config: ModelConfig,
    ) -> int:
        options = "\n".join(
            f"{index + 1}. [{choice.title}]: {choice.summary}"
            for index, choice in enumerate(choices)
        )
        prompt = (
            "You are navigating a document tree to find the answer to a question.\n\n"
            f'Current section: "{node_title}"\n'
            f"Question: {question}\n\n"
            f"Children of this section:\n{options}\n\n"
            "Which child section most likely contains the answer? Reply with only the number."
        )
        raw_choice = self._chat_completion(
            model=model_config.routing_model,
            prompt=prompt,
            max_completion_tokens=model_config.routing_max_completion_tokens,
        ).strip()
        try:
            return int(raw_choice) - 1
        except ValueError as exc:
            raise RoutingError(
                f"Provider returned a non-numeric route choice: {raw_choice!r}"
            ) from exc

    def answer(self, question: str, *, context: str, model_config: ModelConfig) -> str:
        prompt = (
            "Answer using only the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )
        return self._chat_completion(
            model=model_config.answer_model,
            prompt=prompt,
            max_completion_tokens=model_config.answer_max_completion_tokens,
        ).strip()

    def _chat_completion(
        self,
        *,
        model: str,
        prompt: str,
        max_completion_tokens: int,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_completion_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        try:
            response = self.client.chat.completions.create(**kwargs)
        except OpenAIError as exc:
            raise ProviderError(str(exc)) from exc
        return _extract_message_text(response)


class GeminiProvider:
    """Gemini-backed implementation of the provider protocol."""

    def __init__(self, *, client: Any | None = None, api_key: str | None = None) -> None:
        if client is not None:
            self.client = client
            return
        if genai is None:
            raise ProviderError(
                "Gemini support requires the 'google-genai' package to be installed."
            )
        self.client = genai.Client(api_key=api_key)

    def segment(self, text: str, *, model_config: ModelConfig, char_limit: int) -> list[Section]:
        prompt = (
            "Split the following text into logical sections.\n"
            'Return a JSON object with a "sections" key. Each item must have:\n'
            '- "title": a short title (5 words or less)\n'
            '- "content": the text belonging to this section\n\n'
            f"Text:\n{text[:char_limit]}"
        )
        payload = self._generate(
            model=model_config.segmentation_model,
            prompt=prompt,
            max_output_tokens=model_config.segmentation_max_completion_tokens,
            response_json_schema=_SEGMENTATION_SCHEMA,
        )
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ParseError("Provider returned malformed JSON for segmentation.") from exc

        sections = parsed.get("sections")
        if not isinstance(sections, list):
            raise ParseError('Segmentation response must include a list under "sections".')

        normalized: list[Section] = []
        for section in sections:
            if not isinstance(section, dict):
                raise ParseError("Each segmented section must be an object.")
            title = section.get("title")
            content = section.get("content")
            if not isinstance(title, str) or not title.strip():
                raise ParseError("Segmented sections must include a non-empty title.")
            if not isinstance(content, str):
                raise ParseError("Segmented sections must include string content.")
            normalized.append(Section(title=title.strip(), content=content))
        return normalized

    def summarize(
        self,
        text: str,
        *,
        section_name: str,
        model_config: ModelConfig,
        char_limit: int,
    ) -> str:
        hint = f"This is the section titled: {section_name}.\n" if section_name else ""
        prompt = (
            f"{hint}Summarize the following in 2-3 sentences. "
            "Be specific and factual. Do not add anything not in the text.\n\n"
            f"{text[:char_limit]}"
        )
        return self._generate(
            model=model_config.summarization_model,
            prompt=prompt,
            max_output_tokens=model_config.summarization_max_completion_tokens,
        ).strip()

    def route(
        self,
        question: str,
        *,
        node_title: str,
        choices: list[RouteChoice],
        model_config: ModelConfig,
    ) -> int:
        options = "\n".join(
            f"{index + 1}. [{choice.title}]: {choice.summary}"
            for index, choice in enumerate(choices)
        )
        prompt = (
            "You are navigating a document tree to find the answer to a question.\n\n"
            f'Current section: "{node_title}"\n'
            f"Question: {question}\n\n"
            f"Children of this section:\n{options}\n\n"
            "Which child section most likely contains the answer? Reply with only the number."
        )
        raw_choice = self._generate(
            model=model_config.routing_model,
            prompt=prompt,
            max_output_tokens=model_config.routing_max_completion_tokens,
        ).strip()
        try:
            return int(raw_choice) - 1
        except ValueError as exc:
            raise RoutingError(
                f"Provider returned a non-numeric route choice: {raw_choice!r}"
            ) from exc

    def answer(self, question: str, *, context: str, model_config: ModelConfig) -> str:
        prompt = (
            "Answer using only the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )
        return self._generate(
            model=model_config.answer_model,
            prompt=prompt,
            max_output_tokens=model_config.answer_max_completion_tokens,
        ).strip()

    def _generate(
        self,
        *,
        model: str,
        prompt: str,
        max_output_tokens: int,
        response_json_schema: dict[str, Any] | None = None,
    ) -> str:
        config: dict[str, Any] = {"max_output_tokens": max_output_tokens}
        if response_json_schema is not None:
            config["response_mime_type"] = "application/json"
            config["response_json_schema"] = response_json_schema
        try:
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
        except Exception as exc:  # pragma: no cover - SDK-specific subclasses vary by version
            raise ProviderError(str(exc)) from exc
        return _extract_text_response(response)


def create_provider(
    name: str, *, api_key: str | None = None, client: Any | None = None
) -> LLMProvider:
    normalized = name.strip().lower()
    if normalized == "openai":
        return OpenAIProvider(client=client, api_key=api_key)
    if normalized == "gemini":
        return GeminiProvider(client=client, api_key=api_key)
    raise ProviderError(f"Unknown provider {name!r}. Expected one of: openai, gemini.")


def _extract_message_text(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise ProviderError("Provider response did not include a message content payload.") from exc

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_chunks: list[str] = []
        for chunk in content:
            if isinstance(chunk, dict) and isinstance(chunk.get("text"), str):
                text_chunks.append(chunk["text"])
            elif hasattr(chunk, "text") and isinstance(chunk.text, str):
                text_chunks.append(chunk.text)
        if text_chunks:
            return "".join(text_chunks)
    raise ProviderError("Provider response content was not a text payload.")


def _extract_text_response(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text
    raise ProviderError("Provider response did not include a text payload.")


_SEGMENTATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["title", "content"],
            },
        }
    },
    "required": ["sections"],
}
