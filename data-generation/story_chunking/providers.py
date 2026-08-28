"""
Provider adapters for synthetic data generation.

Providers behind one interface:
- Ollama (local HTTP API at OLLAMA_BASE_URL/api/generate, default llama3.1:8b)
- Anthropic (requires ANTHROPIC_API_KEY)
- OpenRouter (requires OPENROUTER_API_KEY; e.g. Claude via OpenRouter)

Each provider exposes generate_json(system_prompt, user_prompt, json_schema=None) -> dict | None.
"""

from __future__ import annotations

import json
import os
from typing import Any

import requests

from .ollama_client import (
    DEFAULT_OLLAMA_MODEL,
    generate_endpoint,
    generate_json as ollama_generate_json,
    resolve_base_url,
)

DEFAULT_ANTHROPIC_MODEL = "claude-3-5-sonnet-latest"
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = "anthropic/claude-sonnet-4.6"

OPENROUTER_TIMEOUT_SECONDS = 180


def _extract_json(raw_text: str) -> dict | None:
    """Parse a JSON object from model text, tolerating leading/trailing prose."""
    if not raw_text:
        return None
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw_text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


class Provider:
    name = "base"

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict | None = None,
        **kwargs: Any,
    ) -> dict | None:
        raise NotImplementedError


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = base_url or resolve_base_url()
        self.model = model or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        self.generate_url = generate_endpoint(self.base_url)

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict | None = None,
        **kwargs: Any,
    ) -> dict | None:
        del kwargs
        parsed, error = ollama_generate_json(
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_schema=json_schema,
            base_url=self.base_url,
        )
        if error:
            print(f"  Ollama: {error}")
        return parsed


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "").strip()
        self.model = model or os.getenv("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)

        if not self.api_key:
            raise SystemExit(
                "ANTHROPIC_API_KEY is not set. Add it to data-generation/.env "
                "or choose --provider ollama."
            )

        try:
            import anthropic  # noqa: PLC0415 - optional dependency
        except ImportError:
            raise SystemExit(
                "The 'anthropic' package is not installed. "
                "Run: pip install -r data-generation/requirements.txt"
            )

        self._client = anthropic.Anthropic(api_key=self.api_key)

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict | None = None,
        **kwargs: Any,
    ) -> dict | None:
        del json_schema, kwargs  # Anthropic uses prompt-only JSON instructions
        try:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=0.8,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text_parts = [
                block.text for block in message.content if getattr(block, "type", "") == "text"
            ]
            return _extract_json("".join(text_parts))
        except Exception as exc:  # noqa: BLE001 - log and skip this attempt
            print(f"  Error calling Anthropic: {exc}")
            return None


class OpenRouterProvider(Provider):
    name = "openrouter"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "").strip()
        self.model = model or os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
        self.url = os.getenv("OPENROUTER_URL", DEFAULT_OPENROUTER_URL).strip()

        if not self.api_key:
            raise SystemExit(
                "OPENROUTER_API_KEY is not set. Add it to data-generation/.env "
                "or choose --provider ollama."
            )

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict | None = None,
        **kwargs: Any,
    ) -> dict | None:
        del json_schema, kwargs  # OpenRouter uses response_format json_object
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.8,
            "max_tokens": 4096,
            "response_format": {"type": "json_object"},
        }

        try:
            response = requests.post(
                self.url,
                headers=headers,
                json=payload,
                timeout=OPENROUTER_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            api_data = response.json()
            content = api_data["choices"][0]["message"]["content"]
            return _extract_json(content)
        except requests.exceptions.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status == 401:
                print("  OpenRouter returned 401 — check OPENROUTER_API_KEY in .env")
            else:
                print(f"  HTTP error from OpenRouter: {exc}")
            return None
        except requests.exceptions.ReadTimeout:
            print(f"  OpenRouter timed out after {OPENROUTER_TIMEOUT_SECONDS} seconds")
            return None
        except (KeyError, IndexError, TypeError) as exc:
            print(f"  Unexpected OpenRouter response shape: {exc}")
            return None
        except Exception as exc:  # noqa: BLE001 - log and skip this attempt
            print(f"  Error calling OpenRouter: {exc}")
            return None


def get_provider(name: str, model: str | None = None) -> Provider:
    name = (name or "").strip().lower()
    if name == "ollama":
        return OllamaProvider()
    if name == "anthropic":
        return AnthropicProvider()
    if name == "openrouter":
        return OpenRouterProvider()
    if name == "transformers":
        from .transformers_provider import TransformersProvider

        return TransformersProvider(model=model)
    raise SystemExit(
        f"Unknown provider '{name}'. Use 'ollama', 'anthropic', 'openrouter', or 'transformers'."
    )
