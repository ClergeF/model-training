"""
HTTP client for the local Ollama REST API.

Uses POST /api/generate (not CLI/subprocess). Supports structured JSON schemas,
server health checks, model verification, and keep-alive warmup for bulk runs.
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import urlparse

import requests

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
GENERATION_TIMEOUT_SECONDS = 600
KEEP_ALIVE = "30m"
OLLAMA_NUM_PREDICT = 1100


class OllamaError(Exception):
    """Base error for Ollama HTTP client failures."""


class OllamaConnectionError(OllamaError):
    """Ollama server is not reachable."""


class OllamaModelNotFoundError(OllamaError):
    """Configured model is not installed locally."""


def resolve_base_url() -> str:
    """
    Read OLLAMA_BASE_URL, falling back to legacy OLLAMA_URL if it includes a path.
    """
    raw = os.getenv("OLLAMA_BASE_URL", "").strip()
    if not raw:
        legacy = os.getenv("OLLAMA_URL", "").strip()
        if legacy:
            parsed = urlparse(legacy)
            if parsed.scheme and parsed.netloc:
                return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
        return DEFAULT_OLLAMA_BASE_URL
    parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
    return DEFAULT_OLLAMA_BASE_URL


def generate_endpoint(base_url: str | None = None) -> str:
    return f"{(base_url or resolve_base_url()).rstrip('/')}/api/generate"


def tags_endpoint(base_url: str | None = None) -> str:
    return f"{(base_url or resolve_base_url()).rstrip('/')}/api/tags"


def _extract_json(raw_text: str) -> dict | None:
    if not raw_text or not raw_text.strip():
        return None
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass
    start = raw_text.find("{")
    if start == -1:
        return None
    end = raw_text.rfind("}")
    if end <= start:
        return None
    try:
        return json.loads(raw_text[start : end + 1])
    except json.JSONDecodeError:
        return None


def check_server(base_url: str | None = None, timeout: float = 5.0) -> None:
    """Raise OllamaConnectionError if the Ollama server is not reachable."""
    url = tags_endpoint(base_url)
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise OllamaConnectionError(
            f"Ollama is not running or refused connection at {url}. "
            "Start Ollama, then try again."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise OllamaConnectionError(
            f"Timed out connecting to Ollama at {url}."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise OllamaConnectionError(f"Could not reach Ollama at {url}: {exc}") from exc


def list_models(base_url: str | None = None, timeout: float = 10.0) -> list[str]:
    response = requests.get(tags_endpoint(base_url), timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return [item.get("name", "") for item in payload.get("models", []) if item.get("name")]


def model_is_available(model: str, base_url: str | None = None) -> bool:
    installed = list_models(base_url)
    if model in installed:
        return True
    # Ollama tags may omit :latest suffix.
    if ":" not in model:
        return f"{model}:latest" in installed
    base, tag = model.rsplit(":", 1)
    return base in installed or f"{base}:latest" in installed


def ensure_model_available(model: str, base_url: str | None = None) -> None:
    if not model_is_available(model, base_url):
        raise OllamaModelNotFoundError(
            f"Model '{model}' is not installed. Run: ollama pull {model}"
        )


def warmup_model(
    model: str,
    base_url: str | None = None,
    timeout: float = 120.0,
) -> None:
    """
    Send a tiny generate request so the model stays loaded for the bulk run.
    """
    payload: dict[str, Any] = {
        "model": model,
        "prompt": "Reply with the JSON object: {\"ok\": true}",
        "stream": False,
        "format": "json",
        "keep_alive": KEEP_ALIVE,
        "options": {"num_predict": 8, "temperature": 0},
    }
    response = requests.post(generate_endpoint(base_url), json=payload, timeout=timeout)
    response.raise_for_status()


def generate_json(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    json_schema: dict | None = None,
    base_url: str | None = None,
    timeout: float = GENERATION_TIMEOUT_SECONDS,
) -> tuple[dict | None, str | None]:
    """
    Call Ollama /api/generate and return (parsed_json, error_message).

    On success error_message is None. On failure parsed_json is None.
    Does not raise for per-example failures (bulk runs keep going).
    """
    payload: dict[str, Any] = {
        "model": model,
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": 0.8,
            "num_predict": OLLAMA_NUM_PREDICT,
        },
    }
    if json_schema:
        payload["format"] = json_schema
    else:
        payload["format"] = "json"

    url = generate_endpoint(base_url)

    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        return None, "Ollama connection refused (is the server running?)"
    except requests.exceptions.Timeout:
        return None, f"Ollama request timed out after {int(timeout)} seconds"
    except requests.exceptions.HTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        if status == 404:
            return None, f"Model '{model}' not found (run: ollama pull {model})"
        return None, f"Ollama HTTP error: {exc}"
    except requests.exceptions.RequestException as exc:
        return None, f"Ollama request failed: {exc}"

    try:
        api_data = response.json()
    except json.JSONDecodeError:
        return None, "Ollama returned a non-JSON HTTP body"

    raw_text = api_data.get("response", "")
    if not raw_text or not str(raw_text).strip():
        return None, "Ollama returned an empty response"

    parsed = _extract_json(str(raw_text))
    if parsed is None:
        preview = str(raw_text)[:120].replace("\n", " ")
        return None, f"Could not parse JSON from model output: {preview!r}..."

    return parsed, None
