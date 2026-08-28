"""
Hugging Face Transformers GPU provider for V2 synthetic data generation.

Loads the model once and keeps it resident in GPU memory for the entire run.
Requires CUDA — fails clearly if unavailable.
"""

from __future__ import annotations

import os
from typing import Any

from .providers import Provider, _extract_json

DEFAULT_TRANSFORMERS_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Module-level singleton cache
_model_cache: dict[str, Any] = {}
_tokenizer_cache: dict[str, Any] = {}
_load_logged: set[str] = set()


def _format_vram() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            return f"{allocated:.2f} GB allocated"
    except Exception:
        pass
    return "unknown"


def _load_model_and_tokenizer(model_name: str) -> tuple[Any, Any]:
    if model_name in _model_cache and model_name in _tokenizer_cache:
        return _model_cache[model_name], _tokenizer_cache[model_name]

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Transformers provider requires torch and transformers. "
            "Run: pip install -r data-generation/requirements.txt"
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit(
            "Transformers provider requires CUDA but no GPU was detected. "
            "Use --provider ollama, anthropic, or openrouter instead."
        )

    device_name = torch.cuda.get_device_name(0)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if model_name not in _load_logged:
        print(f"Loading {model_name} on {device_name}...")
        print("  CUDA: available")
        print(f"  GPU: {device_name}")
        print(f"  dtype: {dtype}")
        _load_logged.add(model_name)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )
        model.eval()
    except OSError as exc:
        msg = str(exc).lower()
        if "gated" in msg or "401" in msg or "access" in msg or "token" in msg:
            raise SystemExit(
                f"Cannot load {model_name}. Hugging Face authentication may be required.\n"
                "  1. Accept the model license at huggingface.co\n"
                "  2. Run: huggingface-cli login\n"
                "  3. Or set HF_TOKEN in data-generation/.env"
            ) from exc
        raise

    _model_cache[model_name] = model
    _tokenizer_cache[model_name] = tokenizer

    print(f"Model loaded. VRAM: {_format_vram()}")

    return model, tokenizer


class TransformersProvider(Provider):
    name = "transformers"

    def __init__(self, model: str | None = None):
        self.model_name = model or os.getenv("TRANSFORMERS_MODEL", DEFAULT_TRANSFORMERS_MODEL)
        self._model, self._tokenizer = _load_model_and_tokenizer(self.model_name)

    def _build_messages(self, system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.85,
        max_new_tokens: int = 4096,
    ) -> str:
        import torch

        messages = self._build_messages(system_prompt, user_prompt)
        input_ids = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0, input_ids.shape[-1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict | None = None,
        *,
        temperature: float = 0.1,
        max_new_tokens: int = 2048,
        max_retries: int = 3,
    ) -> dict | None:
        del json_schema  # Transformers uses prompt-only JSON instructions

        for attempt in range(1, max_retries + 1):
            raw = self.generate_text(
                system_prompt,
                user_prompt,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            parsed = _extract_json(raw)
            if parsed is not None:
                return parsed
            print(f"  Transformers: JSON parse failed (attempt {attempt}/{max_retries})")
        return None

    def vram_summary(self) -> str:
        return _format_vram()
