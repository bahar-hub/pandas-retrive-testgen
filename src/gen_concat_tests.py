#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_concat_tests.py
Generate pytest tests for pandas.concat using multiple LLMs (Ollama local + OpenRouter).

- Works with any source file (e.g., pandas_concat.json or a raw .py file).
- If the source is a JSON spec (from fetch.py), it extracts body_stripped and uses
  signature/docs to enrich the prompt.
- Supports both Ollama and OpenRouter providers.

Usage example:
    python gen_concat_tests.py \
      --src pandas_concat.json \
      --out-pattern "generated/{model}.py" \
      --models "llama-4-scout:free,mistral-7b-instruct:free,gemma-3-12b-it:free,qwen3-14b:free,claude-sonnet-4,gemini-2.5-pro" \
      --default-provider openrouter
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence, Tuple

import requests
import pandas as pd 

# ============================== Config & Templates ==============================

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llama3:8b-instruct"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TIMEOUT = 600  # seconds for generation
DEFAULT_HEALTH_TIMEOUT = 5
DEFAULT_RETRIES = 2
DEFAULT_BACKOFF = 2.0
DEFAULT_NUM_CTX = 8192  # generous for long sources

# OpenRouter defaults
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
ENV_OPENROUTER_API_KEY = "OPENROUTER_API_KEY"
DEFAULT_OPENROUTER_TIMEOUT = 600

# Optional: hardcode your OpenRouter API key here for local use.
HARDCODED_OPENROUTER_API_KEY: Optional[str] = None  # e.g., "sk-or-xxxxxxxxxxxxxxxx"

PROMPT_TEMPLATE = """You are a precise code generator and a senior Python test engineer.
Output ONLY valid Python code (no explanations, no markdown fences).

Target environment:
- Python: {python_version}
- pandas: {pandas_version}

Task:
Given the exact source of the function `pandas.concat` (below), write a comprehensive `pytest` test file named "{outfile}" that validates its behavior across edge cases.

Important constraints:
- All tests MUST pass on the given pandas version in a standard environment.
- Do NOT rely on undocumented internal behavior or private APIs.
- Do NOT use keyword arguments that do not appear in the provided function signature.
- Use only stable, documented behavior that is compatible with pandas {pandas_version}.

Requirements:
- Use `pytest` and `pandas` (and `numpy` if needed).
- Use `pandas.testing.assert_frame_equal` and `assert_series_equal` for equality checks.
- Keep tests deterministic and fast (no I/O, no randomness, no network).
- Cover as many parameters and edge cases as possible, including:
  axis (0/1), keys/names, ignore_index, join=('outer','inner'), sort, levels, verify_integrity,
  concatenating Series vs DataFrame, empty objects, mixed dtypes, non-unique indices/columns,
  MultiIndex rows/columns, differing column sets, categorical dtypes, datetime/timezone dtypes,
  boolean/object/numeric mixes, and handling of copy vs. views if observable.
- Prefer multiple small focused tests, clear names `test_*`.
- Keep the file self-contained: include necessary imports only.

Function signature (for reference):
{signature_block}

Selected examples from the official docs/spec (if any):
{examples_block}

Function source of `pandas.concat`:
<<<BEGIN_SOURCE
{func_source}
<<<END_SOURCE
"""

# ============================== Utilities ==============================

def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")
    return path.read_text(encoding="utf-8")


def extract_code_block(text: str) -> str:
    """
    If the model returns fenced code blocks, extract the first one.
    Otherwise, return the text as-is (trimmed).
    """
    m = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.S)
    if m:
        return m.group(1).strip()
    return text.strip()


def sanitize_filename_piece(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", s)


def load_source_and_spec(src: Path) -> Tuple[str, Optional[dict]]:

    text = read_text(src)
    if src.suffix.lower() == ".json":
        spec = json.loads(text)
        body = spec.get("body_stripped", "")
        if not body:
            raise RuntimeError("JSON spec has no 'body_stripped' field.")
        return body, spec
    else:
        return text, None


def build_prompt(func_source: str, outfile: str, spec: Optional[dict]) -> str:

    python_version = ".".join(map(str, sys.version_info[:3]))
    if spec is not None:
        pandas_version = spec.get("version", pd.__version__)
        signature = spec.get("signature", "")
        examples = spec.get("examples_code") or []
    else:
        pandas_version = pd.__version__
        signature = ""
        examples = []

    if signature:
        signature_block = signature
    else:
        signature_block = "(signature not provided in spec)"

    if examples:
        max_examples = 3
        selected = examples[:max_examples]
        examples_block = "\n\n".join(
            f"Example {i+1}:\n{code}" for i, code in enumerate(selected)
        )
    else:
        examples_block = "(no examples provided in spec)"

    return PROMPT_TEMPLATE.format(
        func_source=func_source,
        outfile=outfile,
        python_version=python_version,
        pandas_version=pandas_version,
        signature_block=signature_block,
        examples_block=examples_block,
    )

# For OpenRouter model aliases (org/model slugs).
OPENROUTER_MODEL_ALIASES = {
    "deepseek-r1t2-chimera:free": "tngtech/deepseek-r1t2-chimera:free",

    "qwen3-coder:free": "qwen/qwen3-coder:free",


    "gemma-3-4b-it:free": "google/gemma-3-4b-it:free",        

    "llama-3.3-70b-instruct:free": "meta-llama/llama-3.3-70b-instruct:free",
    
    "sherlock-dash-alpha": "openrouter/sherlock-dash-alpha",

    "mistral-7b-instruct:free": "mistralai/mistral-7b-instruct:free",
}

def normalize_openrouter_model(model: str) -> str:
    """
    If the model lacks an org prefix, try to map it to a canonical slug.
    If it already contains '/', return as-is.
    """
    if "/" in model:
        return model
    return OPENROUTER_MODEL_ALIASES.get(model, model)


# ============================== Ollama Client ==============================

@dataclass
class OllamaOptions:
    host: str = DEFAULT_OLLAMA_HOST
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    timeout: int = DEFAULT_TIMEOUT
    health_timeout: int = DEFAULT_HEALTH_TIMEOUT
    retries: int = DEFAULT_RETRIES
    backoff: float = DEFAULT_BACKOFF
    stream: bool = True
    num_ctx: int = DEFAULT_NUM_CTX


class OllamaClient:
    def __init__(self, opts: OllamaOptions) -> None:
        self.opts = opts

    # --- Health & Model Checks ---

    def health(self) -> None:
        url = f"{self.opts.host}/api/tags"
        r = requests.get(url, timeout=self.opts.health_timeout)
        r.raise_for_status()

    def model_available(self, model: str | None = None) -> bool:
        model = model or self.opts.model
        url = f"{self.opts.host}/api/tags"
        r = requests.get(url, timeout=self.opts.health_timeout)
        r.raise_for_status()
        data = r.json()
        for m in data.get("models", []):
            if m.get("model") == model or m.get("name") == model:
                return True
        return False

    # --- Generation ---

    def _payload(self, prompt: str, stream: bool) -> dict:
        return {
            "model": self.opts.model,
            "prompt": prompt,
            "temperature": self.opts.temperature,
            "stream": stream,
            "options": {"num_ctx": self.opts.num_ctx},
        }

    def generate_stream(self, prompt: str) -> Iterator[str]:
        url = f"{self.opts.host}/api/generate"
        payload = self._payload(prompt, stream=True)
        with requests.post(url, json=payload, stream=True, timeout=self.opts.timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunk = obj.get("response", "")
                if chunk:
                    yield chunk

    def generate_once(self, prompt: str) -> str:
        url = f"{self.opts.host}/api/generate"
        payload = self._payload(prompt, stream=False)
        r = requests.post(url, json=payload, timeout=self.opts.timeout)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")

    def generate(self, prompt: str) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(1, self.opts.retries + 2):
            try:
                self.health()
                if self.opts.stream:
                    chunks: list[str] = []
                    for chunk in self.generate_stream(prompt):
                        chunks.append(chunk)
                    text = "".join(chunks)
                    if text.strip():
                        return text
                text = self.generate_once(prompt)
                if text.strip():
                    return text
                raise RuntimeError("Empty response from Ollama.")
            except Exception as e:
                last_err = e
                if attempt <= self.opts.retries:
                    time.sleep(self.opts.backoff * attempt)
                else:
                    raise RuntimeError(f"Ollama request failed after retries: {last_err}") from last_err
        return ""  # unreachable


# ============================== OpenRouter Client ==============================

@dataclass
class OpenRouterOptions:
    base_url: str = DEFAULT_OPENROUTER_BASE_URL
    api_key: Optional[str] = None
    model: str = "meta-llama/llama-4-scout:free"
    temperature: float = DEFAULT_TEMPERATURE
    timeout: int = DEFAULT_OPENROUTER_TIMEOUT
    retries: int = DEFAULT_RETRIES
    backoff: float = DEFAULT_BACKOFF
    site_url: Optional[str] = None  # Optional: Referer header
    app_name: Optional[str] = "gen-concat-tests"  # Optional: X-Title header


class OpenRouterClient:
    def __init__(self, opts: OpenRouterOptions) -> None:
        self.opts = opts
        self.session = requests.Session()

    def _resolve_api_key(self) -> str:
        # Precedence: Hardcoded > CLI arg > env
        if HARDCODED_OPENROUTER_API_KEY:
            return HARDCODED_OPENROUTER_API_KEY
        if self.opts.api_key:
            return self.opts.api_key
        env_key = os.getenv(ENV_OPENROUTER_API_KEY)
        if env_key:
            return env_key
        raise RuntimeError(f"Missing OpenRouter API key. Set {ENV_OPENROUTER_API_KEY}=... or hardcode in script.")

    def generate(self, prompt: str) -> str:
        """
        Calls OpenRouter's /chat/completions endpoint (OpenAI-compatible).
        """
        api_key = self._resolve_api_key()

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self.opts.site_url:
            headers["HTTP-Referer"] = self.opts.site_url
        if self.opts.app_name:
            headers["X-Title"] = self.opts.app_name

        payload = {
            "model": self.opts.model,
            "temperature": self.opts.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        last_err: Optional[Exception] = None
        for attempt in range(1, self.opts.retries + 2):
            try:
                r = self.session.post(self.opts.base_url, headers=headers, json=payload, timeout=self.opts.timeout)
                if r.status_code >= 400:
                    try:
                        details = r.json()
                    except Exception:
                        details = {"error_text": r.text}
                    r.raise_for_status()
                data = r.json()
                choices = data.get("choices", [])
                if not choices:
                    raise RuntimeError("No choices returned by OpenRouter.")
                content = choices[0]["message"]["content"]
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    text = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                else:
                    text = str(content)
                if text.strip():
                    return text
                raise RuntimeError("Empty response from OpenRouter.")
            except requests.HTTPError as e:
                try:
                    details = r.json()
                except Exception:
                    details = {"error_text": r.text}
                last_err = RuntimeError(f"{e} · details={details}")
                if attempt <= self.opts.retries:
                    time.sleep(self.opts.backoff * attempt)
                else:
                    raise last_err
            except Exception as e:
                last_err = e
                if attempt <= self.opts.retries:
                    time.sleep(self.opts.backoff * attempt)
                else:
                    raise last_err
        return ""  # unreachable


# ============================== App Logic ==============================

def parse_models_arg(models_str: str, default_provider: str) -> list[Tuple[str, str]]:
    """
    Returns list of (provider, model).

    Behavior:
      - If an item starts with a known provider prefix ("openrouter:" or "ollama:"),
        split at the first colon and treat that as the provider.
      - Otherwise, treat the whole item as a MODEL and use default_provider.
    """
    known_prefixes = ("openrouter:", "ollama:")
    items: list[tuple[str, str]] = []
    for raw in [s.strip() for s in models_str.split(",") if s.strip()]:
        lower = raw.lower()
        if lower.startswith(known_prefixes):
            first_colon = raw.find(":")
            provider = raw[:first_colon].strip().lower()
            model = raw[first_colon + 1 :].strip()
            items.append((provider, model))
        else:
            items.append((default_provider.lower(), raw))
    return items


def run_one(
    provider: str,
    model: str,
    prompt: str,
    *,
    ollama_opts: OllamaOptions,
    openrouter_opts: OpenRouterOptions,
) -> str:
    if provider == "ollama":
        client = OllamaClient(OllamaOptions(**{**ollama_opts.__dict__, "model": model}))
        try:
            if not client.model_available(model):
                print(
                    f"[WARN] Ollama model '{model}' not found. You may need to:  ollama pull {model}",
                    file=sys.stderr,
                )
        except Exception:
            pass
        return client.generate(prompt)

    elif provider == "openrouter":
        norm_model = normalize_openrouter_model(model)
        client = OpenRouterClient(OpenRouterOptions(**{**openrouter_opts.__dict__, "model": norm_model}))
        return client.generate(prompt)

    else:
        raise ValueError(f"Unknown provider: {provider}")


def run_many(
    src: Path,
    out_pattern: str,
    models: Sequence[Tuple[str, str]],
    *,
    print_prompt: bool,
    ollama_opts: OllamaOptions,
    openrouter_opts: OpenRouterOptions,
) -> int:
    func_source, spec = load_source_and_spec(src)

    for provider, model in models:
        safe_provider = sanitize_filename_piece(provider)
        safe_model = sanitize_filename_piece(model)
        out_path = Path(out_pattern.format(provider=safe_provider, model=safe_model))

        prompt = build_prompt(func_source, out_path.name, spec)
        if print_prompt:
            print(f"\n==== Prompt for {provider}:{model} (outfile {out_path.name}) ====\n")
            print(prompt)
            continue

        print(f"→ Running {provider}:{model}")
        try:
            raw = run_one(
                provider,
                model,
                prompt,
                ollama_opts=ollama_opts,
                openrouter_opts=openrouter_opts,
            )
            code = extract_code_block(raw)
            if not code:
                print(f"[ERROR] Empty code from {provider}:{model}", file=sys.stderr)
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(code, encoding="utf-8")
            print(f"[OK] Wrote: {out_path.resolve()}")
        except Exception as e:
            print(f"[FAIL] {provider}:{model} -> {e}", file=sys.stderr)

    return 0


# ============================== CLI ==============================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate pytest tests for pandas.concat using multiple LLMs (Ollama + OpenRouter)."
    )
    p.add_argument(
        "--src",
        default="pandas_concat.json",
        help="Path to saved function body or JSON spec (e.g. pandas_concat.json).",
    )
    p.add_argument(
        "--out-pattern",
        default="generated/{model}.py",
        help="Output pattern, supports {provider} and {model}. Example: generated/{model}.py",
    )
    p.add_argument(
        "--models",
        required=True,
        help=(
            "Comma-separated list of models. Use 'provider:model' to force provider; otherwise "
            "the item is treated as a MODEL and --default-provider is used.\n"
            "Examples:\n"
            "  'openrouter:anthropic/claude-sonnet-4,ollama:llama3:8b-instruct'\n"
            "  'llama-4-scout:free,claude-sonnet-4' (with --default-provider openrouter)"
        ),
    )
    p.add_argument(
        "--default-provider",
        default="openrouter",
        choices=["openrouter", "ollama"],
        help="Provider to assume when a model is given without an explicit provider prefix.",
    )

    # Shared generation knobs
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature")

    # Ollama opts
    p.add_argument("--host", default=DEFAULT_OLLAMA_HOST, help="Ollama host (e.g., http://127.0.0.1:11434)")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Ollama HTTP timeout in seconds")
    p.add_argument("--health-timeout", type=int, default=DEFAULT_HEALTH_TIMEOUT, help="Ollama health check timeout")
    p.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Retries on failure (both providers)")
    p.add_argument("--backoff", type=float, default=DEFAULT_BACKOFF, help="Backoff base seconds (both providers)")
    p.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX, help="Ollama num_ctx")
    p.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="Disable streaming for Ollama; OpenRouter uses non-stream in this script.",
    )
    p.set_defaults(stream=True)

    # OpenRouter opts
    p.add_argument("--openrouter-base-url", default=DEFAULT_OPENROUTER_BASE_URL, help="OpenRouter base URL")
    p.add_argument("--openrouter-api-key", default=None, help=f"OpenRouter API key (or set {ENV_OPENROUTER_API_KEY})")
    p.add_argument("--openrouter-timeout", type=int, default=DEFAULT_OPENROUTER_TIMEOUT, help="OpenRouter timeout")
    p.add_argument("--openrouter-site-url", default=None, help="Optional HTTP-Referer header for OpenRouter")
    p.add_argument("--openrouter-app-name", default="gen-concat-tests", help="Optional X-Title header for OpenRouter")

    p.add_argument("--print-prompt", action="store_true", help="Only print prompts and exit")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    models = parse_models_arg(args.models, default_provider=args.default_provider)

    ollama_opts = OllamaOptions(
        host=args.host,
        model="unused-here",
        temperature=args.temperature,
        timeout=args.timeout,
        health_timeout=args.health_timeout,
        retries=args.retries,
        backoff=args.backoff,
        stream=args.stream,
        num_ctx=args.num_ctx,
    )
    openrouter_opts = OpenRouterOptions(
        base_url=args.openrouter_base_url,
        api_key=args.openrouter_api_key,
        model="unused-here",
        temperature=args.temperature,
        timeout=args.openrouter_timeout,
        retries=args.retries,
        backoff=args.backoff,
        site_url=args.openrouter_site_url,
        app_name=args.openrouter_app_name,
    )

    try:
        return run_many(
            src=Path(args.src),
            out_pattern=args.out_pattern,
            models=models,
            print_prompt=args.print_prompt,
            ollama_opts=ollama_opts,
            openrouter_opts=openrouter_opts,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except requests.HTTPError as e:
        print(f"HTTP ERROR: {e}", file=sys.stderr)
        return 3
    except requests.ConnectionError as e:
        print(f"CONNECTION ERROR: {e}", file=sys.stderr)
        return 4
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}", file=sys.stderr)
        return 5


if __name__ == "__main__":
    raise SystemExit(main())