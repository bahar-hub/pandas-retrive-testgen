#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_concat_tests_ollama.py
Generate pytest tests for pandas.concat using a local LLM via Ollama.

Key features:
- Clean structure with an OllamaClient wrapper (health check, model check, generate).
- Configurable host, model, temperature, streaming, timeouts, retries, backoff.
- Larger context window via --num-ctx for long function bodies.
- Prints prompt (dry-run) or writes generated test code to file.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import requests


# ============================== Config & Templates ==============================

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llama3:8b-instruct"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TIMEOUT = 600  # seconds for generation
DEFAULT_HEALTH_TIMEOUT = 5
DEFAULT_RETRIES = 2
DEFAULT_BACKOFF = 2.0
DEFAULT_NUM_CTX = 8192  # generous for long sources

PROMPT_TEMPLATE = """You are a precise code generator and a senior Python test engineer.
Output ONLY valid Python code (no explanations, no markdown fences).

Task:
Given the exact source of the function `pandas.concat` (below), write a comprehensive `pytest` test file named "{outfile}" that validates its behavior across edge cases.

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
    # Prefer fenced block if present
    m = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.S)
    if m:
        return m.group(1).strip()
    # Sometimes models prefix with quotes or markdown headings; strip gently
    return text.strip()


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
            "options": {
                "num_ctx": self.opts.num_ctx
            },
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
        """
        Try streaming first (if enabled), fallback to non-stream.
        Includes simple retry with exponential backoff.
        """
        last_err: Optional[Exception] = None
        for attempt in range(1, self.opts.retries + 2):
            try:
                # Health check each attempt
                self.health()

                if self.opts.stream:
                    chunks: list[str] = []
                    for chunk in self.generate_stream(prompt):
                        chunks.append(chunk)
                    text = "".join(chunks)
                    if text.strip():
                        return text

                # Non-stream fallback (or when stream disabled)
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


# ============================== App Logic ==============================

def build_prompt(func_source: str, outfile: str) -> str:
    return PROMPT_TEMPLATE.format(func_source=func_source, outfile=outfile)


def run(src: Path, out: Path, client: OllamaClient, print_prompt: bool) -> int:
    func_source = read_text(src)
    prompt = build_prompt(func_source, out.name)

    if print_prompt:
        print(prompt)
        return 0

    # Warn if model not pulled yet (not fatal; Ollama may auto-pull in some setups)
    if not client.model_available():
        print(
            f"[WARN] Model '{client.opts.model}' not found on server. "
            f"Run:  ollama pull {client.opts.model}",
            file=sys.stderr,
        )

    raw = client.generate(prompt)
    code = extract_code_block(raw)
    if not code:
        print("ERROR: empty code from model", file=sys.stderr)
        return 2

    out.write_text(code, encoding="utf-8")
    print(f"[OK] Wrote test file: {out.resolve()}")
    return 0


# ============================== CLI ==============================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate pytest tests for pandas.concat using Ollama (local LLM)."
    )
    p.add_argument("--src", default="concat_full_source.py", help="Path to saved function body")
    p.add_argument("--out", default="test_concat_generated.py", help="Output test filename")
    p.add_argument("--host", default=DEFAULT_OLLAMA_HOST, help="Ollama host (e.g., http://127.0.0.1:11434)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Model name (e.g., llama3:8b-instruct, qwen2.5-coder:7b)")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds for generation")
    p.add_argument("--health-timeout", type=int, default=DEFAULT_HEALTH_TIMEOUT, help="Health check timeout seconds")
    p.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Number of retries on failure")
    p.add_argument("--backoff", type=float, default=DEFAULT_BACKOFF, help="Backoff base (seconds) between retries")
    p.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX, help="Context window tokens (num_ctx)")
    p.add_argument(
        "--no-stream", dest="stream", action="store_false",
        help="Disable streaming; use a single non-stream request"
    )
    p.add_argument("--print-prompt", action="store_true", help="Only print the constructed prompt and exit")
    p.set_defaults(stream=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    opts = OllamaOptions(
        host=args.host,
        model=args.model,
        temperature=args.temperature,
        timeout=args.timeout,
        health_timeout=args.health_timeout,
        retries=args.retries,
        backoff=args.backoff,
        stream=args.stream,
        num_ctx=args.num_ctx,
    )
    client = OllamaClient(opts)

    try:
        return run(
            src=Path(args.src),
            out=Path(args.out),
            client=client,
            print_prompt=args.print_prompt,
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