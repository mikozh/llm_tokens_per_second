#!/usr/bin/env python3
"""
Ollama TPS via OpenAI Client (single file)

Requires:
  pip install openai numpy tiktoken

Default endpoint assumes Ollama's OpenAI-compatible API:
  ollama serve  (exposes http://localhost:11434/v1)

Examples:
  python ollama_tps_openai.py 8
  python ollama_tps_openai.py 16 --model llama3.1:8b --requests-per-thread 10 --max-tokens 128
  python ollama_tps_openai.py 4 --base-url http://127.0.0.1:11434/v1 --api-key ollama

Notes:
- Uses OpenAI Python library with a custom base_url to hit Ollama.
- If response.usage.completion_tokens is available, uses that directly.
- Otherwise counts tokens with tiktoken on the assistant text (required; no heuristics).
- Each request includes a random number in the user prompt to mitigate KV cache effects.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import math
import os
import random
import statistics
import sys
import threading
import time
from typing import List, Optional, Tuple

import numpy as np
import tiktoken
from openai import OpenAI
from openai.types.chat import ChatCompletion

DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY = "ollama"  # any non-empty string works for local Ollama
DEFAULT_PROMPT = "Write 2-3 concise sentences about concurrency and thread safety."
DEFAULT_MAX_TOKENS = 128

def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark tokens/sec against an Ollama instance via OpenAI client.")
    p.add_argument("threads", type=int, help="Number of concurrent worker threads.")
    p.add_argument("--requests-per-thread", type=int, default=5,
                   help="How many requests each thread will issue (default: 5).")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL}).")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL,
                   help=f"OpenAI-compatible base URL (default: {DEFAULT_BASE_URL}).")
    p.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", DEFAULT_API_KEY),
                   help="API key (default: env OPENAI_API_KEY or 'ollama').")
    p.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout in seconds (default: 120).")
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="Base user prompt (random number is appended).")
    p.add_argument("--system", default=None, help="Optional system prompt.")
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                   help=f"Max tokens to generate (default: {DEFAULT_MAX_TOKENS}).")
    p.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (default: 0.2).")
    p.add_argument("--encoding", default="cl100k_base",
                   help="tiktoken encoding to use when counting tokens (default: cl100k_base).")
    p.add_argument("--force-tiktoken", action="store_true",
                   help="Ignore response.usage and always count with tiktoken.")
    return p.parse_args()

def get_client(base_url: str, api_key: str, timeout: float) -> OpenAI:
    # The OpenAI client constructor accepts base_url/api_key; timeout applies per request.
    # We'll store timeout and pass it into each request.
    client = OpenAI(base_url=base_url, api_key=api_key)
    client._tps_timeout = timeout  # stash for later
    return client

def encode_token_count(encoding_name: str, text: str) -> int:
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text or ""))

def one_request(
    client: OpenAI,
    model: str,
    base_prompt: str,
    sys_prompt: Optional[str],
    encoding_name: str,
    max_tokens: int,
    temperature: float,
    force_tiktoken: bool,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Returns (tokens_per_second, error_message).
    """
    # Randomize user content to weaken KV-cache reuse
    rnd = random.randrange(10**9, 10**10)
    user_content = f"{base_prompt}\n\nRandom number: {rnd}"

    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": user_content})

    try:
        start = time.perf_counter()
        # Pass per-call timeout; newer openai client accepts 'timeout' kwarg.
        resp: ChatCompletion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
            timeout=getattr(client, "_tps_timeout", None),
        )
        elapsed = max(1e-12, time.perf_counter() - start)
    except Exception as e:
        return None, f"request error: {e}"

    # Prefer usage if present and not forced off
    comp_tokens = None
    try:
        if not force_tiktoken and hasattr(resp, "usage") and resp.usage and resp.usage.completion_tokens is not None:
            comp_tokens = int(resp.usage.completion_tokens)
    except Exception:
        comp_tokens = None

    if comp_tokens is None:
        # Count tokens in the assistant output
        try:
            assistant_text = (resp.choices[0].message.content or "") if resp.choices else ""
            comp_tokens = encode_token_count(encoding_name, assistant_text)
        except Exception as e:
            return None, f"token count error: {e}"

    if comp_tokens <= 0:
        return None, "zero completion tokens"

    tps = float(comp_tokens) / elapsed
    return tps, None

def main() -> None:
    args = build_args()

    total = args.threads * args.requests_per_thread
    print(f"Starting benchmark: threads={args.threads}, requests_per_thread={args.requests_per_thread}, total={total}")
    print(f"Endpoint={args.base_url}, model={args.model}, timeout={args.timeout}s, max_tokens={args.max_tokens}")
    print("Sending requests...\n", flush=True)

    client = get_client(args.base_url, args.api_key, args.timeout)

    lock = threading.Lock()
    tps_values: List[float] = []
    errors: List[str] = []

    def task():
        tps, err = one_request(
            client=client,
            model=args.model,
            base_prompt=args.prompt,
            sys_prompt=args.system,
            encoding_name=args.encoding,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            force_tiktoken=args.force_tiktoken,
        )
        with lock:
            if tps is not None and math.isfinite(tps):
                tps_values.append(tps)
            elif err:
                errors.append(err)

    with futures.ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs = [ex.submit(task) for _ in range(total)]
        for _ in futures.as_completed(futs):
            pass

    ok = len(tps_values)
    fail = len(errors)

    if ok == 0:
        print("All requests failed.\n", file=sys.stderr)
        for e in errors[:10]:
            print("  -", e, file=sys.stderr)
        sys.exit(2)

    # Stats with NumPy
    arr = np.array(tps_values, dtype=float)
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    p05, p95 = np.quantile(arr, [0.05, 0.95], method="linear")

    print("Results:")
    print(f"  Samples: {ok} ok / {fail} failed (total {ok+fail})")
    print(f"  TPS (tokens/sec):")
    print(f"    mean={mean:.2f}   median={median:.2f}   p05={p05:.2f}   p95={p95:.2f}")

    if fail:
        # show top error types
        from collections import Counter
        counts = Counter(errors)
        print("\nErrors (top 5):")
        for i, (msg, cnt) in enumerate(counts.most_common(5), start=1):
            print(f"  {i}. {msg} (x{cnt})")

if __name__ == "__main__":
    main()
