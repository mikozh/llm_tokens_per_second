# Ollama TPS Benchmark (OpenAI client)

A tiny, single-file benchmark that sends concurrent chat requests to your **Ollama** instance (via the **OpenAI** Python client), then reports **mean**, **median**, and **5/95% quantiles** of **tokens per second (TPS)**.

Each user prompt includes a random number to reduce KV-cache effects. If the response includes `usage.completion_tokens`, those are used; otherwise the script counts tokens with **tiktoken** (required—no heuristics).

---

## Requirements

* Python 3.9+
* Packages:

  ```bash
  pip install openai numpy tiktoken
  ```
* A running Ollama with the OpenAI-compatible API (e.g. `http://localhost:11434/v1`):

  ```bash
  ollama serve
  ```

> Any non-empty API key works for local Ollama (e.g. `ollama`).

---

## Script

Place the script in your repo as `ollama_tps_openai.py` (name is arbitrary).
It uses the OpenAI client pointed at your Ollama endpoint.

---

## Usage

```bash
python ollama_tps_openai.py <threads> [options]
```

### Common options

* `--base-url` (str): OpenAI-compatible endpoint, default `http://localhost:11434/v1`
* `--api-key` (str): API key, default `OPENAI_API_KEY` env or `ollama`
* `--model` (str): Model name, default `llama3.1:8b`
* `--requests-per-thread` (int): Requests per thread (default 5)
* `--timeout` (float): Per-request timeout seconds (default 120)
* `--max-tokens` (int): Generation cap (default 128)
* `--temperature` (float): Sampling temperature (default 0.2)
* `--prompt` (str): Base user prompt (script appends a random number)
* `--system` (str): Optional system prompt
* `--encoding` (str): tiktoken encoding, default `cl100k_base`
* `--force-tiktoken`: Ignore `usage.completion_tokens` and always count via tiktoken

See `python ollama_tps_openai.py -h` for all flags.

---

## Examples

```bash
# 8 concurrent threads, defaults
python ollama_tps_openai.py 8

# 16 threads, different model, more requests per thread, shorter outputs
python ollama_tps_openai.py 16 --model qwen2.5:7b --requests-per-thread 10 --max-tokens 64

# Custom endpoint and API key
python ollama_tps_openai.py 4 --base-url http://127.0.0.1:11434/v1 --api-key ollama
```

---

## Sample output

```
Starting benchmark: threads=8, requests_per_thread=5, total=40
Endpoint=http://localhost:11434/v1, model=llama3.1:8b, timeout=120.0s, max_tokens=128
Sending requests...

Results:
  Samples: 40 ok / 0 failed (total 40)
  TPS (tokens/sec):
    mean=47.31   median=46.82   p05=39.12   p95=57.90
```

If some requests fail, the script prints a short error summary.

---

## How TPS is computed

1. **Timer**: Wall-clock elapsed time around the `chat.completions.create` call.
2. **Token count**:

   * Prefer `response.usage.completion_tokens` (when present).
   * Otherwise, count tokens with **tiktoken** on the assistant text using the selected `--encoding`.
3. **TPS** = `completion_tokens / elapsed_seconds`.

> The user message includes a random number to reduce KV-cache reuse, making results more representative.

---

## Tips & notes

* Throughput depends on model, hardware, load, sampling params, and `--max-tokens`.
* Use larger `threads` × `--requests-per-thread` for more stable statistics.
* If your model doesn’t report `usage`, use `--force-tiktoken` for consistency across runs.
* Match `--encoding` to your tokenizer if you know it; `cl100k_base` is a reasonable default for many LLMs.

---

## Troubleshooting

* **Connection refused / timeouts**: Ensure `ollama serve` is running and the `--base-url` points to `/v1`.
* **401/Unauthorized**: Provide any non-empty `--api-key` (e.g., `--api-key ollama`) or set `OPENAI_API_KEY`.
* **Zero completion tokens**: Increase `--max-tokens` or check that the model can generate text for your prompt.
* **Import errors**: `pip install openai numpy tiktoken` in the same environment you’re running.

---

