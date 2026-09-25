# libre-semif-rs

This is Jev-like semantic decision service, inspired by [SemIf-OpenJev](https://github.com/TheoLeeCJ/SemIf-OpenJev). Prompt construction, row checks, and the JSONL record follow that project's direct mode (`direct-options-v1`, matching `encode_prompt` in `src/semif_phase1/direct.py`). This crate implements only that path, on Apple Silicon Metal through llama.cpp.

`libre-semif-rs` reads JSONL decisions and writes a new JSONL of option probabilities. No answer text is generated.

## Usage

Requires macOS on Apple Silicon. Every layer is offloaded to Metal. `--mode` accepts only `direct`. The output file must not already exist. Prompts are never truncated; `--max-tokens` (default 4096) is a hard limit.

```bash
libre-semif-rs \
  --mode direct \
  --model Qwen/Qwen3.5-4B \
  --revision <40-character-commit> \
  --gguf /path/to/model.gguf \
  --input decisions.jsonl \
  --output results.jsonl
```

`--llama-threads` sets llama.cpp CPU threads. A remote `--model` needs a 40-character revision; a local directory needs a revision label and the tokenizer plus chat template.

Each input line is one row: `id`, `state` (nonempty string, object, or array), `question`, and 2–16 options with unique `id` and `description`.

```json
{"id":"route-1","state":"Customer cannot access an account after a password reset.","question":"Which queue should handle this request?","options":[{"id":"access","description":"Account access support."},{"id":"billing","description":"Billing support."}]}
```

Each result line includes `probabilities`, `option_logits`, `prompt_sha256`, `prompt_version`, and model metadata (source, revision, GGUF checksum, backend). Probabilities are conditional on the declared options and uncalibrated as decision confidence.

## Scoring

1. Build a system message plus a user payload of evidence, criterion, and options labeled `A`–`P`.
2. Render the checkpoint chat template and tokenize with the pinned Hugging Face tokenizer. Each letter must be one token that still appends cleanly. GGUF tokenization must match.
3. One llama.cpp decode of the prompt, in 512-token chunks. Logits are requested only at the last position.
4. Keep the logits for those letter slots, softmax them into probabilities, and record `allowed_token_mass` and `full_vocab_argmax_id`.

## Layout

- `cli` — JSONL in, create-only JSONL out.
- `loader` — pinned tokenizer and local GGUF, one Metal context.
- `prompt` — chat-template render and answer-slot checks (`direct-options-v1`).
- `engine` — last-position logits; no generated tokens.
- `score` — slot logits, softmax, and the result record.
- `row` — row validation and the direct-mode messages.
- `gguf` — Qwen3 / Qwen3.5 architecture and file-type metadata only.
- `pyjson` — Python-compatible JSON so prompt hashes stay aligned with SemIf-OpenJev.
