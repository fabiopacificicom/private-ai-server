# GGUF Support Plan (Alongside Current Transformers Architecture)

Date: 2026-08-11
Status: Draft
Owner: AI Agent

## Goal

Add GGUF inference support for low-resource environments while preserving the current architecture and behavior for existing Hugging Face Transformers/vLLM models.

Key objective:
- Keep current `/pull`, `/chat`, `/chat/multimodal`, `/jobs`, `/models` contracts stable.
- Introduce GGUF as an additional backend, not a replacement.

## Non-Goals

- No removal of current Transformers or vLLM flow.
- No breaking API changes for existing clients.
- No mandatory migration of multimodal workloads to GGUF.
- No destructive model-cache changes.

## Current State (Verified)

- `app.py` currently supports:
  - vLLM backend
  - Transformers backend
  - Multimodal via Transformers (`/chat/multimodal`)
- `.gguf` is explicitly rejected by current pull/chat multimodal flow.
- Quantization policy is currently bitsandbytes 4-bit for Transformers (auto threshold behavior).

## Proposed Architecture

Add a new backend lane:
- `backend = "gguf_llama_cpp"`

Model metadata extension (stored in `model_meta`):
- `backend`: `transformers_pipeline | vllm | multimodal | gguf_llama_cpp`
- `model_type`: `hf_repo | gguf_file`
- `local_path`: path to snapshot dir (HF) or `.gguf` file (GGUF)
- `gguf_variant` (optional): parsed from filename (e.g., `Q4_K_M`, `IQ2_XXS`)
- `n_gpu_layers`, `n_ctx` (optional runtime config snapshot)

No DB schema change required for first iteration.

## Phase Plan

## Phase 1: GGUF Text Chat MVP

### Scope

- Add GGUF runtime support for `/chat` only.
- Keep `/chat/multimodal` Transformers-only.

### Changes

1. Dependencies and runtime checks
- Add optional dependency: `llama-cpp-python`.
- Add capability detection at startup similar to current backend probes.

2. Loader path
- Add `load_model_gguf(model_ref: str)` and route through existing `load_model` dispatcher.
- Accept local `.gguf` path and `hf://.../*.gguf` references for pull flow.

3. Pull flow
- Extend `/pull` to accept payload patterns:
  - `{ "model": "path\\to\\model.gguf" }`
  - `{ "model": "hf://org/repo/file.gguf" }`
- Resolve remote GGUF to local file path under `HF_HOME` (or configured GGUF cache path).
- Persist `model_meta[model_name]` with `backend="gguf_llama_cpp"` and `local_path`.

4. Chat path
- In `/chat`, route generation by backend:
  - existing behavior for vLLM/Transformers
  - new llama.cpp generation for GGUF
- Return same response shape as today.

5. Streaming
- Add GGUF streaming in `_stream_chat_response` producing identical SSE chunk format.

### Acceptance Criteria

- Can pull/load a GGUF file and run `/chat` end-to-end.
- Existing HF repo models still work unchanged.
- Existing response contracts remain stable.

## Phase 2: Backend Selection and Policy

### Scope

- Improve operator control for mixed environments.

### Changes

1. Pull-time backend hint
- Support optional payload key:
  - `backend`: `auto | transformers | gguf`
- Auto selection rules:
  - If `.gguf` input -> gguf backend
  - If HF repo id -> existing Transformers/vLLM logic

2. New environment knobs
- `GGUF_N_GPU_LAYERS` (default tuned for 16GB VRAM)
- `GGUF_N_CTX`
- `GGUF_THREADS`
- `GGUF_SPLIT_MODE` (optional)

3. `/models` and `/status` enrichments
- Expose backend and model type for clear observability.

### Acceptance Criteria

- Operators can explicitly force backend per pull.
- 16GB machines can use GGUF path without touching Transformers flow.

## Phase 3: Operational Hardening

### Scope

- Stability, observability, tests, docs.

### Changes

1. Error handling
- Clear actionable errors for:
  - missing llama-cpp runtime
  - unsupported GGUF file
  - malformed `hf://` reference

2. Tests
- Add focused tests for:
  - GGUF pull parsing
  - backend routing in `/chat`
  - streaming parity

3. Documentation
- Add GGUF section in README:
  - what works
  - what remains Transformers-only (multimodal)
  - low-resource deployment examples

### Acceptance Criteria

- New GGUF flow has explicit diagnostics and tested happy path.

## API Compatibility Strategy

No breaking changes:
- Existing endpoints and response schemas stay unchanged.
- Optional request fields only (`backend` hint).
- Existing client integrations continue to work without modification.

## Risks and Mitigations

1. Build complexity on Windows for `llama-cpp-python`
- Mitigation: document pinned wheels/build matrix and fallback guidance.

2. Performance variability by GGUF quant variant
- Mitigation: add runtime metadata and recommended presets.

3. Confusion between multimodal and text GGUF capabilities
- Mitigation: keep `/chat/multimodal` explicitly Transformers-only in early phases.

## Rollout Checklist

1. Implement Phase 1 in feature branch.
2. Validate with one known GGUF model on 16GB GPU machine.
3. Run existing chat regression checks for Transformers models.
4. Enable Phase 2 controls.
5. Ship docs and tests (Phase 3).

## Suggested First Implementation Slice

- Add runtime probe for `llama_cpp`.
- Add GGUF `model_ref` parsing in `/pull`.
- Add GGUF load and generate path in `/chat` non-streaming.
- Defer streaming to immediate follow-up commit if needed.

This keeps the first merge small and low-risk.
