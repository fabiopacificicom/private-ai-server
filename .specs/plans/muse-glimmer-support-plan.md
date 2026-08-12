# Muse Glimmer Support Plan (Transformers-First, No GGUF Backend)

Date: 2026-08-11
Status: Proposed
Owner: AI Agent

## Goal

Enable support for Muse Glimmer using the existing server architecture:

- Keep the current Hugging Face Hub pull flow.
- Keep Transformers as the model runtime.
- Avoid adding a new GGUF runtime/backend.

This plan intentionally does not add llama.cpp, GPT4All, or other GGUF-specific inference stacks.

## Scope

In scope:

- Audit current loading behavior.
- Define minimal code changes required for Muse Glimmer under current architecture.
- Use the multimodal route for image+text flows aligned with the model card snippet.

Out of scope:

- New backend types.
- Dedicated GGUF inference pipeline.
- New API surfaces unrelated to Muse compatibility.

## Reference Behavior From Model Card

Reference snippet (summarized):

- AutoProcessor.from_pretrained("meta-models/Muse-Glimmer-30B")
- AutoModelForMultimodalLM.from_pretrained("meta-models/Muse-Glimmer-30B", device_map="auto")
- apply_chat_template with image + text parts
- generate + decode

Implication:

- Expected runtime is Transformers multimodal classes from a model repo snapshot.
- A single GGUF file path is not sufficient for this loader pattern.

## Current Server Audit (app.py)

1. Pull path is repo-centric, not single-file GGUF-centric.

- /pull uses snapshot_download(repo_id=model_name) and stores local_path metadata.
- Evidence: app.py lines around /pull and _background_pull.

1. Text load path expects Transformers/vLLM compatible local snapshot or repo id.

- load_model requires local_path metadata and calls from_pretrained/pipeline.
- Evidence: app.py around load_model local_path resolution and AutoModelForCausalLM usage.

1. Multimodal load path currently uses AutoModelForCausalLM instead of AutoModelForMultimodalLM.

- _load_multimodal_model imports AutoProcessor and AutoModelForCausalLM.
- Evidence: app.py around _load_multimodal_model.

1. Multimodal request structure is already compatible with image+text content.

- /chat/multimodal supports messages with images/audio/video and apply_chat_template usage.
- Evidence: app.py around MultimodalMessage, _build_mm_inputs, chat_multimodal.

1. GGUF is not currently a supported loading artifact in this server path.

- No GGUF loader/runtime integration exists in app.py.
- Existing logic is based on HF repo snapshots + Transformers/vLLM model loading APIs.

## Decision

Use the non-GGUF Muse repo with current setup:

- Preferred model id for support work: meta-models/Muse-Glimmer-30B
- Endpoint path for image+text: /chat/multimodal

Do not implement direct loading from:

- E:\models\huggingface\hub\Muse-Glimmer-30B-UD-IQ2_XXS.gguf

Reason:

- That artifact is GGUF and does not map to current Transformers from_pretrained multimodal loading pattern.

## Minimal Implementation Plan

### Step 1: Multimodal Class Alignment

Update _load_multimodal_model to align with Muse reference:

- First attempt: AutoModelForMultimodalLM.from_pretrained(..., device_map="auto", trust_remote_code=True)
- Fallback: AutoModelForCausalLM.from_pretrained(...) only if the multimodal class path is unavailable for other models.

Why minimal:

- Reuses existing endpoint and processor pipeline.
- Adds no new API and no new backend.

Acceptance criteria:

- Muse model can initialize through existing multimodal loader path without introducing a GGUF backend.

### Step 2: Ensure Local Snapshot Source Is Used

In _load_multimodal_model, prefer pulled local snapshot path from model_meta when present.

- Use model_meta[model_name]["local_path"] first.
- Fallback to local-only snapshot resolution if needed.
- Avoid guessing cache paths as primary source.

Why:

- /pull already records reliable local_path in_background_pull.
- Keeps behavior consistent with load_model.

Acceptance criteria:

- After successful /pull, /chat/multimodal uses the pulled local snapshot and does not require network fetching.

### Step 3: Keep GGUF Explicitly Unsupported In Current Path

Add a narrow validation/diagnostic guard for multimodal load requests where model identifier points to a .gguf file.

- Return a clear error that this server path supports HF repo snapshots for Transformers multimodal loading.

Why:

- Prevent confusing runtime failures.
- No backend expansion.

Acceptance criteria:

- Requesting GGUF through current Transformers path returns explicit actionable message.

## Proposed Test Plan

### Test A: Pull Muse Repo Snapshot

Request:

- POST /pull
- Body: {"model":"meta-models/Muse-Glimmer-30B","init":false}

Expect:

- Job accepted.
- Job succeeds.
- local_path populated in job and model_meta.

### Test B: Initialize Muse Through Multimodal Endpoint

Request:

- POST /chat/multimodal
- Use one user message with image URL + text prompt.

Expect:

- Model loads through multimodal path.
- Response returns reply text.

### Test C: GGUF Path Rejection (Current Architecture)

Request:

- POST /chat/multimodal or /pull with model set to gguf file path.

Expect:

- Clear error message indicating GGUF artifact is unsupported by current Transformers path.

## Risks

1. Transformers version support for AutoModelForMultimodalLM may vary.

- Mitigation: keep controlled fallback to AutoModelForCausalLM for non-Muse models.

1. Muse repo may require extra optional dependencies.

- Mitigation: preserve existing missing dependency error surfacing already present in app.py.

1. Large memory footprint.

- Mitigation: keep current quantization heuristics and device_map=auto behavior; avoid architectural changes.

## Rollout Sequence

1. Implement Step 1 and Step 2 in app.py.
2. Add Step 3 validation message.
3. Run pull + multimodal smoke tests.
4. Update README multimodal supported model notes if needed.

## Explicit Non-Goals

- No direct GGUF execution support.
- No migration away from current /pull and local snapshot metadata workflow.
- No new persistence or scheduler changes.

## Done Criteria

This plan is complete when:

- Muse can be pulled and loaded using current Transformers-centric workflow.
- Multimodal requests operate using the existing endpoint contract.
- GGUF usage in this path fails fast with a clear message.
