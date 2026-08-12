# Model Classification & Picker Categorization Plan

Date: 2026-08-12
Status: Completed (Level 1)
Owner: AI Agent

## Goal

The `/api/tags` and `/models` endpoints currently list every model discovered by the
scanner (HF snapshots + Ollama GGUF) as if they were interchangeable chat models.
However, the server can only actually serve **text** and **vision-to-text** inference.
Image-generation, voice/TTS, and embedding models are handled by **dedicated sibling
servers**:

- **Open Fantasia** (`open-fantasia-imagegen`, port 8765) — image generation
- **Olly Voice** (`olly-voice-server`, port 8766) — STT / TTS / voice chat

This plan adds a **modality classification** so the picker groups models by capability
and only presents as chat-usable the models this server can actually serve.

It also lays the groundwork for the server to act as a **local inference orchestrator**:
probe whether sibling servers are available, and (in a later phase) route image/voice
requests to them or prompt the user to install them.

## Scope

In scope:
- Add a `modality` field to model entries in `/api/tags` and `/models`.
- Classify each discovered model by inspecting its id / type.
- Probe availability of sibling servers (Open Fantasia :8765, Olly Voice :8766)
  and expose it in the API + UI.
- Update the web UI picker to group models by modality.
- Update the ADR to document the classification and the sibling-server architecture.

Out of scope (later phases):
- Implementing image-generation / TTS / embeddings endpoints in this server.
- Routing requests to the sibling servers from the backend (Level 2).
- A setup/install wizard for the sibling projects (Level 3).

## Modality taxonomy

| Modality | Meaning | Example model ids |
|----------|---------|-------------------|
| `chat` | Text chat (this server can serve) | `gemma2`, `qwen3`, `deepseek-r1`, `gpt2`, `llama3` |
| `vision` | Image/audio/video → text (this server can serve via `/chat/multimodal`) | `Qwen3-VL`, `gemma-4`, `Nemotron-3-Nano-Omni` |
| `imagegen` | Text → image (Open Fantasia :8765) | `FLUX.1`, `stable-diffusion`, `sd-turbo`, `diffusiongemma`, `Nemotron-Labs-Diffusion` |
| `voice` | STT/TTS/voice (Olly Voice :8766) | `CosyVoice2`, `Qwen3-TTS`, `faster-whisper`, `qwen3-embedding`? |
| `embeddings` | Embedding models (no endpoint here) | `embeddinggemma`, `nomic-embed-text` |
| `unknown` | Could not classify by name | fallback |

## Classification rules (heuristic by model id)

Use case-insensitive substring matching on the model id (lowercased):

1. **imagegen** if id contains any of:
   `flux`, `stable-diffusion`, `sd-turbo`, `sd15`, `diffusion`, `imagegen`, `image-edit`, `image_edit`, `z-image`, `z-img`, `flux2`
2. **voice** if id contains any of:
   `tts`, `cosyvoice`, `whisper`, `voice`, `speech`, `audio`, `stt`
3. **embeddings** if id contains any of:
   `embedding`, `embed`, `retrieval`
4. **vision** if id contains any of:
   `vl`, `vision`, `omni`, `gemma-4`, `qwen2.5vl`, `qwen3-vl`, `nano-omni` (and NOT matched by imagegen/voice)
5. **chat** otherwise (default for text LLMs)

Order matters: check imagegen → voice → embeddings → vision → chat.

## API changes

### `/models` and `/api/tags`

Each model entry gains a `modality` field:

```json
{
  "model": "black-forest-labs/FLUX.1-dev",
  "modality": "imagegen",
  "backend": null,
  "size_bytes": 0,
  ...
}
```

`/api/tags` maps `modality` through to each model too.

### Dedicated-server hint (optional)

For `imagegen` and `voice` modalities, include a `service` hint:

```json
{ "modality": "imagegen", "service": "open-fantasia:8765" }
{ "modality": "voice", "service": "olly-voice:8766" }
```

## UI changes (static/index.html + static/app.js)

- Group the `<select>` options by modality using `<optgroup>`.
- Label groups: `Chat`, `Vision`, `Image Gen (Open Fantasia)`, `Voice (Olly Voice)`, `Embeddings`, `Other`.
- Only `chat` and `vision` models are selectable for chat. Image-gen / voice / embeddings
  are shown in their groups but disabled (or shown with a note pointing to the right server).
- Show the selected model's modality in the model-meta panel.

## Acceptance criteria

- `/models` and `/api/tags` return a `modality` per model.
- The picker groups models by modality.
- Image-gen / voice / embeddings models are not selectable for chat (or clearly marked).
- Existing chat/vision models still work unchanged.
- Unit tests cover the classification helper.

## Sibling-server availability probe (Level 1 groundwork)

Add config for the sibling servers and a lightweight probe:

- `IMAGE_SERVER_URL` (default `http://127.0.0.1:8765`) — Open Fantasia
- `VOICE_SERVER_URL` (default `http://127.0.0.1:8766`) — Olly Voice

A helper checks each server's `/health` (short timeout) and reports availability.
The `/api/tags` and `/models` responses include `services`:

```json
"services": {
  "imagegen": { "available": true,  "url": "http://127.0.0.1:8765" },
  "voice":    { "available": false, "url": "http://127.0.0.1:8766" }
}
```

The UI uses this to:
- Show a badge next to image/voice groups: "available" or "install required".
- When a sibling is unavailable, show a prompt with the install command
  (e.g. `git clone https://github.com/fabiopacifici-bot/open-fantasia-imagegen`).

## Test plan

- Unit test for the classification function across representative ids
  (FLUX → imagegen, CosyVoice → voice, embeddinggemma → embeddings,
   Qwen3-VL → vision, gemma2 → chat).
- Unit test for the sibling availability helper (mock reachable/unreachable).
- Integration test asserting `/api/tags` entries have a `modality` field.

## Done criteria

- Classification helper implemented and unit-tested.
- `/api/tags` and `/models` expose `modality`.
- Sibling availability probe implemented and exposed in the API.
- UI picker groups by modality and shows sibling availability.
- ADR updated to document the classification and sibling-server split.
