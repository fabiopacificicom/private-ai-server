# Sibling-Server Orchestration — Level 2 & 3 Plan

Date: 2026-08-12
Status: Completed (Level 2 & 3)
Owner: AI Agent

## Goal

This server is the **text + vision-to-text brain** of a local inference microservices
stack. Image generation (Open Fantasia :8765) and voice (Olly Voice :8766) are handled
by dedicated sibling servers. Level 1 added modality classification + availability
probing. This plan implements:

- **Level 2**: proxy `/generate` (image) and `/tts` (voice) requests to the sibling
  servers, and prompt the user with install instructions when a sibling is down.
- **Level 3**: a setup wizard (UI + helper) that shows each sibling's install command
  and status.

## Scope

In scope:
- New `src/siblings.py` — HTTP proxy helpers for image generation and TTS.
- New `src/routes/siblings.py` — `/generate` and `/tts` endpoints that route to the
  matching sibling server, returning a clear install prompt when it's unreachable.
- A `GET /services` (or reuse `/models` `services`) endpoint with install instructions.
- UI: a "Services" panel in the sidebar showing Open Fantasia / Olly Voice status and
  a button to copy the install command; a setup wizard modal.
- Unit tests for the proxy helper (mock reachable/unreachable, payload forwarding).
- Integration tests for the new endpoints.
- ADR + README + plan updates.

Out of scope:
- Actually implementing image/voice inference in this server.
- Streaming image/voice generation (Level 2.5 / future).

## Sibling API contract (assumed)

Both siblings expose simple HTTP APIs. This server proxies generic JSON payloads
through to them:

- **Open Fantasia** (imagegen) — `POST {IMAGE_SERVER_URL}/generate`
  - Body: `{ "model": "...", "prompt": "...", ... }`
  - Response: JSON with an image (base64 or URL).
- **Olly Voice** (voice) — `POST {VOICE_SERVER_URL}/tts`
  - Body: `{ "model": "...", "text": "...", ... }`
  - Response: audio (base64 or URL).

Because the exact sibling schemas may evolve independently, the proxy forwards the
client's JSON body as-is and returns the sibling's JSON response. This keeps this
server decoupled from sibling internals.

## API changes

### `POST /generate` (imagegen → Open Fantasia)

```json
// request
{ "model": "black-forest-labs/FLUX.1-dev", "prompt": "a red cat", "width": 512 }
// response (200) — proxied from sibling
{ "images": ["data:image/png;base64,..."] }
// response (503) — sibling down
{ "detail": "Open Fantasia (:8765) is not running. Install: git clone ..." }
```

### `POST /tts` (voice → Olly Voice)

```json
// request
{ "model": "CosyVoice2", "text": "hello world" }
// response (200) — proxied from sibling
{ "audio": "data:audio/wav;base64,..." }
// response (503) — sibling down
{ "detail": "Olly Voice (:8766) is not running. Install: git clone ..." }
```

### `GET /services`

Returns availability + install instructions for each sibling:

```json
{
  "imagegen": { "available": true,  "url": "http://127.0.0.1:8765", "service": "Open Fantasia",
                "install": "git clone https://github.com/fabiopacifici-bot/open-fantasia-imagegen" },
  "voice":    { "available": false, "url": "http://127.0.0.1:8766", "service": "Olly Voice",
                "install": "git clone https://github.com/fabiopacificicom/olly-voice-server" }
}
```

## Module design

### `src/siblings.py`

```python
def proxy_json(url, payload, timeout=60) -> dict:
    """POST a JSON payload to a sibling server and return the parsed response.
    Raises SiblingUnavailable if the server can't be reached."""

class SiblingUnavailable(Exception):
    """Raised when a sibling server is unreachable."""
```

### `src/routes/siblings.py`

- `POST /generate` — validates payload, checks `classify`-style service, proxies to
  `IMAGE_SERVER_URL`, returns image response or 503 with install prompt.
- `POST /tts` — same for `VOICE_SERVER_URL`.
- `GET /services` — returns `check_services()` enriched with install instructions.

## UI changes

- Sidebar "Services" panel:
  - Open Fantasia: status badge (available / install required) + copy-install button.
  - Olly Voice: same.
- Setup wizard modal (Level 3): lists both siblings, their status, install command,
  and a "copy" button. Triggered from the Services panel.

## Install commands

- Open Fantasia: `git clone https://github.com/fabiopacifici-bot/open-fantasia-imagegen`
- Olly Voice: `git clone https://github.com/fabiopacificicom/olly-voice-server`

## Acceptance criteria

- `POST /generate` routes to Open Fantasia when available; returns install prompt (503)
  when down.
- `POST /tts` routes to Olly Voice when available; returns install prompt (503) when down.
- `GET /services` returns availability + install commands.
- UI shows sibling status and a setup wizard with copyable install commands.
- Unit + integration tests pass.

## Test plan

- Unit: `proxy_json` success + `SiblingUnavailable` on connection error (mock).
- Integration: `/services` returns install commands; `/generate` and `/tts` return 503
  with install prompt when siblings are unreachable (they won't be running in CI).

## Done criteria

- Sibling proxy + endpoints implemented.
- UI Services panel + setup wizard.
- Tests pass.
- ADR / README / plan updated.
- Committed and pushed.
