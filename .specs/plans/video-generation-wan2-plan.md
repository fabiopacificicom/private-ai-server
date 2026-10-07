# Video Generation — Wan2.1 Text-to-Video Sibling Support

Date: 2026-08-28
Status: Completed
Owner: AI Agent

## Goal

Add native video generation to the local inference stack using **Wan-AI/Wan2.1-T2V-1.3B**,
following the existing sibling-server pattern used for image generation (Open Fantasia)
and voice (Olly Voice). Add a `[video]` button to the web UI next to Image.

## Decision

**Video generation lives in Open Fantasia (the sibling server), not in this AI
server.** Rationale:

1. **Matches the existing architecture.** This server is deliberately the text/vision
   brain. Image gen already lives in Open Fantasia; video is just another media
   sibling. Keeping the pattern consistent avoids introducing a second, conflicting
   way of doing media generation.
2. **VRAM management is already solved there.** Open Fantasia has model hot-swap
   (`_unload_flux_for_swap`, `_unload_vl`, `_unload_edit`) and a RAM guard. Video
   models are VRAM-heavy; that orchestration belongs in the server that already
   juggles competing VRAM consumers. Putting video here would force it to fight the
   LLM cache for VRAM — exactly what the sibling pattern avoids.
3. **The plumbing here is trivial to extend** because image already works via siblings.

## Scope

### Open Fantasia (sibling, :8765)
- New `src/videogen.py` — `WanPipeline` loading + MP4 encoding (imageio-ffmpeg).
- New `POST /video` endpoint in `src/server.py` — hot-swaps out image/Qwen pipelines,
  generates frames, returns raw MP4 bytes.
- `GET /health` now reports `video_loaded` / `wan-video` active state.
- `requirements.txt` — added `imageio` + `imageio-ffmpeg`.
- `SKILL.md` — documented the `/video` endpoint.

### Private AI server (this repo, :8005)
- New `POST /video` route in `src/routes/siblings.py` → proxies to Open Fantasia `/video`.
- `src/siblings.py` — `video` entry in `SIBLING_INSTALL` + `get_sibling_config`.
- `src/classify.py` — `_VIDEO_KEYWORDS` + `video` in `check_services` / `MODALITY_SERVICE`.
- `static/index.html` — `[video]` mode tab + `.gen-video` CSS.
- `static/app.js` — `generateVideo()`, `addMediaMessage('video', ...)`, `setMode` video branch, `handleSubmit` routing.
- Tests: classification, sibling config, and integration coverage for `/video`.
- `README.md` — documented the `/video` endpoint.

## API contract

### `POST {IMAGE_SERVER_URL}/video` (Open Fantasia)
```json
{
  "prompt": "a cat walking through a neon city at night",
  "quality": "mid",          // low | mid | high
  "steps": 30,
  "num_frames": 81,          // 16 fps → 81 ≈ 5s
  "guidance_scale": 6.0,
  "seed": 42,
  "enhance": false,
  "model": null              // null = Wan-AI/Wan2.1-T2V-1.3B
}
```
Response: raw `video/mp4` bytes, `X-Saved-Paths` header → `~/.openclaw/media/fantasia/videos/<ts>.mp4`.

### `POST /video` (this server → proxy)
Forwarded as-is to Open Fantasia. Returns raw MP4 bytes, or 503 with install
instructions when Open Fantasia is unreachable.

## Model notes (Wan2.1-T2V-1.3B)
- ~8.19 GB VRAM — fits RTX 4090 alongside other work.
- 480P recommended (5s clip in ~4 min on a 4090).
- Apache 2.0 license.
- Native diffusers `WanPipeline` — same library Open Fantasia already uses.

## Verification
- `python -c "import classify; classify.classify_modality('Wan-AI/Wan2.1-T2V-1.3B')"` → `video`
- `GET /services` returns `imagegen`, `video`, `voice`.
- `POST /video` proxies to Open Fantasia; returns MP4 when up, 503 + install cmd when down.
- Web UI shows the `[video]` tab; selecting it routes to `generateVideo()`.
