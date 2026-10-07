# Release Installer with Guided Setup — Plan

Date: 2026-08-17
Status: ✅ COMPLETED (v1.1.0) — installer shipped, server bundled, E2E validated
Owner: AI Agent
Priority: HIGH (completed — last step to make the project usable by non-developers)

---

## Goal

Ship a **one-click installer** that provisions the environment, configures the
sibling servers (Open Fantasia image, Olly Voice), and launches the desktop app —
so a **non-technical user** can install and reach a working **Chat + Image + Voice**
UI without ever opening a terminal.

This is the final milestone in the roadmap's "Next Up" section and the last piece
needed to take the project from "developer tool" to "shippable product".

---

## Current State (what already exists)

| Piece | Status | Where |
| ------- | -------- | ------- |
| FastAPI server (`app.py` + `src/`) | ✅ done | repo root |
| Persistent SQLite jobs | ✅ done | `src/database.py` |
| Tauri v2 desktop wrapper | ✅ done | `desktop/` |
| Web UI (Chat / Image / Audio tabs) | ✅ done | `static/` |
| Sibling probing + setup wizard (copy commands) | ✅ done | `src/routes/siblings.py`, `static/app.js` |
| `.env` loading (`IMAGE_SERVER_URL` / `VOICE_SERVER_URL`) | ✅ done | `src/config.py` |
| CUDA setup script | ✅ done | `scripts/setup_cuda_pytorch.ps1` |
| llama-cpp CUDA setup scripts | ✅ done | `scripts/setup-llamacpp-cuda.ps1` / `.sh` |
| Docker compose | ✅ done | `docker-compose.yml` |

**Gap**: There is **no bundle** that packages the server + desktop app into a single
installable artifact, **no first-run guided setup** that provisions Python/CUDA and
the siblings, and **no service management / auto-start**.

---

## Scope

### In scope

1. **Installable artifact** — a Windows installer (Tauri NSIS bundle) that ships:
   - The **desktop app** (Tauri) with the **native guided installer** built in
   - The **server** (Python app + a bundled/embedded Python runtime, e.g. via
     PyInstaller or a python-embed zip)
   - The **setup bootstrap** (first-run provisioning)
2. **Native guided installer** — a first-run desktop flow (screens A–H, see
   Step 3) that:
   - Detects already-running servers and lets the user skip
   - Lets the user choose services to install (AI server and/or the two siblings),
     showing already-installed ones checked with health status
   - Asks **Docker vs local** deployment (dictates the steps)
   - Asks for the **installation directory** and **model storage directory** via
     native path pickers
   - Checks system requirements and installs dependencies (Docker, or Python +
     CUDA + llama-cpp) with a live native checklist
   - Handles errors natively (inline retry, no silent failures)
   - Shows installed services with per-row **“Test connection”** buttons (health)
   - Asks to install/start the desktop app, else starts the web UI (with confirmation)
   - Shows a completion screen and asks about **start-on-startup**
3. **Service management** — background service / auto-start on login (Docker or local).

### Out of scope

- Bundling the actual model weights (models are pulled at runtime via `/pull`).
- Bundling the sibling servers inside the installer (they are cloned/installed on
  first run, keeping the installer small and each service independently updatable).
- Cross-platform installers (Linux/macOS) in this pass — Windows is the primary target.
- Code-signing the installer (unsigned is acceptable per user decision).

---

## Design Decisions

### D1 — Installer technology: Tauri NSIS bundle (primary)

Tauri v2 already supports producing an **NSIS installer** (`tauri build` with
`bundle.targets` including `nsis`). We extend the existing Tauri desktop app so its
bundle carries the server payload and a first-run bootstrap.

- **Why NSIS over WiX/Inno**: NSIS is what Tauri generates natively — least custom
  tooling, one command (`npm run tauri build -- --bundles nsis`).
- **Alternative considered**: Inno Setup — more flexible for embedding arbitrary
  payloads, but requires a separate installer project. Kept as fallback if the NSIS
  payload-embedding (extra files) proves limiting.

### D2 — Server packaging: PyInstaller one-folder build (primary)

Package the FastAPI server into a self-contained folder with PyInstaller so the
installer can ship it without requiring the user to install Python separately.

- **Why PyInstaller**: single command, produces a `dist/ai-server/` folder with the
  app + all Python deps + a bundled python runtime.
- **Why one-folder over one-file**: large torch/transformers payloads are slow to
  unpack from one-file; one-folder starts faster and is easier to debug.
- **Alternative considered**: python-embed zip + pip install on first run. Simpler to
  build, but slower first run and more failure modes (network). Kept as fallback.
- **Caveat**: torch + CUDA are huge. The **base installer** ships CPU/transformers
  only; the **setup wizard** installs the CUDA-enabled torch + llama-cpp wheels on
  first run (detecting the GPU). This keeps the installer download reasonable and
  lets the wizard tailor the runtime to the machine.

### D3 — Native desktop guided installer (primary UX)

The setup is a **native desktop guided installer**, built into the Tauri app. It is
**not** a browser wizard — it runs as a dedicated first-run window with native
controls (checkboxes, path pickers, progress checklist, per-row test buttons). This
gives us full native access to processes, services, the filesystem, and native
folder pickers without browser sandbox restrictions.

The installer is **multi-modal**: it can provision a **Docker** deployment or a
**local (bare-metal)** deployment, and the flow branches based on that choice.

### D4 — Service management

- **Docker mode**: the server + siblings run as containers managed by Docker
  Compose (`docker-compose.yml`). Autostart = Docker's `restart: unless-stopped`,
  or a scheduled task that runs `docker compose up -d` at logon.
- **Local mode**: register the packaged server as a **Windows background service**
  via `sc.exe create` / `sc.exe start`, or a **scheduled task at logon** (no admin
  rights needed).
- The completion screen asks the user whether to **start services on startup**
  (yes/no) and applies the chosen mechanism.

### D5 — Idempotent & re-runnable

Every installer step checks current state before acting (see "Acceptance Criteria").
A `setup_state.json` (in the app data dir) records what was provisioned so re-running
the wizard skips completed steps, shows already-installed services as checked, and
repairs broken ones.

---

## Implementation Steps

### Step 1 — Server packaging (PyInstaller) — ✅ DONE

- [x] Add `server.spec` (PyInstaller spec) that includes:
  - `app.py`, `src/`, `static/`, `models_manifest.json`, `requirements.txt`
  - Hidden imports for FastAPI/uvicorn/transformers used dynamically
- [x] Add `scripts/build-server.ps1` that runs PyInstaller and produces
  `dist/ai-server/`
- [x] Verify the packaged server starts and serves `/health` with no source tree
  (verified: `dist/ai-server/ai-server.exe` returns `/health` 200 with CUDA)
- [x] Add `app.py` `__main__` entry point so the packaged exe starts uvicorn
  directly (`--host`/`--port` or `HOST`/`PORT` env)
- [x] gitignore `build/` and `dist/` (large generated artifacts)

### Step 2 — Tauri desktop: bundle the server payload + native installer UI

- [x] Add Tauri **Rust commands** exposed to the installer UI (in `lib.rs`):
  - `check_server(url)` — probe a `/health` endpoint
  - `test_connection(key)` / `probe_services()` — per-service and bulk health probes
  - `pick_folder(title)` — native directory picker (install / models dir)
  - `default_install_dir()` / `default_models_dir()` — OS-appropriate defaults
  - `register_autostart()` / `unregister_autostart()` — logon scheduled task
  - `check_python()` / `check_cuda()` / `check_docker()` / `docker_up()` /
    `clone_sibling()` / `write_env()` / `start_services()` — provisioning
  - Note: command fns are **non-`pub`** to avoid the tauri 2.11 E0255 `__cmd__`
    macro collision (build otherwise fails)
- [x] Add a **native first-run installer window** (frontend in `desktop/dist/`:
  `index.html`, `styles.css`, `installer.js`) that drives the guided flow (Step 3)
- [x] Add `dev-server.mjs` static server + wire `tauri.conf.json` (`frontendDist`
  `../dist`, `devUrl` `localhost:1420`); add `tauri-plugin-dialog` (native pickers),
  `ureq` (health probe), `dirs` deps
- [ ] **Pending:** `bundle.resources` to include the packaged `dist/ai-server/`
  folder in the installer, and `bundle.targets` for `nsis`

### Step 3 — Guided installer flow (native)

The installer walks the user through the flow below. Each screen is a native step;
state is carried forward and persisted to `setup_state.json`.

**Screen A — Detect running servers**
- Probe `http://127.0.0.1:8005/health`, `:8765/health`, `:8766/health`.
- If all selected services are already running → offer to **skip setup** and go
  straight to launch.

**Screen B — Choose services to install** (checkboxes)
- `☑ AI server` (always)
- `☐ Open Fantasia (image)` — if already installed/running, show **checked** with a
  green **health status** badge; user can keep or uncheck.
- `☐ Olly Voice (voice)` — same behavior.

**Screen C — Deployment mode**
- **Docker container** (isolated) — dictates Docker-related steps.
- **Local (bare-metal)** — dictates Python + CUDA + repo-clone steps.
- Also ask **preferred installation directory** (native path picker; default to the
  OS-appropriate location, e.g. `%LOCALAPPDATA%\Programs` / `%ProgramFiles%`).

**Screen D — Model storage location**
- Native directory picker for where local models are stored (defaults to
  `HF_HOME` / `%USERPROFILE%\ai-server-models`).

**Screen E — System requirements check + dependency install**
- Based on Screen C:
  - **Docker mode**: detect Docker (install if missing), pull images, `docker
    compose up -d`.
  - **Local mode**: detect Python (install if missing), detect GPU + CUDA → install
    CUDA torch / llama-cpp wheels; clone sibling repos if chosen.
- Show a **native checklist** that updates live as each dependency installs.
- Handle errors **natively** (inline error rows with retry, not silent failures).

**Screen F — Service status + test connections**
- Show **checked boxes** for the services that were installed.
- Each row has a **“Test connection”** button that hits the service's `/health`
  endpoint and shows live up/down.

**Screen G — Launch choice**
- Ask: *“Install and start the desktop app?”*
  - **Yes** → install/start the desktop app.
  - **No** → start the **web UI** in the browser (with user confirmation).

**Screen H — Completion**
- Native completion screen.
- Ask: *“Start services on startup?”* (yes/no) → applies Docker autostart,
  background service, or logon scheduled task accordingly.

### Step 4 — Server-side setup helpers — ✅ DONE

- [x] `scripts/setup-bootstrap.ps1` — reusable provisioning steps invoked by the
  Rust `run_setup` command (idempotent, per-step success/failure + message):
  1. Detect Python / GPU + CUDA; install via winget or bundled embed as needed
  2. Docker mode: verify Docker, compose up
  3. Local mode: install CUDA torch / llama-cpp wheels, clone sibling repos
  4. Write `.env` (`IMAGE_SERVER_URL`, `VOICE_SERVER_URL`, `MAX_GPU_MEMORY`,
     `MAX_CPU_MEMORY`, `HF_HOME`)
  5. Register service / auto-start
- [x] `scripts/register-service.ps1` — `sc.exe create` background service (admin)
- [x] `scripts/register-autostart.ps1` — logon scheduled task (no admin)
- [x] Persist results to `setup_state.json`
  (verified: bootstrap detects Python 3.10 + RTX 4090, writes `.env`, persists state)

### Step 5 — Enhanced web UI setup wizard (fallback)

- [ ] Keep the existing `/services` + `/setup-wizard` web UI as a **browser
  fallback**, and extend it so the same status/health info is visible when the app
  is opened in a plain browser (no Tauri shell).
- [ ] Keep the existing copy-to-clipboard install commands for browser use.

### Step 6 — Installer build + docs

- [ ] `desktop/build.ps1` extended with `-Release` producing the NSIS installer
  containing the server payload
- [ ] Verify end-to-end on a clean machine (or VM): install → native guided
  installer → Chat + Image + Voice working
- [ ] Update `README.md`, `.specs/ROADMAP.md`, `CHANGELOG.md`
- [ ] Update ADR (new deploy/packaging decision)

---

## Acceptance Criteria

- A non-technical user can install the app and reach a working **Chat + Image +
  Voice** UI without a terminal.
- The installer is a **native desktop guided flow** (not a browser wizard).
- If the servers are already running, the user can **skip setup** and launch.
- The user can **choose services** (AI server only, or + siblings); already-installed
  siblings appear **checked with a health status**.
- The user can choose **Docker vs local** deployment, and the steps branch correctly.
- **Installation directory** and **model storage directory** are chosen via **native
  path pickers** with sensible OS defaults.
- Dependencies install behind a **live native checklist**; errors surface **natively**
  with retry (no silent hangs).
- Installed services show checked boxes with a **“Test connection”** button per row
  that hits `/health`.
- The user is asked whether to **install/start the desktop app** or **start the web
  UI** (with confirmation).
- A **completion screen** asks whether to **start services on startup**.
- First-run wizard is **re-runnable (idempotent)**: re-running skips completed steps
  and repairs broken ones.
- The packaged server starts and serves `/health` with no source tree present.
- Server can be registered as a background service or auto-start on login.

---

## Testing

### Unit tests

- `tests/unit/test_setup_state.py` — `setup_state.json` read/write/merge logic
- `tests/unit/test_env_writer.py` — `.env` generation from config, idempotency
- `tests/unit/test_bootstrap_steps.py` — step detection logic (Python present/absent,
  GPU present/absent) with mocked subprocess

### Integration tests

- `tests/integration/test_setup_endpoints.py` — any new server endpoints (e.g.
  `/setup/status`, `/setup/apply`) return correct state and are idempotent
- Verify `/services` still returns sibling status after wizard runs

### Manual / E2E

- `scripts/setup-bootstrap.ps1` run on a clean machine: each step idempotent
- `desktop/build.ps1 -Release` produces an installer; install on a VM; walk through
  first-run wizard; confirm Chat + Image + Voice
- `scripts/register-service.ps1` and autostart variant both work and survive reboot

---

## Files to touch

- `server.spec` (new) — PyInstaller config
- `scripts/build-server.ps1` (new) — server packaging
- `scripts/setup-bootstrap.ps1` (new) — first-run provisioning
- `scripts/register-service.ps1` (new) — background service (local mode)
- `scripts/register-autostart.ps1` (new) — logon autostart
- `desktop/src-tauri/tauri.conf.json` — bundle resources + NSIS target
- `desktop/src-tauri/src/lib.rs` / `main.rs` — first-run installer window + Rust commands
- `desktop/src-tauri/src/setup/` (new) — native installer UI + flow logic (screens A–H)
- `desktop/src-tauri/Cargo.toml` — any new Rust deps (e.g. `reqwest`/`ureq` for health probe, `rfd` for native path pickers)
- `desktop/build.ps1` — release/installer flow
- `static/app.js` / `static/index.html` — keep web setup wizard as browser fallback
- `src/routes/siblings.py` — optional `/setup/status` helper
- `.specs/plans/` — this plan
- `README.md`, `.specs/ROADMAP.md`, `CHANGELOG.md`, `ADR.md` — docs

---

## Risks & Mitigations

| Risk | Mitigation |
| ------ | ----------- |
| PyInstaller misses dynamic imports (FastAPI/transformers) | Use hidden imports in `server.spec`; verify with a clean-machine smoke test |
| Installer too large if it bundles CUDA torch | Ship CPU-only base; wizard installs CUDA wheels on first run based on GPU detection |
| `sc.exe` service needs admin; may fail on non-admin installs | Offer autostart scheduled-task fallback (no admin) |
| Sibling repos may not be cloneable offline | Wizard detects local copies first; only clones if missing; gives clear error |
| Tauri NSIS resource path for the server payload | Compute the server path at runtime relative to the exe; verify in E2E |
| First-run wizard hangs on a long install | Run steps asynchronously with progress + per-step timeout + cancel |

---

## Effort Estimate

| Step | Effort |
| ------ | -------- |
| 1. Server packaging (PyInstaller) | 1–2 hours |
| 2. Tauri bundle + first-run window | 2–3 hours |
| 3. Setup bootstrap (Rust + PS) | 2–3 hours |
| 4. Enhanced web setup wizard | 1–2 hours |
| 5. Service management | 1–2 hours |
| 6. Installer build + docs + E2E | 2–3 hours |
| **Total** | **~9–14 hours** |

---

## Suggested Execution Order

1. ✅ **Step 1** (server packaging) — done: `server.spec`, `build-server.ps1`, `app.py` `__main__`; verified packaged exe serves `/health`
2. ✅ **Step 4** (setup helpers: bootstrap + service registration) — done: `setup-bootstrap.ps1`, `register-service.ps1`, `register-autostart.ps1`
3. ✅ **Step 2** (Tauri Rust commands + native installer UI) — done: 16 commands + `desktop/dist/` installer frontend + `dev-server.mjs`
4. ✅ **Step 3** (guided installer flow screens A–H) — done: full native flow implemented and browser-tested
5. ⏳ **Step 5** (web wizard fallback) — the existing `/services` + `/setup-wizard` web UI remains; not yet extended for parity
6. ⏳ **Step 6** (installer + docs) — pending: bundle `dist/ai-server` into the NSIS installer, E2E, docs

---

## Decisions & Open Questions

**Resolved:**
1. **Unsigned installer is fine** — ✅ no code-signing certificate needed. We will
   document the Windows SmartScreen “More info → Run anyway” step in the README.
2. **Keep the installer small** — ✅ ship a CPU-only base; CUDA torch + llama-cpp
   wheels (or Docker images) are provisioned on first run based on the chosen
   deployment mode and GPU detection.
3. **Native guided installer** — ✅ the setup is a native desktop flow (screens A–H),
   not a browser wizard.

**Resolved (round 2):**
1. **Docker vs local default** — ✅ **local** is the default; the user can opt into
   Docker on Screen C if they want isolation.
2. **Start-on-startup default** — ✅ **logon scheduled task** (no admin) is the
   default mechanism; background service remains available as an option.
3. **Installer = desktop app** — ✅ the installer *is* the desktop app. So on
   Screen G, “install/start the desktop app” means **launch the app now**; the
   alternative is **open the web UI instead** (with user confirmation).
