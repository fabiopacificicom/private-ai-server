# Private AI — Desktop App & Installer (Tauri)

A native Windows desktop app built with **Tauri v2** that ships the Private AI
Inference Server as a one-click **NSIS installer** with a guided setup wizard.
Non-technical users can reach a working Chat + Image + Voice UI without ever
opening a terminal.

The app doubles as:
1. **Installer / setup wizard** — an 8-screen guided flow (A–H) that detects
   the machine, provisions dependencies, and configures services.
2. **Desktop launcher** — starts the bundled server and opens the web UI in a
   native window.

## What the installer does

The release build (`build.ps1 -Release`) produces a single `*-setup.exe` that:

- Bundles a **stripped server bootstrap** (`dist/ai-server`) that serves
  `/health` and the setup flow — kept small (~30 MB) to stay under the NSIS
  2 GB file limit.
- On first run, the guided setup wizard provisions the **full runtime**
  (CUDA torch + llama-cpp / transformers) based on detected GPU, using the
  helper scripts under `scripts/`.
- Registers an optional **scheduled task** so services start on log-in.

## The 8 setup screens

| Screen | Purpose |
|--------|---------|
| **A — Detect** | Probes for already-running services (server, Docker, CUDA). |
| **B — Services** | Choose which services to install (already-running ones pre-checked). |
| **C — Run mode** | Local (Python + CUDA) vs Docker container + installation directory. |
| **D — Models** | Pick the model storage directory. |
| **E — Install** | Checks requirements and installs dependencies (checklist). |
| **F — Test** | Tests each service's connection. |
| **G — Launch** | Launch desktop app or open web UI in a browser. |
| **H — Done** | Optional "start on log-in" autostart, then Finish. |

## Screenshots

See [`../assets/screenshots/`](../assets/screenshots/) for captures of all 8 setup
screens plus the Chat / Image / Audio web UI modes.

## Rust commands (18)

Defined in `src-tauri/src/lib.rs` (all non-`pub fn` — required in Tauri 2.11.5
to avoid the E0255 `__cmd__ defined multiple times` error):

- **Detection**: `check_server`, `test_connection`, `probe_services`,
  `check_python`, `check_docker`, `check_cuda`
- **Paths**: `pick_folder`, `default_install_dir`, `default_models_dir`,
  `web_ui_url`
- **Autostart**: `register_autostart`, `unregister_autostart`
- **Provisioning**: `docker_up`, `clone_sibling`, `write_env`, `start_services`
- **Bundled server**: `server_binary_path`, `start_bundled_server`

## Prerequisites (build host)

- **Rust** — install via [rustup](https://rustup.rs) (`rustup-init.exe -y --profile minimal`)
- **MSVC Build Tools** — "Desktop development with C++" workload
  (this repo expects it at `C:\BuildTools`; adjust `build.ps1` if different)
- **WebView2 Runtime** — preinstalled on Windows 11 / recent Windows 10
- **Node.js + npm** — for the Tauri CLI
- **Python** — project `.venv` for PyInstaller server packaging

## Build the installer

The recommended way is a **single command** that rebuilds both the server bundle
and the NSIS installer:

```powershell
# from the repo root — rebuilds server bundle + installer
.\scripts\package-release.ps1
```

Or run the two steps manually:

```powershell
# 1. Package the stripped server bootstrap (from repo root)
.\.venv\Scripts\python.exe -m PyInstaller --noconfirm server.spec

# 2. Build the NSIS installer
cd desktop
.\build.ps1 -Release
# output: src-tauri\target\release\bundle\nsis\Private AI_1.1.0_x64-setup.exe
```

`package-release.ps1` also supports `-Clean`, `-SkipServer`, and `-SkipDesktop`
switches. A cross-platform `scripts/package-release.sh` is provided for
Linux/WSL/CI (note: the NSIS installer itself is Windows-only).

## Development

```powershell
cd desktop
npm install            # first time only (installs Tauri CLI)
node dev-server.mjs    # static dev server on http://localhost:1420/
npm run tauri dev      # run dev (loads installer frontend from dev server)
```

## Structure

```
desktop/
  package.json          # Tauri CLI wrapper
  build.ps1             # Windows build helper (sets up MSVC env)
  dev-server.mjs        # static dev server (port 1420, no-cache)
  dist/                 # installer frontend (index.html, styles.css, installer.js)
  src-tauri/
    tauri.conf.json     # window config, nsis target, bundles dist/ai-server
    Cargo.toml
    src/main.rs
    src/lib.rs          # 18 Tauri commands
    capabilities/default.json
    icons/              # app icons (generated)
```

## Notes

- The server bundle (`server.spec`) **excludes** torch/transformers/etc. to stay
  under the NSIS 2 GB limit. The setup wizard installs the full runtime on first
  run — see `scripts/setup-bootstrap.ps1`.
- The generated `gen_icon.py` regenerates `src-tauri/icons/` for a custom icon.
