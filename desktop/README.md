# Private AI — Desktop App (Tauri)

A native Windows desktop wrapper around the Private AI Inference Server web UI.
It opens a Tauri window that loads the server UI at `http://127.0.0.1:8005/`.

## Prerequisites (Windows host)

- **Rust** — install via [rustup](https://rustup.rs) (`rustup-init.exe -y --profile minimal`)
- **MSVC Build Tools** — "Desktop development with C++" workload
  (this repo expects it at `C:\BuildTools`; adjust `build.ps1` if different)
- **WebView2 Runtime** — preinstalled on Windows 11 / recent Windows 10
- **Node.js + npm** — for the Tauri CLI

## Run the server first

The desktop app is a wrapper — it needs the FastAPI server running on port 8005:

```powershell
# from the repo root
python -m uvicorn app:app --host 0.0.0.0 --port 8005
```

## Build & run

```powershell
cd desktop
npm install                 # first time only (installs Tauri CLI)
.\build.ps1                 # debug build
.\src-tauri\target\debug\private-ai-desktop.exe   # run it
```

### Release build (with installer)

```powershell
cd desktop
.\build.ps1 -Release
# installer/portable exe in: src-tauri\target\release\bundle\
```

## Development

- `npm run tauri dev` — run in dev (still needs the server running on :8005)

## Structure

```
desktop/
  package.json          # Tauri CLI wrapper
  build.ps1             # Windows build helper (sets up MSVC env)
  src-tauri/
    tauri.conf.json     # window config — loads http://127.0.0.1:8005/
    Cargo.toml
    src/main.rs
    src/lib.rs
    capabilities/default.json
    icons/              # app icons (generated)
```

## Notes

- The app points at `http://127.0.0.1:8005/`. If you run the server on a different
  host/port, update `url` in `src-tauri/tauri.conf.json`.
- The generated `gen_icon.py` regenerates `src-tauri/icons/` if you want a custom icon.
