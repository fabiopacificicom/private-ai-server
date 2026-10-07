#!/usr/bin/env bash
#
# Build the complete Private AI release artifacts in one command (Linux/WSL/macOS).
#
# This orchestrates BOTH rebuilds so the packaged artifacts stay in sync with the
# latest source before a new release. On Windows, prefer scripts/package-release.ps1.
#
#   1. Server bundle  -> PyInstaller (dist/ai-server)   [scripts/build-server.ps1]
#   2. Desktop/installer -> Tauri bundle                [desktop/build.ps1 -Release]
#
# NOTE: The Tauri NSIS installer is Windows-only. On a non-Windows host this script
# builds the server bundle and the desktop app; for the actual NSIS installer run it
# on Windows (or in a Windows CI runner).
#
# Usage:
#   ./scripts/package-release.sh             # build server + desktop
#   ./scripts/package-release.sh --clean     # remove dist/build first, then build
#   ./scripts/package-release.sh --skip-server
#   ./scripts/package-release.sh --skip-desktop
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLEAN=0
SKIP_SERVER=0
SKIP_DESKTOP=0

for arg in "$@"; do
  case "$arg" in
    --clean) CLEAN=1 ;;
    --skip-server) SKIP_SERVER=1 ;;
    --skip-desktop) SKIP_DESKTOP=1 ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

step() { printf '\n== %s ==\n' "$1"; }

# ---------------------------------------------------------------------------
# 1) Server bundle (PyInstaller)
# ---------------------------------------------------------------------------
if [ "$SKIP_SERVER" -eq 0 ]; then
  step "Step 1/2: Building server bundle (PyInstaller)"

  if ! command -v python >/dev/null 2>&1; then
    echo "ERROR: python not found on PATH"; exit 1
  fi
  if ! python -c "import PyInstaller" >/dev/null 2>&1; then
    echo "Installing PyInstaller..."
    python -m pip install --quiet pyinstaller
  fi

  if [ "$CLEAN" -eq 1 ]; then
    rm -rf "$ROOT/dist" "$ROOT/build"
  fi

  (cd "$ROOT" && python -m PyInstaller --noconfirm server.spec)

  if [ -d "$ROOT/dist/ai-server" ]; then
    size=$(du -sh "$ROOT/dist/ai-server" | cut -f1)
    echo "Server bundle size: $size"
  else
    echo "ERROR: server bundle not found at dist/ai-server"; exit 1
  fi
else
  step "Step 1/2: SKIPPED server bundle (using existing dist/ai-server)"
fi

# ---------------------------------------------------------------------------
# 2) Desktop app + installer (Tauri)
# ---------------------------------------------------------------------------
if [ "$SKIP_DESKTOP" -eq 0 ]; then
  step "Step 2/2: Building desktop app (Tauri)"
  (cd "$ROOT/desktop" && npm run tauri build)
else
  step "Step 2/2: SKIPPED desktop build"
fi

step "Build complete"
echo "Server bundle: $ROOT/dist/ai-server"
echo "Desktop bundle: $ROOT/desktop/src-tauri/target/release/bundle/"
