# Build the inference_core_rb Rust core as a native Windows .pyd wheel.
# Run on the Windows host (the deploy target) — WSL produces a Linux .so, but
# the PyInstaller server bundle needs an MSVC .pyd for CPython.
#
# Usage (from repo root on Windows):
#   powershell -ExecutionPolicy Bypass -File scripts/build_inference_core_win.ps1
#
# Requires: Rust (MSVC toolchain), Python 3.13 with pip on PATH.

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Crate = Join-Path $Root 'inference_core_rb'
Set-Location $Crate

Write-Host "==> inference_core_rb Windows .pyd build" -ForegroundColor Cyan
Write-Host "    crate: $Crate"

# pyo3 0.20 caps at CPython 3.12; 3.13 builds via the stable ABI (forward-compat).
$env:PYO3_USE_ABI3_FORWARD_COMPATIBILITY = '1'

# 1. Rust unit tests
Write-Host "`n==> cargo test --release" -ForegroundColor Cyan
& cargo test --release
if ($LASTEXITCODE -ne 0) { throw "cargo test failed" }

# 2. Ensure maturin
Write-Host "`n==> ensuring maturin" -ForegroundColor Cyan
python -m pip install --upgrade maturin
if ($LASTEXITCODE -ne 0) { throw "maturin install failed" }

# 3. Build the wheel (.pyd)
Write-Host "`n==> maturin build --release" -ForegroundColor Cyan
& maturin build --release --interpreter python
if ($LASTEXITCODE -ne 0) { throw "maturin build failed" }

$wheel = Get-ChildItem (Join-Path $Crate 'target/wheels/*.whl') | Select-Object -First 1
if (-not $wheel) { throw "no wheel produced" }
Write-Host "`n==> wheel: $($wheel.FullName)" -ForegroundColor Green

# 4. Install + import smoke test in the server's Python
Write-Host "`n==> install wheel + import smoke test" -ForegroundColor Cyan
python -m pip install --force-reinstall $wheel.FullName
if ($LASTEXITCODE -ne 0) { throw "wheel install failed" }
python -c "from inference_core_rb import BatchCoordinator, TokenBatchRequest; print('OK: inference_core_rb importable as .pyd')"
if ($LASTEXITCODE -ne 0) { throw "import smoke test failed" }

Write-Host "`n==> DONE. Ship $($wheel.Name) with the server bundle (or run the server with PYTHONPATH pointing at its directory)." -ForegroundColor Green
