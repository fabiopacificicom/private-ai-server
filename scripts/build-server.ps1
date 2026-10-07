<#
Build the Private AI server into a self-contained PyInstaller one-folder bundle.

Output: dist/ai-server/ (the server binary + src + static + deps).
The installer / desktop app bundles this folder.

Usage:
  .\scripts\build-server.ps1          # build
  .\scripts\build-server.ps1 -Clean   # remove dist/build first

Prerequisites: python with pyinstaller installed in the active environment.
#>

param(
    [switch]$Clean
)

# "Continue" so native commands writing to stderr (e.g. a failed import probe)
# don't abort the script; we check $LASTEXITCODE explicitly where it matters.
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot

function Write-Log($msg) { Write-Host "[build-server] $msg" }

# Ensure pyinstaller is available
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
    Write-Error "Python not found on PATH."
    exit 1
}
Write-Log "Python: $($py.Path)"

$pyi = & python -c "import PyInstaller; print(PyInstaller.__version__)" 2>&1 | Out-String
$pyi = $pyi.Trim()
if ($pyi -notmatch '^\d+\.') {
    Write-Log "PyInstaller not installed. Installing..."
    & python -m pip install --quiet pyinstaller
    if ($LASTEXITCODE -ne 0) { Write-Error "Failed to install PyInstaller."; exit 1 }
    $pyi = & python -c "import PyInstaller; print(PyInstaller.__version__)" 2>&1 | Out-String
    $pyi = $pyi.Trim()
}
Write-Log "PyInstaller: $pyi"

if ($Clean) {
    Write-Log "Cleaning dist/ and build/ ..."
    Remove-Item -Recurse -Force (Join-Path $root "dist") -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force (Join-Path $root "build") -ErrorAction SilentlyContinue
}

Push-Location $root
try {
    Write-Log "Running PyInstaller (server.spec)..."
    & python -m PyInstaller --noconfirm server.spec
    if ($LASTEXITCODE -ne 0) { Write-Error "PyInstaller failed (exit $LASTEXITCODE)."; exit 1 }
}
finally {
    Pop-Location
}

$out = Join-Path $root "dist\ai-server"
if (Test-Path $out) {
    Write-Log "Build complete: $out"
    Write-Log "Smoke test: start with  $out\ai-server.exe  then hit http://127.0.0.1:8005/health"
} else {
    Write-Error "Expected output not found: $out"
    exit 1
}
