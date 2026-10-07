# Builds the Private AI Tauri desktop app on Windows.
# Usage:
#   .\build.ps1          # debug build
#   .\build.ps1 -Release # release build + bundle installer
param(
    [switch]$Release
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$vcvars = "C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

if (-not (Test-Path $vcvars)) {
    Write-Host "MSVC Build Tools not found at $vcvars" -ForegroundColor Red
    Write-Host "Install 'Desktop development with C++' via the Visual Studio Build Tools installer."
    exit 1
}

# Ensure Rust is on PATH for this session
$env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Host "Rust (cargo) not found. Install it: https://rustup.rs" -ForegroundColor Red
    exit 1
}

Write-Host "Building Private AI desktop app..." -ForegroundColor Cyan
Push-Location $root
try {
    if ($Release) {
        # Use the npm-installed Tauri CLI (cargo tauri is not installed).
        cmd /c "call `"$vcvars`" >nul 2>&1 && npm run tauri build -- --bundles nsis"
    } else {
        Push-Location (Join-Path $root "src-tauri")
        try {
            cmd /c "call `"$vcvars`" >nul 2>&1 && cargo build"
        }
        finally {
            Pop-Location
        }
    }
    if ($LASTEXITCODE -ne 0) { throw "Build failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}

Write-Host "Build complete." -ForegroundColor Green
