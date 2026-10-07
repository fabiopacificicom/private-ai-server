<#
  Build the complete Private AI release artifacts in one command.

  This orchestrates BOTH rebuilds so the packaged artifacts stay in sync
  with the latest source before a new release:

    1. Server bundle  -> PyInstaller (dist/ai-server)   [scripts/build-server.ps1]
    2. Desktop/installer -> Tauri NSIS (Private AI_*.exe) [desktop/build.ps1 -Release]

  Output:
    - Server bundle:  dist/ai-server/
    - Installer:      desktop/src-tauri/target/release/bundle/nsis/Private AI_*.exe

  Usage:
    .\scripts\package-release.ps1            # build server + installer
    .\scripts\package-release.ps1 -Clean     # remove dist/build first, then build
    .\scripts\package-release.ps1 -SkipServer # only rebuild the installer (uses existing dist)
    .\scripts\package-release.ps1 -SkipDesktop # only rebuild the server bundle

  Prerequisites: python + PyInstaller, Rust, MSVC Build Tools, Node/npm, WebView2.
#>

[CmdletBinding()]
param(
    [switch]$Clean,
    [switch]$SkipServer,
    [switch]$SkipDesktop
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)  # repo root

function Write-Step($msg) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
}

# ---------------------------------------------------------------------------
# 1) Server bundle (PyInstaller)
# ---------------------------------------------------------------------------
if (-not $SkipServer) {
    Write-Step "Step 1/2: Building server bundle (PyInstaller)"
    $serverArgs = @()
    if ($Clean) { $serverArgs += "-Clean" }
    & (Join-Path $root "scripts\build-server.ps1") @serverArgs
    if ($LASTEXITCODE -ne 0) { Write-Error "Server build failed (exit $LASTEXITCODE)."; exit 1 }

    # Verify the bundle is small enough for NSIS (2 GB limit)
    $bundle = Join-Path $root "dist\ai-server"
    if (Test-Path $bundle) {
        $size = (Get-ChildItem $bundle -Recurse -File | Measure-Object -Property Length -Sum).Sum
        $sizeGB = $size / 1GB
        Write-Host ("Server bundle size: {0:N2} GB" -f $sizeGB) -ForegroundColor Yellow
        if ($sizeGB -gt 1.9) {
            Write-Warning "Server bundle is close to the NSIS 2 GB limit. The installer build may fail."
        }
    } else {
        Write-Error "Server bundle not found after build: $bundle"
        exit 1
    }
} else {
    Write-Step "Step 1/2: SKIPPED server bundle (using existing dist/ai-server)"
}

# ---------------------------------------------------------------------------
# 2) Desktop app + NSIS installer (Tauri)
# ---------------------------------------------------------------------------
if (-not $SkipDesktop) {
    Write-Step "Step 2/2: Building desktop app + NSIS installer (Tauri)"
    Push-Location (Join-Path $root "desktop")
    try {
        & .\build.ps1 -Release
        if ($LASTEXITCODE -ne 0) { Write-Error "Desktop/installer build failed (exit $LASTEXITCODE)."; exit 1 }
    }
    finally {
        Pop-Location
    }
} else {
    Write-Step "Step 2/2: SKIPPED desktop/installer build"
}

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
Write-Step "Build complete"

$installerDir = Join-Path $root "desktop\src-tauri\target\release\bundle\nsis"
if (Test-Path $installerDir) {
    $installer = Get-ChildItem $installerDir -Filter "*.exe" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($installer) {
        Write-Host ("Installer: {0}" -f $installer.FullName) -ForegroundColor Green
        Write-Host ("Size:      {0:N2} MB" -f ($installer.Length / 1MB)) -ForegroundColor Green
        Write-Host ("Built:     {0}" -f $installer.LastWriteTime) -ForegroundColor Green
    }
}
Write-Host ""
Write-Host "Ready to distribute. Run the installer on a clean Windows machine to validate." -ForegroundColor Green
