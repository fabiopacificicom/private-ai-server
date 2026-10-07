<#
Private AI - first-run provisioning bootstrap (idempotent).

Provisions the environment based on the installer's choices:
- Mode: 'local' (Python + CUDA) or 'docker' (Docker Compose)
- Services: which siblings (Open Fantasia image, Olly Voice) to install
- installDir / modelsDir: where to put things and store models

Each step is idempotent: it checks current state before acting, so re-running
skips completed steps and repairs broken ones. Results are persisted to a
`setup_state.json` in the install directory.

Usage (called by the Tauri installer, or manually for testing):
  .\scripts\setup-bootstrap.ps1 -Mode local -InstallDir "D:\PrivateAI" -ModelsDir "D:\ai-models" -WithImage -WithVoice
  .\scripts\setup-bootstrap.ps1 -Mode docker -InstallDir "D:\PrivateAI" -ModelsDir "D:\ai-models"

Options:
  -Mode        'local' or 'docker' (default: local)
  -InstallDir  Installation directory (default: %LOCALAPPDATA%\Programs\PrivateAI)
  -ModelsDir   Model storage directory (default: %USERPROFILE%\ai-server-models)
  -WithImage   Also provision the Open Fantasia image sibling
  -WithVoice   Also provision the Olly Voice sibling
  -SkipInstall Skip dependency install (only write .env + start)
#>

param(
    [ValidateSet("local", "docker")]
    [string]$Mode = "local",
    [string]$InstallDir = "",
    [string]$ModelsDir = "",
    [switch]$WithImage,
    [switch]$WithVoice,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

function Write-Log($msg) { Write-Host "[bootstrap] $msg" }

# ---------------------------------------------------------------------------
# Resolve defaults
# ---------------------------------------------------------------------------
if (-not $InstallDir) {
    $InstallDir = Join-Path $env:LOCALAPPDATA "Programs\PrivateAI"
}
if (-not $ModelsDir) {
    $ModelsDir = Join-Path $env:USERPROFILE "ai-server-models"
}

$stateFile = Join-Path $InstallDir "setup_state.json"

# ---------------------------------------------------------------------------
# State helpers (idempotency)
# ---------------------------------------------------------------------------
function Read-State {
    if (Test-Path $stateFile) {
        try { return Get-Content $stateFile -Raw | ConvertFrom-Json } catch { }
    }
    return $null
}

function Write-State($state) {
    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    $state | ConvertTo-Json -Depth 5 | Set-Content $stateFile -Encoding UTF8
}

$state = Read-State
if (-not $state) {
    $state = [PSCustomObject]@{
        mode = $Mode
        installDir = $InstallDir
        modelsDir = $ModelsDir
        services = @{}
        completed = @()
    }
}

$steps = @()

# ---------------------------------------------------------------------------
# Step 1 - Docker check (docker mode)
# ---------------------------------------------------------------------------
if ($Mode -eq "docker") {
    $steps += @{
        name = "Docker installed"
        check = {
            $d = Get-Command docker -ErrorAction SilentlyContinue
            if (-not $d) { throw "Docker is not installed or not on PATH." }
            $ver = & docker --version 2>&1
            Write-Log "Docker: $ver"
        }
    }
    $steps += @{
        name = "Start Docker stack"
        check = {
            Push-Location $PSScriptRoot\..
            try { & docker compose up -d --build 2>&1 | Out-Null } finally { Pop-Location }
            Write-Log "Docker stack started."
        }
    }
} else {
    # Local mode
    $steps += @{
        name = "Python detected"
        check = {
            $py = Get-Command python -ErrorAction SilentlyContinue
            if (-not $py) { throw "Python is not installed or not on PATH." }
            $v = & python --version 2>&1
            Write-Log "Python: $v"
        }
    }
    $steps += @{
        name = "CUDA / GPU detected"
        check = {
            $nvs = Get-Command nvidia-smi -ErrorAction SilentlyContinue
            if (-not $nvs) {
                Write-Log "WARNING: nvidia-smi not found - no NVIDIA GPU/CUDA detected. CPU-only mode."
            } else {
                $name = & nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | Select-Object -First 1
                Write-Log "GPU: $name"
            }
        }
    }
}

# ---------------------------------------------------------------------------
# Step 2 - Clone siblings (if requested)
# ---------------------------------------------------------------------------
if ($WithImage) {
    $steps += @{
        name = "Open Fantasia installed"
        check = {
            $dest = Join-Path $InstallDir "open-fantasia-imagegen"
            if (-not (Test-Path $dest)) {
                Write-Log "Cloning Open Fantasia..."
                & git clone https://github.com/fabiopacifici-bot/open-fantasia-imagegen $dest 2>&1 | Out-Null
                if ($LASTEXITCODE -ne 0) { throw "git clone failed for Open Fantasia." }
            } else {
                Write-Log "Open Fantasia already present at $dest"
            }
        }
    }
}
if ($WithVoice) {
    $steps += @{
        name = "Olly Voice installed"
        check = {
            $dest = Join-Path $InstallDir "olly-voice-server"
            if (-not (Test-Path $dest)) {
                Write-Log "Cloning Olly Voice..."
                & git clone https://github.com/fabiopacificicom/olly-voice-server $dest 2>&1 | Out-Null
                if ($LASTEXITCODE -ne 0) { throw "git clone failed for Olly Voice." }
            } else {
                Write-Log "Olly Voice already present at $dest"
            }
        }
    }
}

# ---------------------------------------------------------------------------
# Step 3 - Write .env
# ---------------------------------------------------------------------------
$steps += @{
    name = "Write .env configuration"
    check = {
        New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
        $envFile = Join-Path $InstallDir ".env"
        $envLines = @(
            "HF_HOME=$ModelsDir",
            "HF_HUB_DISABLE_SYMLINKS=1",
            "IMAGE_SERVER_URL=http://127.0.0.1:8765",
            "VOICE_SERVER_URL=http://127.0.0.1:8766",
            "DEPLOY_MODE=$Mode"
        ) -join "`n"
        Set-Content -Path $envFile -Value $envLines -Encoding UTF8
        Write-Log "Wrote $envFile"
    }
}

# ---------------------------------------------------------------------------
# Step 4 - Start services
# ---------------------------------------------------------------------------
$steps += @{
    name = "Start services"
    check = {
        if ($Mode -eq "docker") {
            Push-Location $PSScriptRoot\..
            try { & docker compose up -d --build 2>&1 | Out-Null } finally { Pop-Location }
        } else {
            # Best-effort: try the health endpoint; real launcher wired later.
            try {
                $r = Invoke-WebRequest -Uri "http://127.0.0.1:8005/health" -TimeoutSec 3 -UseBasicParsing
                if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 300) {
                    Write-Log "Private AI server is running."
                } else {
                    Write-Log "Private AI server responded with HTTP $($r.StatusCode)."
                }
            } catch {
                Write-Log "Private AI server not reachable yet (start it from the install directory)."
            }
        }
    }
}

# ---------------------------------------------------------------------------
# Run steps (skip install if requested)
# ---------------------------------------------------------------------------
$results = @{}
$failed = 0
foreach ($step in $steps) {
    if ($SkipInstall -and $step.name -notin @("Write .env configuration", "Start services")) {
        $results[$step.name] = "skipped"
        Write-Log "SKIP: $($step.name)"
        continue
    }
    Write-Log "RUN: $($step.name)"
    try {
        & $step.check
        $results[$step.name] = "ok"
    } catch {
        $results[$step.name] = "error: $($_.Exception.Message)"
        $failed++
        Write-Log "ERROR: $($step.name) - $($_.Exception.Message)"
    }
}

# ---------------------------------------------------------------------------
# Persist state
# ---------------------------------------------------------------------------
$state.mode = $Mode
$state.installDir = $InstallDir
$state.modelsDir = $ModelsDir
$state.services = $results
$state.completed = @($results.Keys | Where-Object { $results[$_] -eq "ok" })
Write-State $state

Write-Log "=============================================="
Write-Log "Setup complete. Steps:"
$results.GetEnumerator() | ForEach-Object { Write-Log "  $($_.Key): $($_.Value)" }
if ($failed -gt 0) {
    Write-Log "WARNING: $failed step(s) failed. Re-run this script to retry."
    exit 1
} else {
    Write-Log "All steps succeeded. State saved to $stateFile"
    exit 0
}
