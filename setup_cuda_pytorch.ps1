<#
PowerShell helper to install a CUDA-enabled PyTorch into your environment.

Behavior:
- Detects `nvidia-smi` and reports CUDA/driver info
- Prefers `conda` if available: creates/uses env `aihub` and installs `pytorch pytorch-cuda=12.6`
- Otherwise uses an existing `.venv` (or creates it) and installs pip wheels for CUDA 12.6
- Verifies installation by printing `torch.cuda.is_available()` and device names

Run this from the repository root. It will prompt before making changes.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Log($msg) { Write-Host "[setup] $msg" }

# Check nvidia-smi
Write-Log "Checking GPU (nvidia-smi)..."
try {
    $nvs = & nvidia-smi 2>&1
    Write-Log "nvidia-smi available"
    Write-Host $nvs
} catch {
    Write-Log "nvidia-smi not found or failed: $_"
}

# Check whether conda is available
$conda = Get-Command conda -ErrorAction SilentlyContinue
if ($conda) { Write-Log "Conda detected at: $($conda.Path)" } else { Write-Log "Conda not found" }

# Locate Python
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Log "Python executable not found in PATH. Please run this script from an environment where python is available." 
    exit 1
}
Write-Log "Python: $($pythonCmd.Path)"

# Report current python + torch diag
Write-Log "Current python diagnostics:" 
$tmp = [System.IO.Path]::GetTempFileName() + ".py"
$py = @'
import sys
import platform
print('python', sys.executable)
try:
    import torch
    print('torch', getattr(torch,'__version__',None))
    print('cuda_available', torch.cuda.is_available())
    try:
        print('cuda_count', torch.cuda.device_count())
        if torch.cuda.is_available():
            print('device0', torch.cuda.get_device_name(0))
    except Exception as e:
        print('cuda_check_error', e)
except Exception as e:
    print('torch_import_error', e)
'@
Set-Content -Path $tmp -Value $py -Encoding UTF8
& python $tmp
Remove-Item $tmp

# Ask for confirmation to proceed
$proceed = Read-Host "Proceed to install CUDA-enabled PyTorch (CUDA 12.6) into this environment? [y/N]"
if ($proceed -ne 'y' -and $proceed -ne 'Y') { Write-Log 'Aborting per user request.'; exit 0 }

# Preferred path: conda
if ($conda) {
    Write-Log "Using conda path: creating/activating 'aihub' env and installing pytorch+pytorch-cuda=12.6"

    Write-Log "Creating conda env 'aihub' (if not exists)"
    & conda env list | Out-Null
    try {
        & conda create -n aihub python=3.11 -y
    } catch {
        Write-Log "Conda create may have failed or env exists; continuing"
    }

    Write-Log "Activating conda env 'aihub' and installing pytorch..."
    # Activation in a non-interactive script: call conda run
    & conda run -n aihub --no-capture-output bash -lc "conda activate aihub; conda install -n aihub pytorch pytorch-cuda=12.6 -c pytorch -c nvidia -y" 2>&1 | Write-Host

    Write-Log "Installing project requirements into conda env (if requirements.txt exists)"
    if (Test-Path requirements.txt) {
        & conda run -n aihub --no-capture-output pip install -r requirements.txt 2>&1 | Write-Host
    }

    Write-Log "Verifying torch in conda env"
    $tmp = [System.IO.Path]::GetTempFileName() + ".py"
    $py = @'
import torch
print('torch',torch.__version__)
print('cuda_available',torch.cuda.is_available())
try:
    print('cuda_count',torch.cuda.device_count())
    if torch.cuda.is_available():
        print('device0', torch.cuda.get_device_name(0))
except Exception as e:
    print('cuda_check_error',e)
'@
    Set-Content -Path $tmp -Value $py -Encoding UTF8
    & conda run -n aihub --no-capture-output python $tmp 2>&1 | Write-Host
    Remove-Item $tmp
    exit 0
}

# Fallback: use (or create) .venv in repo root
$venvPath = Join-Path (Get-Location) '.venv'
if (-not (Test-Path $venvPath)) {
    Write-Log "No .venv found, creating one at $venvPath"
    & python -m venv $venvPath
}

# Activate venv for script commands
$activate = Join-Path $venvPath 'Scripts\Activate.ps1'
if (-not (Test-Path $activate)) { Write-Log "Cannot find activation script at $activate"; exit 1 }

Write-Log "Activating .venv"
& powershell -ExecutionPolicy Bypass -NoProfile -Command "& '$activate'; pip --version"

Write-Log "Uninstalling CPU-only torch if present"
& powershell -ExecutionPolicy Bypass -NoProfile -Command "& '$activate'; pip uninstall -y torch torchvision torchaudio" 2>&1 | Write-Host

Write-Log "Clearing pip cache"
& powershell -ExecutionPolicy Bypass -NoProfile -Command "& '$activate'; pip cache purge" 2>&1 | Write-Host

Write-Log "Upgrading pip"
& powershell -ExecutionPolicy Bypass -NoProfile -Command "& '$activate'; python -m pip install --upgrade pip" 2>&1 | Write-Host

Write-Log "Installing CUDA 12.6 PyTorch wheels via pip index (may take some time)"
& powershell -ExecutionPolicy Bypass -NoProfile -Command "& '$activate'; pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126" 2>&1 | Write-Host

Write-Log "If you use bitsandbytes or other CUDA-specific packages, reinstall them now"
& powershell -ExecutionPolicy Bypass -NoProfile -Command "& '$activate'; pip install --upgrade bitsandbytes --no-deps" 2>&1 | Write-Host

if (Test-Path requirements.txt) {
    Write-Log "Installing requirements.txt into .venv"
    & powershell -ExecutionPolicy Bypass -NoProfile -Command "& '$activate'; pip install -r requirements.txt" 2>&1 | Write-Host
}

Write-Log "Verification in .venv"
$tmp = [System.IO.Path]::GetTempFileName() + ".py"
$py = @'
import torch
print('torch', getattr(torch,'__version__',None))
print('cuda_available', torch.cuda.is_available())
try:
    print('cuda_count', torch.cuda.device_count())
    if torch.cuda.is_available():
        print('device0', torch.cuda.get_device_name(0))
except Exception as e:
    print('cuda_check_error', e)
'@
Set-Content -Path $tmp -Value $py -Encoding UTF8
& powershell -ExecutionPolicy Bypass -NoProfile -Command "& '$activate'; python $tmp" 2>&1 | Write-Host
Remove-Item $tmp

Write-Log "Setup script finished"
