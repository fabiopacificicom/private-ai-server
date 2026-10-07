<#
setup-llamacpp-cuda.ps1
-----------------------
Install llama-cpp-python with CUDA support so GGUF models (DeepSeek R1, Muse, etc.)
run on the NVIDIA GPU instead of silently falling back to CPU.

WHY THIS EXISTS
---------------
llama-cpp-python's official releases ship only a CPU wheel (`py3-none-win_amd64.whl`)
on Windows; the prebuilt CUDA wheels are Linux-only. So on Windows, CUDA support is
gained by building llama.cpp from source at install time. On Linux (and in the
project Dockerfile) the same env-var-driven source build applies.

This is the *install-time* step, not part of the runtime. It should be run once when
the environment is set up (or invoked from a bootstrap/setup step) — never at server
startup.

PREREQUISITES
-------------
- NVIDIA GPU with a recent driver (verify: `nvidia-smi`)
- CUDA Toolkit installed (verify: `nvcc --version`). Your torch is cu126 -> CUDA 12.x.
- Windows: Visual Studio with "Desktop development with C++" (MSVC) + CMake.
- Linux/Docker: build-essential, git, cmake (already present in the project Dockerfile).

USAGE
-----
From the repository root (the folder containing requirements.txt):

    powershell -ExecutionPolicy Bypass -File scripts/setup-llamacpp-cuda.ps1

Optional args (all default to sensible values for a 4090 / CUDA 12.x):
    -PythonExec  path to the python/pip to use          (default: .venv, else `python`)
    -CudaArch    comma list of CUDA architectures       (default: 89 = Ada / RTX 40-series)
    -Force       reinstall even if a CUDA llama_cpp is already present
#>
[CmdletBinding()]
param(
    [string]$PythonExec = "",
    [string]$CudaArch  = "89",
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Step($msg) { Write-Host "[setup-llamacpp-cuda] $msg" -ForegroundColor Cyan }
function Write-OK($msg)   { Write-Host "[setup-llamacpp-cuda] $msg" -ForegroundColor Green }

# ---------------------------------------------------------------- 0. locate python
if (-not $PythonExec) {
    if (Test-Path ".venv\Scripts\python.exe")      { $PythonExec = ".venv\Scripts\python.exe" }
    elseif (Test-Path ".venv\bin\python")          { $PythonExec = ".venv\bin\python" }
    else {
        $pyCmd = Get-Command python -ErrorAction SilentlyContinue
        if (-not $pyCmd) { Write-Host "Python not found. Pass -PythonExec." -ForegroundColor Red; exit 1 }
        $PythonExec = $pyCmd.Source
    }
}
Write-Step "Using Python: $PythonExec"
# resolve pip as a module (works for both .venv and system python)
$pipArgs = @("-m", "pip")

# ---------------------------------------------------------------- 1. detect CUDA
Write-Step "Checking CUDA availability..."
$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
if ($nvcc) {
    Write-OK "nvcc found: $($nvcc.Source)"
} else {
    Write-Host "WARNING: nvcc not found on PATH. Make sure the CUDA Toolkit is installed (nvcc --version should work)." -ForegroundColor Yellow
}

$nvsmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvsmi) {
    Write-OK "nvidia-smi found: $($nvsmi.Source)"
} else {
    Write-Host "WARNING: nvidia-smi not found. No NVIDIA GPU detected?" -ForegroundColor Yellow
}

# ---------------------------------------------------------------- 2. already CUDA?
Write-Step "Checking existing llama-cpp-python build..."
$already = & $PythonExec $pipArgs show llama-cpp-python 2>$null | Select-String "Version:"
if ($already -and -not $Force) {
    # Check whether ggml-cuda is present in the installed package (best-effort)
    $libDir = $null
    try {
        $site = & $PythonExec -c "import llama_cpp, os; print(os.path.join(os.path.dirname(llama_cpp.__file__),'lib'))" 2>$null
        if ($site -and (Test-Path $site)) { $libDir = $site }
    } catch {}

    $hasCuda = $false
    if ($libDir) { $hasCuda = (Get-ChildItem $libDir -Filter "*cuda*" -ErrorAction SilentlyContinue | Measure-Object).Count -gt 0 }
    if ($hasCuda) {
        Write-OK "llama-cpp-python already built with CUDA ($already). Nothing to do. Pass -Force to rebuild."
        exit 0
    }
    Write-Host "llama-cpp-python is installed but WITHOUT a CUDA backend. Rebuilding with CUDA..." -ForegroundColor Yellow
}

# ---------------------------------------------------------------- 3. build with CUDA
# GGML_CUDA + CMAKE_CUDA_ARCHITECTURES must be visible to the pip build (sdist from source).
Write-Step "Building llama-cpp-python with CUDA (arch=$CudaArch). This compiles from source and may take a few minutes..."
$env:GGML_CUDA = "on"
$env:CMAKE_ARGS = "-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=$CudaArch"

& $PythonExec @pipArgs install --force-reinstall --no-cache-dir "llama-cpp-python>=0.3.34"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[setup-llamacpp-cuda] pip install FAILED (exit $LASTEXITCODE). Check CUDA toolkit + compiler (MSVC on Windows)." -ForegroundColor Red
    exit $LASTEXITCODE
}

# ---------------------------------------------------------------- 4. verify
Write-Step "Verifying CUDA backend..."
$libDir = & $PythonExec -c "import llama_cpp, os; print(os.path.join(os.path.dirname(llama_cpp.__file__),'lib'))" 2>$null
if ($libDir -and (Test-Path $libDir)) {
    $cudaDll = Get-ChildItem $libDir -Filter "*cuda*" -ErrorAction SilentlyContinue
    if ($cudaDll) {
        Write-OK "CUDA backend installed: $($cudaDll.Name -join ', ')"
        Write-OK "GGUF models (DeepSeek R1, Muse, qwen3:*, etc.) will now offload to the NVIDIA GPU."
    } else {
        Write-Host "Installed, but no *cuda* lib found in $libDir. The build may have fallen back to CPU. Re-check CMAKE_ARGS." -ForegroundColor Yellow
    }
} else {
    Write-Host "Could not locate llama_cpp lib dir to verify CUDA. Check the install manually: ggml-cuda.dll should exist under site-packages/llama_cpp/lib/." -ForegroundColor Yellow
}

Write-OK "Done. Restart the ai-server for the GPU-enabled llama-cpp to take effect."
