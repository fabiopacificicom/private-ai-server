#!/usr/bin/env bash
#===============================================================================
# setup-llamacpp-cuda.sh
#----------------------
# Install llama-cpp-python with CUDA support so GGUF models (DeepSeek R1, Muse,
# etc.) run on the NVIDIA GPU instead of silently falling back to CPU.
#
# Linux/Docker twin of scripts/setup-llamacpp-cuda.ps1. Same behavior, bash.
#
# WHY THIS EXISTS
# ---------------
# llama-cpp-python's official releases ship only a CPU wheel for some platforms;
# the prebuilt CUDA wheels are not always available. CUDA support is gained by
# building llama.cpp from source at install time (env-var driven via CMAKE_ARGS).
#
# Install-time step only — run once during environment setup, never at runtime.
#
# PREREQUISITES
# -------------
# - NVIDIA GPU (+ driver)          -> verify: nvidia-smi
# - CUDA Toolkit                   -> verify: nvcc --version  (torch cu126 -> CUDA 12.x)
# - build-essential, git, cmake    -> already present in the project Dockerfile
#
# USAGE
# -----
# From the repository root:
#     bash scripts/setup-llamacpp-cuda.sh
# Args:  -p, --python PATH   python/pip to use     (default: .venv, else `python3`)
#        -a, --arch ARCHS    CUDA arch list        (default: 89 = Ada / RTX 40-series)
#        -f, --force         reinstall even if CUDA build already present
#===============================================================================

set -euo pipefail

PYTHON_EXEC=""
CUDA_ARCH="89"
FORCE=0

usage() {
  echo "Usage: $0 [-p PYTHON] [-a ARCHS] [-f]" >&2
  exit 1
}

# --- parse args --------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--python) PYTHON_EXEC="$2"; shift 2 ;;
    -a|--arch)   CUDA_ARCH="$2";   shift 2 ;;
    -f|--force)  FORCE=1;          shift   ;;
    -h|--help)   usage ;;
    *) echo "Unknown arg: $1" >&2; usage ;;
  esac
done

step() { printf '\033[36m[setup-llamacpp-cuda]\033[0m %s\n' "$*"; }
ok()   { printf '\033[32m[setup-llamacpp-cuda]\033[0m %s\n' "$*"; }

# --- 0. locate python --------------------------------------------------------
if [[ -z "$PYTHON_EXEC" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_EXEC=".venv/bin/python"
  else
    command -v python3 >/dev/null 2>&1 && PYTHON_EXEC="python3" || PYTHON_EXEC="python"
  fi
fi
step "Using Python: $PYTHON_EXEC"

# --- 1. detect CUDA ----------------------------------------------------------
step "Checking CUDA availability..."
if command -v nvcc >/dev/null 2>&1; then
  ok "nvcc found: $(command -v nvcc)"
else
  echo "WARNING: nvcc not found on PATH. Ensure the CUDA Toolkit is installed." >&2
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  ok "nvidia-smi found"
else
  echo "WARNING: nvidia-smi not found. No NVIDIA GPU detected?" >&2
fi

# --- 2. already CUDA? --------------------------------------------------------
step "Checking existing llama-cpp-python build..."
if [[ "$FORCE" -eq 0 ]] && "$PYTHON_EXEC" -c "import llama_cpp" >/dev/null 2>&1; then
  LIBDIR="$("$PYTHON_EXEC" -c "import llama_cpp,os;print(os.path.join(os.path.dirname(llama_cpp.__file__),'lib'))" 2>/dev/null || true)"
  if [[ -n "$LIBDIR" && -d "$LIBDIR" ]] && ls "$LIBDIR"/*cuda* >/dev/null 2>&1; then
    VER="$("$PYTHON_EXEC" -c "import importlib.metadata as m;print(m.version('llama-cpp-python'))" 2>/dev/null || echo '?')"
    ok "llama-cpp-python $VER already built with CUDA. Nothing to do. Use -f to rebuild."
    exit 0
  fi
  echo "llama-cpp-python installed but WITHOUT CUDA backend. Rebuilding with CUDA..." >&2
fi

# --- 3. build with CUDA ------------------------------------------------------
step "Building llama-cpp-python with CUDA (arch=$CUDA_ARCH). Compiling from source; may take a few minutes..."
export GGML_CUDA="on"
export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH"

"$PYTHON_EXEC" -m pip install --force-reinstall --no-cache-dir "llama-cpp-python>=0.3.34"

# --- 4. verify ---------------------------------------------------------------
step "Verifying CUDA backend..."
LIBDIR="$("$PYTHON_EXEC" -c "import llama_cpp,os;print(os.path.join(os.path.dirname(llama_cpp.__file__),'lib'))" 2>/dev/null || true)"
if [[ -n "$LIBDIR" && -d "$LIBDIR" ]]; then
  CUDA_LIBS="$(ls "$LIBDIR"/*cuda* 2>/dev/null || true)"
  if [[ -n "$CUDA_LIBS" ]]; then
    ok "CUDA backend installed:"
    echo "$CUDA_LIBS" | sed 's/^/    /'
    ok "GGUF models (DeepSeek R1, Muse, qwen3:*, etc.) will now offload to the NVIDIA GPU."
  else
    echo "WARNING: installed, but no *cuda* lib found in $LIBDIR. Build may have fallen back to CPU. Re-check CMAKE_ARGS." >&2
  fi
else
  echo "WARNING: could not locate llama_cpp lib dir to verify CUDA. Check for ggml-cuda.so under site-packages/llama_cpp/lib/." >&2
fi

ok "Done. Restart the ai-server for the GPU-enabled llama-cpp to take effect."
