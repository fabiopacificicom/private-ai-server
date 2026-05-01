#!/bin/bash
# Start AI Server in WSL with vLLM support

set -e

# Navigate to project directory
cd /mnt/d/ai-server-py

# Load environment variables from .env
export $(grep -v '^#' .env | xargs)

# Ensure HF_HOME is set
export HF_HOME=${HF_HOME:-/mnt/e/private-ai-server/models}

# Add local bin to PATH for vLLM and other tools
export PATH="$HOME/.local/bin:$PATH"

echo "=== AI Server WSL Startup ==="
echo "HF_HOME: $HF_HOME"
echo "Python: $(python3 --version)"

# Test vLLM availability
echo "Testing vLLM..."
python3 -c "from vllm import LLM; print('✅ vLLM is available')" || echo "⚠️  vLLM import failed"

# Install Python dependencies if needed
if [ ! -d ".venv-wsl" ]; then
    echo "Creating WSL virtual environment..."
    python3 -m venv .venv-wsl
fi

source .venv-wsl/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt 2>/dev/null || echo "Some packages may already be installed"

# Start the server
echo "Starting server on http://0.0.0.0:8005..."
python3 -m uvicorn app:app --host 0.0.0.0 --port 8005 --reload
