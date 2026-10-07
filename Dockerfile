FROM python:3.11-slim

WORKDIR /app

# System deps: build tools + libsndfile for audio (soundfile)
RUN apt-get update && apt-get install -y \
    build-essential git curl \
    libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# Install base deps. llama-cpp-python from PyPI is a CPU wheel; we replace it
# with the CUDA build below so GGUF models (DeepSeek R1, Muse, ...) use the GPU.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
        "https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.34-cu124/llama_cpp_python-0.3.34-py3-none-manylinux_2_35_x86_64.whl"

COPY . .

EXPOSE 8005

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8005"]
