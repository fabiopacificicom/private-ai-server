FROM python:3.11-slim

WORKDIR /app

# System deps: build tools + libsndfile for audio (soundfile)
RUN apt-get update && apt-get install -y \
    build-essential git curl \
    libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8005

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8005"]
