FROM python:3.13-slim

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    EXOMAC_LOCAL_DIR=/app/ExoMACModel/ExoMAC-KKT \
    HF_HOME=/home/user/.cache/huggingface \
    PORT=7860

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy only what the app needs (avoid notebooks/datasets/.venv)
COPY app.py ./app.py
COPY models ./models
COPY ExoMACModel ./ExoMACModel

# Create non-root user matching Spaces runtime and fix permissions
RUN useradd -m -u 1000 -s /bin/bash user \
    && mkdir -p /home/user/.cache/huggingface \
    && chown -R user:user /app /home/user

USER 1000:1000

# API port
EXPOSE 7860

# Start FastAPI with Uvicorn on the provided PORT (Spaces uses 7860)
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
