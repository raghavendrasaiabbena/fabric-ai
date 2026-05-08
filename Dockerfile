FROM python:3.11-slim

# OpenCV needs these system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (Docker layer cache — only rebuilds when requirements change)
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy source
COPY . .

# Runtime directories (ephemeral — files exist during session, cleared on redeploy)
RUN mkdir -p uploads enhanced composed models

EXPOSE 8000

# Run from backend/ so local imports (enhancer, composer, etc.) resolve correctly
WORKDIR /app/backend
CMD ["python", "main.py"]
