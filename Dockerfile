FROM python:3.11-slim

# Install system dependencies for audio processing (libsndfile and ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs as user with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Install Python dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all application code, model weights, and assets
COPY --chown=user:user . .

# Hugging Face Spaces expects traffic on port 7860
EXPOSE 7860

CMD ["python", "-u", "server.py"]
