FROM python:3.11-slim

# Install system deps + yt-dlp + deno (needed by yt-dlp for YouTube JS challenge solving)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg wget curl unzip \
    && rm -rf /var/lib/apt/lists/* \
    && wget -qO /usr/local/bin/yt-dlp \
       https://github.com/yt-dlp/yt-dlp/releases/download/2026.03.17/yt-dlp \
    && chmod +x /usr/local/bin/yt-dlp \
    && curl -fsSL https://deno.land/install.sh | DENO_INSTALL=/usr/local sh \
    && deno --version

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY review_app.py festivals.py prompts.py wordpress.py db.py config.py downloader.py analyzer.py ./

ENV PORT=8080
EXPOSE 8080

CMD exec gunicorn --bind :$PORT \
    --workers 2 \
    --threads 8 \
    --timeout 3600 \
    review_app:app
