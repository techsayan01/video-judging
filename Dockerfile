FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg wget curl \
    && rm -rf /var/lib/apt/lists/* \
    && wget -qO /usr/local/bin/yt-dlp \
       https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp \
    && chmod +x /usr/local/bin/yt-dlp

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY review_app.py festivals.py prompts.py wordpress.py db.py config.py downloader.py analyzer.py ./

ENV PORT=8080
EXPOSE 8080

CMD exec gunicorn --bind :$PORT \
    --workers 2 \
    --threads 8 \
    --timeout 600 \
    review_app:app
