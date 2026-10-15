FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ app/
COPY pipeline/ pipeline/

ENV PYTHONUNBUFFERED=1
# Видео и результаты лежат в /data (монтируется с хоста), ключ приходит из env (.env через compose).
VOLUME /data
ENTRYPOINT ["python", "-m", "pipeline.prepare", "--videos", "/data/videos", "--out", "/data"]
