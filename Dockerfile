FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ app/
COPY pipeline/ pipeline/

ENV PYTHONUNBUFFERED=1 DATA_DIR=/data
# Видео и результаты лежат в /data (монтируется с хоста), ключ приходит из env (.env через compose).
VOLUME /data
EXPOSE 8000
# По умолчанию веб-интерфейс; пакетная обработка папки: docker compose run --rm prepare
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
