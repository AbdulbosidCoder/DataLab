FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# Медленный или нестабильный интернет: долгий таймаут и повторы, скачанные пакеты остаются в кэше BuildKit
ENV PIP_DEFAULT_TIMEOUT=300 PIP_RETRIES=10
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
COPY app/ app/
COPY pipeline/ pipeline/

ENV PYTHONUNBUFFERED=1 DATA_DIR=/data
# Видео и результаты лежат в /data (монтируется с хоста), ключ приходит из env (.env через compose).
VOLUME /data
EXPOSE 8000
# По умолчанию веб-интерфейс; пакетная обработка папки: docker compose run --rm prepare
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
