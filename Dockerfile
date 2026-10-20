FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-diarization.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# Диаризация pyannote: torch 2.6 с CUDA 12.4 (на Linux так ставится с PyPI; без видеокарты работает на cpu), образ ~10 ГБ.
# Образ без неё (меньше, видео режется только по паузам): DIARIZATION=0 docker compose build
ARG DIARIZATION=1
RUN if [ "$DIARIZATION" = "1" ]; then \
        pip install --no-cache-dir torch==2.6.0 torchaudio==2.6.0 \
        && pip install --no-cache-dir -r requirements-diarization.txt; \
    fi
COPY app/ app/
COPY pipeline/ pipeline/

ENV PYTHONUNBUFFERED=1 DATA_DIR=/data
# модель pyannote скачивается один раз и хранится в data/.cache, а не теряется при пересборке контейнера
ENV HF_HOME=/data/.cache/huggingface
# Видео и результаты лежат в /data (монтируется с хоста), ключ приходит из env (.env через compose).
VOLUME /data
EXPOSE 8000
# По умолчанию веб-интерфейс; пакетная обработка папки: docker compose run --rm prepare
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
