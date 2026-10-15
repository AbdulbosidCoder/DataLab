# DataLab — узбекский речевой датасет

MP4 → wav 16 kHz mono → транскрипция **VoiceLab STT** (текст + спикеры) → сегменты 1.5–15 с → SQLite + файлы
→ проверка и правка текста каждого сегмента в веб-интерфейсе.

## Установка

Нужны Python 3.10+ и [ffmpeg](https://ffmpeg.org/download.html) в `PATH`.

```bash
pip install -r requirements.txt
cp .env.example .env      # Windows: copy .env.example .env
# в .env впишите VOICELAB_API_KEY=vlk_...
```

## Запуск без Docker

```bash
uvicorn app.server:app --port 8000   # веб-интерфейс
python -m pipeline.prepare           # или пакетно: видео из data/videos/
```

Флаги: `--videos`, `--out`, `--language uz`, `--min-dur 1.5`, `--max-dur 15`, `--max-gap 0.5`,
`--keep-overlap` (оставить наложения со статусом `overlap`), `--no-speakers`,
`--upload-format wav|flac|mp3` (если большой файл не принимается, попробуйте `mp3`), `--force`.

Ответ VoiceLab кэшируется в `transcripts/<video>.voicelab.json`, поэтому повторный запуск
не отправляет видео заново и не тратит кредиты (кроме `--force`).

## Веб-интерфейс (Docker)

Нужен только Docker (Python и ffmpeg уже внутри образа).

```bash
cp .env.example .env              # Windows: copy .env.example .env, впишите ключ
docker compose up -d --build      # открыть http://localhost:8000
```

1. Перетащите видео или аудио в блок «Загрузить». Файл встанет в очередь, VoiceLab его расшифрует
   (статус виден в блоке «Обработка»), сегменты появятся в списке.
2. Нажмите «Следующий»: играет аудио сегмента, рядом соседние реплики, ниже текст для правки.
3. Исправьте текст и нажмите «Одобрить» (`Ctrl+Enter`), «Брак» (`Ctrl+Shift+X`) или «Наложение» (`Ctrl+Shift+O`).
   `Ctrl+Space` пауза, `Ctrl+←` назад 2 с, `Ctrl+↓` пропустить. Кнопки `oʻ gʻ ʼ` и «Исправить апострофы»
   помогают с узбекской латиницей; интерфейс предупреждает о цифрах и кириллице.
4. «Скачать approved.jsonl» отдаёт проверенные сегменты.

Пакетная обработка папки `data/videos` без браузера:

```bash
docker compose run --rm prepare   # флаги передаются как есть, например --upload-format mp3
```

## Что получается в `data/`

| Путь | Содержимое |
|---|---|
| `audio/<video>.wav` | полное аудио, 16 kHz mono 16-bit PCM |
| `segments/<video>/<video>_<start_ms>_<end_ms>.wav` | реплики одного спикера 1.5–15 с |
| `transcripts/<video>.txt` / `.srt` | полная расшифровка со спикерами и таймкодами |
| `transcripts/<video>.voicelab.json` | сырой ответ VoiceLab |
| `manifest.jsonl` | все сегменты: `audio_filepath`, `duration`, `text`, `speaker`, `video`, `status` |
| `speech_lab.db` | SQLite, таблица `segments` (схема speech-lab) |

`speaker` = `<video>_SPEAKER_xx`, потому что VoiceLab нумерует спикеров заново в каждом видео.
