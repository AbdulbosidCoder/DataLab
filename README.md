# DataLab — узбекский речевой датасет

MP4 → wav 16 kHz mono → транскрипция **VoiceLab STT** (текст + спикеры) → сегменты 1.5–15 с → SQLite + файлы.

## Установка

Нужны Python 3.10+ и [ffmpeg](https://ffmpeg.org/download.html) в `PATH`.

```bash
pip install -r requirements.txt
cp .env.example .env      # Windows: copy .env.example .env
# в .env впишите VOICELAB_API_KEY=vlk_...
```

## Запуск

Положите видео в `data/videos/` и запустите:

```bash
python -m pipeline.prepare
```

Флаги: `--videos`, `--out`, `--language uz`, `--min-dur 1.5`, `--max-dur 15`, `--max-gap 0.5`,
`--keep-overlap` (оставить наложения со статусом `overlap`), `--no-speakers`,
`--upload-format wav|flac|mp3` (если большой файл не принимается, попробуйте `mp3`), `--force`.

Ответ VoiceLab кэшируется в `transcripts/<video>.voicelab.json`, поэтому повторный запуск
не отправляет видео заново и не тратит кредиты (кроме `--force`).

## Запуск в Docker

Нужен только Docker (Python и ffmpeg уже внутри образа).

```bash
cp .env.example .env              # Windows: copy .env.example .env, впишите ключ
docker compose build
docker compose run --rm prepare   # видео из data/videos, результат в data/
docker compose run --rm prepare --keep-overlap --upload-format mp3   # флаги передаются как есть
```

Папка `data/` монтируется в контейнер, поэтому аудио, сегменты и расшифровки остаются у вас на диске.

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
