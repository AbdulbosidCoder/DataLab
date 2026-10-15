"""Клиент VoiceLab STT (https://api.voicelab.uz).

POST /v1/stt (multipart: audio, language, include_speakers) -> задача,
затем опрос GET /v1/stt/transcriptions/{id}, пока status != completed.
"""
import os
import time
import uuid
from pathlib import Path

import requests

BASE_URL = os.environ.get("VOICELAB_BASE_URL", "https://api.voicelab.uz").rstrip("/")


class VoiceLabError(RuntimeError):
    pass


def load_env(path: str | Path = ".env") -> None:
    """Подхватывает KEY=VALUE из .env в os.environ (без внешних зависимостей)."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_api_key() -> str:
    key = os.environ.get("VOICELAB_API_KEY", "").strip()
    if not key:
        raise VoiceLabError("VOICELAB_API_KEY не задан: добавьте его в .env")
    return key


def _check(resp: requests.Response) -> dict:
    if resp.ok:
        return resp.json()
    try:
        body = resp.json()
        err = body.get("error") or {}
        msg = err.get("message") or body.get("message") or resp.text
        code = err.get("code", "unknown_error")
    except ValueError:
        msg, code = resp.text, "unknown_error"
    raise VoiceLabError(f"VoiceLab HTTP {resp.status_code} [{code}]: {msg}")


def transcribe(audio_path: str | Path, language: str = "uz", include_speakers: bool = True,
               poll_interval: float = 5.0, timeout: float = 3 * 3600) -> dict:
    """Отправляет файл в VoiceLab и ждёт готовую транскрипцию.

    Возвращает объект Transcription: transcript, segments[{start_ms, end_ms, text,
    speaker, words[]}], speakers[{id, display_name}], duration_ms.
    """
    headers = {"Authorization": f"Bearer {get_api_key()}",
               "Idempotency-Key": str(uuid.uuid4())}
    audio_path = Path(audio_path)
    with audio_path.open("rb") as f:
        resp = requests.post(
            f"{BASE_URL}/v1/stt",
            headers=headers,
            files={"audio": (audio_path.name, f)},
            data={"language": language, "include_speakers": str(include_speakers).lower()},
            timeout=600,
        )
    job = _check(resp)

    deadline = time.time() + timeout
    while job.get("status") not in ("completed", "failed"):
        if time.time() > deadline:
            raise VoiceLabError(f"Таймаут ожидания транскрипции {job.get('id')}")
        time.sleep(poll_interval)
        job = _check(requests.get(f"{BASE_URL}/v1/stt/transcriptions/{job['id']}",
                                  headers={"Authorization": headers["Authorization"]},
                                  timeout=60))
    if job["status"] == "failed":
        err = job.get("error") or {}
        raise VoiceLabError(f"Транскрипция {job.get('id')} не удалась: {err.get('message', err)}")
    return job
