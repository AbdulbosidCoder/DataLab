"""Клиент VoiceLab STT по документации https://docs.voicelab.uz/api/stt.

GET  /v1/stt/preflight?duration_ms=...      хватит ли кредитов (ничего не резервирует)
POST /v1/stt  (multipart: audio, language, include_speakers; заголовок Idempotency-Key) -> 202 {id, status}
GET  /v1/stt/transcriptions/{id}            опрос до status = completed | failed

Ограничения API: до 500 MiB на файл, от 0.5 с; форматы MP3, WAV, M4A/AAC, OGG/Opus, WebM/Opus, FLAC.
Повторы при 429/500/503, обрыве соединения и недочитанном ответе (IncompleteRead) идут с тем же Idempotency-Key, поэтому кредиты не списываются дважды.
"""
import json
import os
import random
import time
import uuid
from pathlib import Path

import requests

BASE_URL = os.environ.get("VOICELAB_BASE_URL", "https://api.voicelab.uz").rstrip("/")
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
RETRY_STATUSES = {429, 500, 502, 503, 504}


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


def _auth() -> dict:
    return {"Authorization": f"Bearer {get_api_key()}"}


def _error(resp: requests.Response) -> VoiceLabError:
    try:
        body = resp.json()
        err = body.get("error") or {}
        msg = err.get("message") or body.get("message") or resp.text
        code = err.get("code", "unknown_error")
    except ValueError:
        msg, code = resp.text, "unknown_error"
    return VoiceLabError(f"VoiceLab HTTP {resp.status_code} [{code}]: {msg}")


def _wait(resp: requests.Response | None, attempt: int) -> None:
    """Retry-After, если сервер его прислал, иначе экспоненциальная пауза с джиттером."""
    ra = resp.headers.get("Retry-After") if resp is not None else None
    try:
        delay = float(ra) if ra else 2 ** attempt + random.random()
    except ValueError:
        delay = 2 ** attempt + random.random()
    time.sleep(min(delay, 60))


def _request(method: str, path: str, attempts: int = 5, **kw) -> requests.Response:
    """Запрос с повторами на 429/5xx и сетевых сбоях. Для POST вызывающий передаёт тот же Idempotency-Key."""
    last = None
    for attempt in range(attempts):
        files = kw.get("files")
        if files:  # файл перечитывается с начала при каждой попытке
            for _, (_, fh) in files.items():
                fh.seek(0)
        try:
            resp = requests.request(method, f"{BASE_URL}{path}", **kw)
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ContentDecodingError) as e:  # обрыв посреди ответа: IncompleteRead
            last = e
            _wait(None, attempt)
            continue
        if resp.status_code in RETRY_STATUSES and attempt < attempts - 1:
            _wait(resp, attempt)
            continue
        if not resp.ok:
            raise _error(resp)
        return resp
    raise VoiceLabError(f"VoiceLab недоступен после {attempts} попыток: {last}")


def preflight(duration_ms: int, idempotency_key: str | None = None) -> dict:
    """Проверяет, хватит ли кредитов на файл такой длительности."""
    params = {"duration_ms": max(500, int(duration_ms))}
    if idempotency_key:
        params["idempotency_key"] = idempotency_key
    try:
        return _request("GET", "/v1/stt/preflight", headers=_auth(), params=params, timeout=30).json()
    except VoiceLabError as e:
        if "insufficient_credits" in str(e):
            raise VoiceLabError("Недостаточно кредитов VoiceLab для этого файла") from e
        raise


def submit(audio_path: str | Path, language: str, include_speakers: bool, idempotency_key: str) -> dict:
    audio_path = Path(audio_path)
    if audio_path.stat().st_size > MAX_UPLOAD_BYTES:
        raise VoiceLabError(f"{audio_path.name} больше 500 MiB: используйте --upload-format mp3 или flac")
    with audio_path.open("rb") as f:
        resp = _request("POST", "/v1/stt",
                        headers={**_auth(), "Idempotency-Key": idempotency_key},
                        files={"audio": (audio_path.name, f)},
                        data={"language": language, "include_speakers": str(include_speakers).lower()},
                        timeout=900)
    return resp.json()


def wait(job_id: str, timeout: float = 3 * 3600, poll_interval: float = 2.0) -> dict:
    deadline = time.time() + timeout
    while True:
        resp = _request("GET", f"/v1/stt/transcriptions/{job_id}", headers=_auth(), timeout=60)
        job = resp.json()
        if job.get("status") == "completed":
            return job
        if job.get("status") == "failed":
            err = job.get("error") or {}
            raise VoiceLabError(f"Транскрипция {job_id} не удалась: {err.get('message', err)}")
        if time.time() > deadline:
            raise VoiceLabError(f"Таймаут ожидания транскрипции {job_id}")
        ra = resp.headers.get("Retry-After")
        time.sleep(float(ra) if ra and ra.replace(".", "", 1).isdigit() else poll_interval)


def transcribe(audio_path: str | Path, language: str = "uz", include_speakers: bool = True,
               duration_s: float | None = None, state_file: str | Path | None = None,
               poll_interval: float = 2.0, timeout: float = 3 * 3600, billing: dict | None = None) -> dict:
    """Отправляет файл в VoiceLab и ждёт готовую транскрипцию.

    state_file хранит Idempotency-Key и id задачи: если процесс оборвался, следующий запуск
    продолжит опрос той же задачи вместо повторной (платной) отправки.
    Возвращает Transcription: transcript, language, duration_ms, segments[{start_ms, end_ms, text,
    speaker?, words?}], speakers[].
    billing (если передан) заполняется: job_id, credits (оценка VoiceLab по фактической длительности или,
    если её не получить, оценка перед отправкой; None, если неизвестно), duration_ms.
    """
    state_path = Path(state_file) if state_file else None
    state = json.loads(state_path.read_text()) if state_path and state_path.exists() else {}
    opts = {"language": language, "include_speakers": include_speakers, "file": Path(audio_path).name}
    if state.get("opts") != opts:  # другой файл или параметры -> новый ключ
        state = {"opts": opts, "idempotency_key": str(uuid.uuid4())}

    def save():
        if state_path:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps(state))

    if not state.get("job_id"):
        if duration_s:
            pf = preflight(int(duration_s * 1000), state["idempotency_key"])
            if not pf.get("can_upload", True):
                raise VoiceLabError(f"Недостаточно кредитов VoiceLab (нужно ≈{pf.get('estimated_credits')})")
            state["estimated_credits"] = pf.get("estimated_credits")
        save()
        job = submit(audio_path, language, include_speakers, state["idempotency_key"])
        state["job_id"] = job["id"]
        save()
    if billing is not None:
        billing["job_id"] = state["job_id"]
    result = wait(state["job_id"], timeout=timeout, poll_interval=poll_interval)
    if billing is not None:
        billing["duration_ms"] = result.get("duration_ms")
        billing["credits"] = state.get("estimated_credits")
        if result.get("duration_ms"):  # списание идёт по фактической длительности: уточняем оценку (бесплатно)
            try:
                billing["credits"] = preflight(int(result["duration_ms"])).get("estimated_credits", billing["credits"])
            except VoiceLabError:
                pass
    if state_path:
        state_path.unlink(missing_ok=True)
    return result
