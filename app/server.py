"""Веб-интерфейс: загрузка видео/аудио -> VoiceLab -> проверка и правка сегментов.

Запуск:  uvicorn app.server:app --host 0.0.0.0 --port 8000
"""
import hashlib
import json
import os
import queue
import shutil
import sqlite3
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from pydantic import BaseModel

from app import bundle, db
from pipeline import prepare, voicelab

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("DATA_DIR", ROOT / "data")).resolve()
DB_PATH = DATA / "speech_lab.db"
UPLOADS = DATA / "videos"
CLAIM_MINUTES = 10
STATUSES = {"pending", "approved", "rejected", "overlap"}

voicelab.load_env(ROOT / ".env")
DATA.mkdir(parents=True, exist_ok=True)
UPLOADS.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Speech Lab")


@contextmanager
def conn():
    c = db.connect(DB_PATH)
    try:
        yield c
        c.commit()
    finally:
        c.close()


# ---------- очередь обработки: по одному файлу за раз ----------

_jobs: "queue.Queue[int]" = queue.Queue()


def _set_job(job_id: int, **fields) -> None:
    sets = ", ".join(f"{k} = :{k}" for k in fields)
    with conn() as c:
        c.execute(f"UPDATE jobs SET {sets}, updated_at = CURRENT_TIMESTAMP WHERE id = :id",
                  {**fields, "id": job_id})


def _worker() -> None:
    args = prepare.build_parser().parse_args(["--out", str(DATA), "--keep-overlap"])
    while True:
        job_id = _jobs.get()
        c = db.connect(DB_PATH)
        try:
            job = c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            _set_job(job_id, status="processing", message="Конвертация и отправка в VoiceLab")
            res = prepare.process(UPLOADS / job["filename"], DATA, c, args)
            if res.get("duplicate_of"):  # другой файл, но звук тот же, что у уже расшифрованного видео
                (UPLOADS / job["filename"]).unlink(missing_ok=True)
                _set_job(job_id, status="duplicate", video=res["duplicate_of"], segments=0,
                         message=f"Тот же звук, что у {res['duplicate_of']}: VoiceLab не вызывался")
                continue
            prepare.write_manifest(c, DATA)
            _set_job(job_id, status="done", video=res["video"], segments=res["segments"],
                     message=f"Сегментов: {res['segments']}, коротких отброшено: {res['short']}")
        except Exception as e:  # ошибку показываем в интерфейсе, очередь продолжает работать
            traceback.print_exc()
            _set_job(job_id, status="error", message=str(e)[:500])
        finally:
            c.close()
            _jobs.task_done()


@app.on_event("startup")
def _start_worker() -> None:
    with conn() as c:  # задачи, прерванные перезапуском сервера, ставим в очередь заново
        for r in c.execute("SELECT id FROM jobs WHERE status IN ('queued', 'processing') ORDER BY id"):
            _jobs.put(r["id"])
    exports = DATA / "exports"  # недособранные архивы и ссылки для скачивания от прошлого запуска
    for p in [*exports.glob(".*.part"), *exports.glob(".serve/*.zip")]:
        p.unlink(missing_ok=True)
    threading.Thread(target=_worker, daemon=True).start()


# ---------- загрузка ----------

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    name = Path(file.filename or "file").name
    if Path(name).suffix.lower() not in prepare.MEDIA_EXT:
        raise HTTPException(400, f"Неподдерживаемый формат: {Path(name).suffix}")
    stem, ext = prepare.safe_name(Path(name).stem), Path(name).suffix.lower()
    dst, i = UPLOADS / f"{stem}{ext}", 1
    while dst.exists():  # одинаковые имена не перезаписываем
        dst, i = UPLOADS / f"{stem}_{i}{ext}", i + 1
    h = hashlib.sha256()
    with dst.open("wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk); h.update(chunk)
    with conn() as c:
        # тот же файл уже загружали: не ставим в очередь, кредиты VoiceLab не тратятся
        dup = c.execute("SELECT filename, video FROM jobs WHERE sha256 = ? AND status != 'error' ORDER BY id LIMIT 1",
                        (h.hexdigest(),)).fetchone()
        # тот же файл уже падал с ошибкой: повторяем ту задачу. Имя видео то же, поэтому VoiceLab
        # продолжит уже созданную расшифровку (Idempotency-Key из transcripts/*.pending.json), а не спишет заново
        failed = c.execute("SELECT id, filename FROM jobs WHERE sha256 = ? AND status = 'error' ORDER BY id DESC LIMIT 1",
                           (h.hexdigest(),)).fetchone()
        if not dup and failed and (UPLOADS / failed["filename"]).exists():
            dst.unlink(missing_ok=True)
            _requeue(c, failed["id"])
            return {"id": failed["id"], "filename": failed["filename"], "retried": True}
        if dup:
            dst.unlink(missing_ok=True)
            raise HTTPException(409, f"Этот файл уже загружен как {dup['filename']}"
                                     f"{' (видео ' + dup['video'] + ')' if dup['video'] else ''}")
        job_id = c.execute("INSERT INTO jobs (filename, status, message, sha256) VALUES (?, 'queued', 'В очереди', ?)",
                           (dst.name, h.hexdigest())).lastrowid
    _jobs.put(job_id)
    return {"id": job_id, "filename": dst.name}


def _requeue(c: sqlite3.Connection, job_id: int) -> None:
    c.execute("UPDATE jobs SET status = 'queued', message = 'Повтор: в очереди' WHERE id = ?", (job_id,))
    c.commit()
    _jobs.put(job_id)


@app.post("/api/jobs/{job_id}/retry")
def retry_job(job_id: int):
    """Повторить упавшую задачу с тем же файлом: уже оплаченная расшифровка VoiceLab не оплачивается снова."""
    with conn() as c:
        job = c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not job:
            raise HTTPException(404, "Задача не найдена")
        if job["status"] != "error":
            raise HTTPException(409, "Повторить можно только задачу с ошибкой")
        if not (UPLOADS / job["filename"]).exists():
            raise HTTPException(409, "Файл задачи уже удалён: загрузите видео заново")
        _requeue(c, job_id)
    return {"id": job_id, "status": "queued"}


@app.get("/api/jobs")
def jobs():
    with conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT 50")]


# ---------- сегменты ----------

@app.get("/api/stats")
def stats():
    with conn() as c:
        rows = c.execute("SELECT status, COUNT(*) n, COALESCE(SUM(duration), 0) d FROM segments GROUP BY status")
        by = {r["status"]: {"count": r["n"], "hours": round(r["d"] / 3600, 3)} for r in rows}
        videos = [dict(r) for r in c.execute(
            "SELECT video, COUNT(*) total, SUM(status = 'approved') approved FROM segments GROUP BY video ORDER BY video")]
    return {"by_status": by, "videos": videos}


@app.get("/api/segments")
def segments(status: str | None = None, video: str | None = None, q: str | None = None,
             limit: int = 200, offset: int = 0):
    where, params = [], []
    if status:
        where.append("status = ?"); params.append(status)
    if video:
        where.append("video = ?"); params.append(video)
    if q:
        where.append("(text LIKE ? OR auto_text LIKE ? OR speaker LIKE ?)"); params += [f"%{q}%"] * 3
    sql = "SELECT * FROM segments" + (" WHERE " + " AND ".join(where) if where else "")
    with conn() as c:
        total = c.execute(sql.replace("SELECT *", "SELECT COUNT(*)"), params).fetchone()[0]
        rows = c.execute(sql + " ORDER BY video, start LIMIT ? OFFSET ?", params + [limit, offset])
        return {"total": total, "items": [dict(r) for r in rows]}


def _get(c, seg_id: int) -> dict:
    r = c.execute("SELECT * FROM segments WHERE id = ?", (seg_id,)).fetchone()
    if not r:
        raise HTTPException(404, "Сегмент не найден")
    return dict(r)


@app.get("/api/segments/{seg_id}")
def get_segment(seg_id: int):
    with conn() as c:
        seg = _get(c, seg_id)
        # соседние реплики того же видео: помогают понять, на своём ли месте текст
        seg["prev"] = [dict(r) for r in c.execute(
            "SELECT id, start, end, speaker, text FROM segments WHERE video = ? AND start < ? ORDER BY start DESC LIMIT 2",
            (seg["video"], seg["start"]))][::-1]
        seg["next"] = [dict(r) for r in c.execute(
            "SELECT id, start, end, speaker, text FROM segments WHERE video = ? AND start > ? ORDER BY start LIMIT 2",
            (seg["video"], seg["start"]))]
        return seg


class SegmentUpdate(BaseModel):
    text: str | None = None
    speaker: str | None = None
    status: str | None = None
    annotator: str | None = None
    notes: str | None = None
    start: float | None = None  # новые границы сегмента в секундах исходного видео
    end: float | None = None


def _full_wav(video: str) -> Path:
    return DATA / "audio" / f"{video}.wav"


def _recut(seg: dict, start: float, end: float) -> dict:
    """Режет сегмент заново из полного аудио по новым границам."""
    wav = _full_wav(seg["video"])
    if not wav.exists():
        raise HTTPException(409, "Полное аудио видео не найдено, границы изменить нельзя")
    with prepare.wave.open(str(wav), "rb") as r:
        total = r.getnframes() / r.getframerate()
    start, end = round(max(0.0, start), 3), round(min(total, end), 3)
    if end - start < 0.2:
        raise HTTPException(400, "Сегмент слишком короткий")
    rel = f"segments/{seg['video']}/{seg['video']}_{int(start * 1000):08d}_{int(end * 1000):08d}.wav"
    prepare.cut_wav(wav, DATA / rel, start, end)
    if seg["audio_path"] and rel != seg["audio_path"]:
        (DATA / seg["audio_path"]).unlink(missing_ok=True)
    return {"start": start, "end": end, "duration": round(end - start, 3), "audio_path": rel}


class SegmentNew(BaseModel):
    video: str
    start: float
    end: float
    speaker: str | None = None
    text: str = ""
    annotator: str | None = None
    # для отмены удаления: вернуть сегмент как был
    auto_text: str = ""
    status: str = "pending"
    notes: str | None = "добавлен вручную"


@app.post("/api/segments")
def create_segment(body: SegmentNew):
    """Новый сегмент вручную: речь, которую VoiceLab пропустил. wav режется из полного аудио."""
    cut = _recut({"video": body.video, "audio_path": ""}, body.start, body.end)
    with conn() as c:
        if c.execute("SELECT 1 FROM segments WHERE audio_path = ?", (cut["audio_path"],)).fetchone():
            raise HTTPException(409, "Сегмент с такими же границами уже есть")
        if body.status not in STATUSES:
            raise HTTPException(400, "Неизвестный статус")
        seg_id = c.execute(
            "INSERT INTO segments (video, audio_path, start, end, duration, speaker, auto_text, text, status, annotator, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (body.video, cut["audio_path"], cut["start"], cut["end"], cut["duration"], body.speaker, body.auto_text,
             body.text.strip(), body.status, body.annotator, body.notes),
        ).lastrowid
        return _get(c, seg_id)


@app.delete("/api/segments/{seg_id}")
def delete_segment(seg_id: int):
    """Удаляет сегмент и его wav. Отменить можно, создав его заново по возвращённым данным."""
    with conn() as c:
        seg = _get(c, seg_id)
        c.execute("DELETE FROM segments WHERE id = ?", (seg_id,))
    (DATA / seg["audio_path"]).unlink(missing_ok=True)
    return seg


@app.put("/api/segments/{seg_id}")
def update_segment(seg_id: int, body: SegmentUpdate):
    fields = body.model_dump(exclude_none=True)
    if "status" in fields and fields["status"] not in STATUSES:
        raise HTTPException(400, "Неизвестный статус")
    with conn() as c:
        seg = _get(c, seg_id)
        start, end = fields.pop("start", None), fields.pop("end", None)
        if start is not None or end is not None:
            start = seg["start"] if start is None else start
            end = seg["end"] if end is None else end
            if abs(start - seg["start"]) > 1e-3 or abs(end - seg["end"]) > 1e-3:
                fields.update(_recut(seg, start, end))
        if fields:
            sets = ", ".join(f"{k} = :{k}" for k in fields)
            c.execute(f"UPDATE segments SET {sets}, claimed_by = NULL, claimed_at = NULL, "
                      f"updated_at = CURRENT_TIMESTAMP WHERE id = :id", {**fields, "id": seg_id})
        return _get(c, seg_id)


# ---------- студия: исходное видео, волна, таймлайн ----------

def _source(video: str) -> Path | None:
    """Исходный файл видео/аудио, из которого сделаны сегменты."""
    with conn() as c:
        r = c.execute("SELECT filename FROM jobs WHERE video = ? AND status = 'done' ORDER BY id DESC LIMIT 1",
                      (video,)).fetchone()
    if r and (UPLOADS / r["filename"]).exists():
        return UPLOADS / r["filename"]
    for p in sorted(UPLOADS.glob("*")):
        if p.suffix.lower() in prepare.MEDIA_EXT and prepare.safe_name(p.stem) == video:
            return p
    return None


@app.get("/media/{video}")
def media(video: str, audio_only: bool = False):
    src = None if audio_only else _source(video)
    path = src or _full_wav(video)
    if not path.exists():
        raise HTTPException(404, "Файл не найден")
    return FileResponse(path)


@app.get("/api/waveform/{video}")
def waveform(video: str, per_sec: int = 50):
    """Пики громкости для отрисовки волны: per_sec значений на секунду (0..1)."""
    wav = _full_wav(video)
    if not wav.exists():
        raise HTTPException(404, "Аудио не найдено")
    cache = DATA / "waveforms" / f"{video}_{per_sec}.json"
    if cache.exists() and cache.stat().st_mtime >= wav.stat().st_mtime:
        return json.loads(cache.read_text())
    import array
    with prepare.wave.open(str(wav), "rb") as r:
        sr, n = r.getframerate(), r.getnframes()
        samples = array.array("h", r.readframes(n))
    step = max(1, sr // per_sec)
    peaks = [max(map(abs, samples[i:i + step]), default=0) for i in range(0, len(samples), step)]
    top = max(peaks, default=1) or 1
    res = {"per_sec": per_sec, "duration": n / sr, "peaks": [round(p / top, 3) for p in peaks]}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(res))
    return res


@app.get("/api/timeline/{video}")
def timeline(video: str):
    with conn() as c:
        segs = [dict(r) for r in c.execute("SELECT * FROM segments WHERE video = ? ORDER BY start", (video,))]
    wav = _full_wav(video)
    duration = 0.0
    if wav.exists():
        with prepare.wave.open(str(wav), "rb") as r:
            duration = r.getnframes() / r.getframerate()
    src = _source(video)
    return {"video": video, "duration": duration or max((s["end"] for s in segs), default=0),
            "source": src.name if src else None, "segments": segs, **_speakers(video, segs)}


# ---------- спикеры: метки SPEAKER_xx остаются, редактор даёт имена и переназначает ----------

def _speakers(video: str, segs: list[dict] | None = None) -> dict:
    with conn() as c:
        if segs is None:
            segs = [dict(r) for r in c.execute("SELECT speaker FROM segments WHERE video = ?", (video,))]
        rows = c.execute("SELECT speaker, name FROM speakers WHERE video = ?", (video,)).fetchall()
    ids = {s["speaker"] for s in segs if s["speaker"]} | {r["speaker"] for r in rows}
    return {"speakers": sorted(ids), "speaker_names": {r["speaker"]: r["name"] for r in rows if r["name"]}}


class SpeakerIn(BaseModel):
    name: str | None = None


class MergeIn(BaseModel):
    into: str


@app.post("/api/speakers/{video}")
def add_speaker(video: str):
    """Новый пустой спикер со следующим номером: <video>_SPEAKER_NN."""
    existing = _speakers(video)["speakers"]
    nums = [int(s.rsplit("_", 1)[1]) for s in existing if s.rsplit("_", 1)[-1].isdigit()]
    spk = f"{video}_SPEAKER_{(max(nums) + 1 if nums else 0):02d}"
    with conn() as c:
        c.execute("INSERT OR IGNORE INTO speakers (video, speaker) VALUES (?, ?)", (video, spk))
    return {"speaker": spk, **_speakers(video)}


@app.put("/api/speakers/{video}/{speaker}")
def name_speaker(video: str, speaker: str, body: SpeakerIn):
    """Имя спикера (например, «Ali»); пустое имя возвращает просто метку."""
    name = (body.name or "").strip() or None
    with conn() as c:
        c.execute("INSERT INTO speakers (video, speaker, name) VALUES (?, ?, ?) "
                  "ON CONFLICT(video, speaker) DO UPDATE SET name = excluded.name", (video, speaker, name))
    return _speakers(video)


@app.post("/api/speakers/{video}/{speaker}/merge")
def merge_speaker(video: str, speaker: str, body: MergeIn):
    """Все реплики спикера переходят к другому (когда VoiceLab разделил одного человека на двоих)."""
    if body.into == speaker:
        raise HTTPException(400, "Нельзя объединить спикера с самим собой")
    if body.into not in _speakers(video)["speakers"]:
        raise HTTPException(404, "Спикер не найден")
    with conn() as c:
        n = c.execute("UPDATE segments SET speaker = ?, updated_at = CURRENT_TIMESTAMP WHERE video = ? AND speaker = ?",
                      (body.into, video, speaker)).rowcount
        c.execute("DELETE FROM speakers WHERE video = ? AND speaker = ?", (video, speaker))
    return {"moved": n, **_speakers(video)}


@app.get("/api/next")
def next_segment(annotator: str = "anon", video: str | None = None, after: int | None = None):
    """Следующий pending-сегмент; закрепляется за разметчиком на CLAIM_MINUTES минут."""
    sql = (f"SELECT id FROM segments WHERE status = 'pending' AND (claimed_by IS NULL OR claimed_by = ? "
           f"OR claimed_at < datetime('now', '-{CLAIM_MINUTES} minutes'))")
    params: list = [annotator]
    if video:
        sql += " AND video = ?"; params.append(video)
    if after:
        sql += " AND id != ?"; params.append(after)
    with conn() as c:
        r = c.execute(sql + " ORDER BY video, start LIMIT 1", params).fetchone()
        if not r:
            return None
        c.execute("UPDATE segments SET claimed_by = ?, claimed_at = datetime('now') WHERE id = ?", (annotator, r["id"]))
    return get_segment(r["id"])


@app.get("/audio/{seg_id}")
def audio(seg_id: int):
    with conn() as c:
        seg = _get(c, seg_id)
    path = (DATA / seg["audio_path"]).resolve()
    if DATA not in path.parents or not path.exists():
        raise HTTPException(404, "Аудио не найдено")
    return FileResponse(path, media_type="audio/wav")


# ---------- очистка с корзиной: всё уходит в data/trash/<время>/ и возвращается одной кнопкой ----------

TRASH = DATA / "trash"
DATA_PARTS = ["videos", "audio", "segments", "transcripts", "waveforms", "exports", "manifest.jsonl"]
_trash_lock = threading.Lock()


def _busy() -> bool:
    with conn() as c:
        return bool(c.execute("SELECT 1 FROM jobs WHERE status IN ('queued', 'processing') LIMIT 1").fetchone())


def _trash_info(d: Path) -> dict:
    info = {"name": d.name, "created": (d / "speech_lab.db").stat().st_mtime, "videos": 0, "segments": 0}
    try:
        c = sqlite3.connect(d / "speech_lab.db")
        try:
            info["segments"] = c.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
            info["videos"] = c.execute("SELECT COUNT(DISTINCT video) FROM segments").fetchone()[0]
        finally:
            c.close()
    except sqlite3.Error:
        pass
    return info


@app.post("/api/clear")
def clear_all():
    """Убирает все видео, аудио, сегменты и тексты в корзину. Ничего не удаляется насовсем."""
    with _trash_lock:
        if _busy():
            raise HTTPException(409, "Идёт обработка видео: дождитесь окончания очереди")
        with conn() as c:
            n = c.execute("SELECT COUNT(*) FROM segments").fetchone()[0] + \
                c.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        if not n and not any(UPLOADS.iterdir()):
            raise HTTPException(400, "Нечего очищать")
        dst = TRASH / time.strftime("%Y-%m-%d_%H-%M-%S")
        dst.mkdir(parents=True, exist_ok=False)
        backup = sqlite3.connect(dst / "speech_lab.db")
        try:
            with conn() as c:
                c.backup(backup)  # копия базы: из неё восстанавливаются сегменты, правки, спикеры и задачи
        finally:
            backup.close()
        for part in DATA_PARTS:
            if (DATA / part).exists():
                shutil.move(str(DATA / part), str(dst / part))
        with conn() as c:
            c.execute("DELETE FROM segments"); c.execute("DELETE FROM speakers"); c.execute("DELETE FROM jobs")
        UPLOADS.mkdir(parents=True, exist_ok=True)
    return _trash_info(dst)


@app.get("/api/trash")
def trash_list():
    if not TRASH.exists():
        return []
    return [_trash_info(d) for d in sorted(TRASH.iterdir(), reverse=True) if (d / "speech_lab.db").exists()]


def _trash_dir(name: str) -> Path:
    d = TRASH / Path(name).name
    if not (d / "speech_lab.db").exists():
        raise HTTPException(404, "В корзине такого нет")
    return d


def _move_back(src: Path, dst: Path) -> None:
    """Возвращает файлы на место; если файл с тем же именем уже появился, текущий не трогаем."""
    if src.is_file():
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        return
    for p in src.rglob("*"):
        if p.is_file():
            _move_back(p, dst / p.relative_to(src))


@app.post("/api/trash/{name}/restore")
def trash_restore(name: str):
    with _trash_lock:
        d = _trash_dir(name)
        index = DATA / "transcripts" / "audio_index.json"
        old_index = d / "transcripts" / "audio_index.json"
        merged = {**(json.loads(old_index.read_text(encoding="utf-8")) if old_index.exists() else {}),
                  **(json.loads(index.read_text(encoding="utf-8")) if index.exists() else {})}
        for part in DATA_PARTS:
            if (d / part).exists():
                _move_back(d / part, DATA / part)
        if merged:
            index.parent.mkdir(parents=True, exist_ok=True)
            index.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
        with conn() as c:
            c.execute("ATTACH DATABASE ? AS old", (str(d / "speech_lab.db"),))
            cols = ", ".join(r[1] for r in c.execute("PRAGMA old.table_info(segments)") if r[1] != "id")
            n = c.execute(f"INSERT OR IGNORE INTO main.segments ({cols}) SELECT {cols} FROM old.segments").rowcount
            c.execute("INSERT OR IGNORE INTO main.speakers SELECT * FROM old.speakers")
            jcols = ", ".join(r[1] for r in c.execute("PRAGMA old.table_info(jobs)") if r[1] != "id")
            c.execute(f"INSERT INTO main.jobs ({jcols}) SELECT {jcols} FROM old.jobs WHERE status NOT IN ('queued', 'processing')")
            c.commit()
            c.execute("DETACH DATABASE old")
            prepare.write_manifest(c, DATA)
        shutil.rmtree(d)
    return {"restored": n}


@app.delete("/api/trash/{name}")
def trash_delete(name: str):
    """Удалить из корзины насовсем."""
    with _trash_lock:
        shutil.rmtree(_trash_dir(name))
    return {"ok": True}


@app.get("/api/export.jsonl", response_class=PlainTextResponse)
def export(status: str = "approved"):
    with conn() as c:
        names = db.speaker_names(c)
        rows = c.execute("SELECT * FROM segments WHERE status = ? ORDER BY video, start", (status,))
        lines = [json.dumps({"audio_filepath": r["audio_path"], "duration": r["duration"], "text": r["text"],
                             "speaker": r["speaker"], "speaker_name": names.get((r["video"], r["speaker"])),
                             "video": r["video"]}, ensure_ascii=False) for r in rows]
    return PlainTextResponse("\n".join(lines) + ("\n" if lines else ""), media_type="application/x-ndjson",
                             headers={"Content-Disposition": f'attachment; filename="{status}.jsonl"'})


TEXT_FORMATS = {"srt": "application/x-subrip", "txt": "text/plain"}


def _text_status(status: str | None) -> None:
    if status and status not in STATUSES - {"rejected"}:
        raise HTTPException(400, "Неизвестный статус (брак в выгрузку не попадает)")


@app.get("/api/transcripts.zip")
def export_transcripts(fmt: str = "all", status: str | None = None):
    """Только тексты всех видео в SRT и/или TXT, с правками и именами спикеров. Копии в data/exports/transcripts/."""
    fmts = list(TEXT_FORMATS) if fmt == "all" else [fmt]
    if not set(fmts) <= TEXT_FORMATS.keys():
        raise HTTPException(400, "Формат: srt, txt или all")
    _text_status(status)
    with conn() as c:
        body, counts = bundle.build_transcripts(c, DATA / "exports" / "transcripts", fmts, status)
    if not counts["videos"]:
        raise HTTPException(404, "Нет сегментов для выгрузки")
    name = f"transcripts_{fmt}{'_' + status if status else ''}.zip"
    return Response(body, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/api/transcript/{video}.{fmt}")
def export_transcript(video: str, fmt: str, status: str | None = None):
    """Текст одного видео в SRT или TXT."""
    if fmt not in TEXT_FORMATS:
        raise HTTPException(400, "Формат: srt или txt")
    _text_status(status)
    with conn() as c:
        text = bundle.transcript(c, video, fmt, status)
    if text is None:
        raise HTTPException(404, "Нет сегментов для выгрузки")
    copy = DATA / "exports" / "transcripts" / f"{video}.{fmt}"
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_text(text, encoding="utf-8")
    return Response(text.encode("utf-8"), media_type=f"{TEXT_FORMATS[fmt]}; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{video}.{fmt}"'})


_zip_locks: dict[str, threading.Lock] = {}
_zip_locks_guard = threading.Lock()


@app.get("/api/export.zip")
def export_zip(video: str | None = None, status: str | None = None):
    """Всё в одном архиве: видео, аудио, сегменты, текст. Копия остаётся в data/exports/."""
    if status and status not in STATUSES - {"rejected"}:
        raise HTTPException(400, "Неизвестный статус (брак в архив не попадает)")
    name = f"{video or 'speech_lab_all'}{'_' + status if status else ''}.zip"
    dst = DATA / "exports" / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    asked_at = time.time()
    with _zip_locks_guard:
        lock = _zip_locks.setdefault(name, threading.Lock())
    # повторный клик или повторный запрос браузера ждёт уже идущую сборку и отдаёт её результат
    with lock:
        if not (dst.exists() and dst.stat().st_mtime >= asked_at):
            tmp = dst.with_name(f".{name}.{uuid.uuid4().hex}.part")
            try:
                with conn() as c:
                    counts = bundle.build_zip(c, DATA, tmp, _source, [video] if video else None, status)
                if not counts["segments"]:
                    raise HTTPException(404, "Нет сегментов для архива")
                tmp.replace(dst)
            except OSError as e:
                traceback.print_exc()
                raise HTTPException(500, f"Не удалось собрать архив: {e}") from e
            finally:
                tmp.unlink(missing_ok=True)
        # отдаём отдельную жёсткую ссылку, чтобы следующая пересборка не подменила файл посреди скачивания
        serve, cleanup = dst, None
        link = dst.parent / ".serve" / f"{uuid.uuid4().hex}.zip"
        try:
            link.parent.mkdir(exist_ok=True)
            os.link(dst, link)
            serve, cleanup = link, BackgroundTask(link.unlink, missing_ok=True)
        except OSError:
            pass
    return FileResponse(serve, media_type="application/zip", filename=name, background=cleanup)


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
