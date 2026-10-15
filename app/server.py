"""Веб-интерфейс: загрузка видео/аудио -> VoiceLab -> проверка и правка сегментов.

Запуск:  uvicorn app.server:app --host 0.0.0.0 --port 8000
"""
import json
import os
import queue
import threading
import traceback
from contextlib import contextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import db
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
    with dst.open("wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    with conn() as c:
        job_id = c.execute("INSERT INTO jobs (filename, status, message) VALUES (?, 'queued', 'В очереди')",
                           (dst.name,)).lastrowid
    _jobs.put(job_id)
    return {"id": job_id, "filename": dst.name}


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


@app.put("/api/segments/{seg_id}")
def update_segment(seg_id: int, body: SegmentUpdate):
    fields = body.model_dump(exclude_none=True)
    if "status" in fields and fields["status"] not in STATUSES:
        raise HTTPException(400, "Неизвестный статус")
    with conn() as c:
        _get(c, seg_id)
        if fields:
            sets = ", ".join(f"{k} = :{k}" for k in fields)
            c.execute(f"UPDATE segments SET {sets}, claimed_by = NULL, claimed_at = NULL, "
                      f"updated_at = CURRENT_TIMESTAMP WHERE id = :id", {**fields, "id": seg_id})
        return _get(c, seg_id)


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


@app.get("/api/export.jsonl", response_class=PlainTextResponse)
def export(status: str = "approved"):
    with conn() as c:
        rows = c.execute("SELECT * FROM segments WHERE status = ? ORDER BY video, start", (status,))
        lines = [json.dumps({"audio_filepath": r["audio_path"], "duration": r["duration"], "text": r["text"],
                             "speaker": r["speaker"], "video": r["video"]}, ensure_ascii=False) for r in rows]
    return PlainTextResponse("\n".join(lines) + ("\n" if lines else ""), media_type="application/x-ndjson",
                             headers={"Content-Disposition": f'attachment; filename="{status}.jsonl"'})


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
