"""Учёт расхода STT: каждая отправка и каждый пропущенный дубликат пишутся в data/usage.jsonl.

Файл только дописывается и не уходит в корзину при «Очистить»: потраченные кредиты остаются потраченными.
Строка: {"time", "video", "file", "event", "duration_s", "credits", "job_id", "provider"}
    provider = xonsaroy | voicelab (у старых строк поля нет: это voicelab)
    event = charged    файл расшифрован, credits = оценка VoiceLab по фактической длительности
                       (у xonsaroy кредитов нет: credits = None, duration_s = отправленная речь)
            failed     задача VoiceLab упала, резерв кредитов возвращается (credits = 0)
            duplicate  тот же звук уже был, VoiceLab не вызывался (duration_s = сэкономлено)
"""
import json
import threading
import time
from collections import defaultdict
from pathlib import Path

LOG = "usage.jsonl"
_lock = threading.Lock()


def record(out: Path, video: str, event: str, duration_s: float = 0.0, credits: int | None = None,
           file: str | None = None, job_id: str | None = None, provider: str = "voicelab") -> dict:
    entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "video": video, "file": file, "event": event,
             "duration_s": round(duration_s or 0.0, 3), "credits": credits, "job_id": job_id,
             "provider": provider}
    with _lock:
        out.mkdir(parents=True, exist_ok=True)
        with (out / LOG).open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def load(out: Path) -> list[dict]:
    p = out / LOG
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:  # недописанная строка после сбоя
            continue
    return rows


def summary(out: Path) -> dict:
    """Итого, по дням и по файлам. Кредиты None у VoiceLab (оценку не получили) не суммируются, а считаются
    в unknown; у xonsaroy кредитов нет вообще, его строки дают только минуты."""
    rows = load(out)
    total = {"credits": 0, "minutes": 0.0, "files": 0, "failed": 0, "unknown": 0,
             "saved_files": 0, "saved_minutes": 0.0}
    days: dict[str, dict] = defaultdict(lambda: {"credits": 0, "minutes": 0.0, "files": 0})
    for r in rows:
        mins = (r.get("duration_s") or 0) / 60
        if r["event"] == "charged":
            day = days[r["time"][:10]]
            total["files"] += 1; day["files"] += 1
            total["minutes"] += mins; day["minutes"] += mins
            if r.get("credits") is None:
                total["unknown"] += r.get("provider", "voicelab") == "voicelab"
            else:
                total["credits"] += r["credits"]; day["credits"] += r["credits"]
        elif r["event"] == "failed":
            total["failed"] += 1
        elif r["event"] == "duplicate":
            total["saved_files"] += 1; total["saved_minutes"] += mins
    for d in [total, *days.values()]:
        for k in ("minutes", "saved_minutes"):
            if k in d:
                d[k] = round(d[k], 2)
    return {"total": total, "by_day": [{"day": k, **v} for k, v in sorted(days.items(), reverse=True)],
            "rows": rows[::-1]}


def fmt(total: dict) -> str:
    s = f'{total["minutes"]:.1f} мин, файлов {total["files"]}, кредитов VoiceLab {total["credits"]}'
    if total["unknown"]:
        s += f' (у {total["unknown"]} кредиты неизвестны)'
    return s
