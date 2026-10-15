"""ZIP-архив датасета: видео + аудио + сегменты + текст в одном файле.

Структура архива:
    manifest.jsonl / manifest.csv       все сегменты всех видео в архиве (пути внутри архива)
    <video>/video.<ext>                 исходный файл (если есть)
    <video>/audio.wav                   полное аудио 16 kHz mono
    <video>/segments/*.wav              нарезанные реплики
    <video>/transcript.txt / .srt       расшифровка с правками разметчиков
    <video>/segments.jsonl              сегменты этого видео
    <video>/voicelab.json               сырой ответ VoiceLab
"""
import csv
import io
import json
import sqlite3
import zipfile
from pathlib import Path

from app import db
from pipeline import prepare

MANIFEST_FIELDS = ["audio_filepath", "duration", "text", "auto_text", "speaker", "speaker_name", "video",
                   "start", "end", "status", "annotator"]


def _row(r: sqlite3.Row, video: str, names: dict) -> dict:
    return {
        "audio_filepath": f"{video}/segments/{Path(r['audio_path']).name}",
        "duration": r["duration"], "text": r["text"] or r["auto_text"] or "", "auto_text": r["auto_text"],
        "speaker": r["speaker"], "speaker_name": names.get((video, r["speaker"])), "video": video,
        "start": r["start"], "end": r["end"],
        "status": r["status"], "annotator": r["annotator"],
    }


def _who(r: dict) -> str:
    return f'{r["speaker"]} ({r["speaker_name"]})' if r["speaker_name"] else str(r["speaker"])


def _transcripts(rows: list[dict]) -> tuple[str, str]:
    txt = "".join(f'[{prepare.fmt_ts(r["start"], ".")} - {prepare.fmt_ts(r["end"], ".")}] '
                  f'{_who(r)}: {r["text"]}\n' for r in rows)
    srt = "\n".join(f'{i}\n{prepare.fmt_ts(r["start"])} --> {prepare.fmt_ts(r["end"])}\n{_who(r)}: {r["text"]}\n'
                    for i, r in enumerate(rows, 1))
    return txt, srt


def build_zip(conn: sqlite3.Connection, data: Path, dst: Path, source_for, videos: list[str] | None = None,
              status: str | None = None) -> dict:
    """Пишет архив в dst. source_for(video) -> Path исходного файла или None."""
    if videos is None:
        videos = [r[0] for r in conn.execute("SELECT DISTINCT video FROM segments ORDER BY video")]
    all_rows, counts = [], {"videos": 0, "segments": 0}
    names = db.speaker_names(conn)
    # медиа уже сжаты, поэтому без компрессии: быстрее и почти того же размера;
    # strict_timestamps=False: файлы с датой до 1980 (бывает после копирования) не роняют сборку
    with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_STORED, allowZip64=True, strict_timestamps=False) as z:
        for video in videos:
            sql, params = "SELECT * FROM segments WHERE video = ?", [video]
            if status:
                sql += " AND status = ?"; params.append(status)
            segs = conn.execute(sql + " ORDER BY start", params).fetchall()
            if not segs:
                continue
            counts["videos"] += 1
            rows = []
            for r in segs:
                wav = data / r["audio_path"]
                if wav.exists():
                    z.write(wav, f"{video}/segments/{wav.name}")
                    rows.append(_row(r, video, names))
            src = source_for(video)
            if src and src.exists():
                z.write(src, f"{video}/video{src.suffix.lower()}")
            full = data / "audio" / f"{video}.wav"
            if full.exists():
                z.write(full, f"{video}/audio.wav")
            raw = data / "transcripts" / f"{video}.voicelab.json"
            if raw.exists():
                z.write(raw, f"{video}/voicelab.json", compress_type=zipfile.ZIP_DEFLATED)
            txt, srt = _transcripts(rows)
            z.writestr(f"{video}/transcript.txt", txt, compress_type=zipfile.ZIP_DEFLATED)
            z.writestr(f"{video}/transcript.srt", srt, compress_type=zipfile.ZIP_DEFLATED)
            z.writestr(f"{video}/segments.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                       compress_type=zipfile.ZIP_DEFLATED)
            all_rows += rows
        counts["segments"] = len(all_rows)
        z.writestr("manifest.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_rows),
                   compress_type=zipfile.ZIP_DEFLATED)
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=MANIFEST_FIELDS)
        w.writeheader(); w.writerows(all_rows)
        z.writestr("manifest.csv", "﻿" + buf.getvalue(), compress_type=zipfile.ZIP_DEFLATED)  # BOM для Excel
    return counts


def find_source(data: Path, video: str) -> Path | None:
    for p in sorted((data / "videos").glob("*")):
        if p.suffix.lower() in prepare.MEDIA_EXT and prepare.safe_name(p.stem) == video:
            return p
    return None


if __name__ == "__main__":  # python -m app.bundle [--status approved] [--video NAME]
    import argparse
    import os

    ap = argparse.ArgumentParser(description="Собрать ZIP: видео, аудио, сегменты, текст")
    ap.add_argument("--data", default=os.environ.get("DATA_DIR", "data"))
    ap.add_argument("--video")
    ap.add_argument("--status", choices=["pending", "approved", "rejected", "overlap"])
    args = ap.parse_args()
    data = Path(args.data)
    out = data / "exports" / f"{args.video or 'speech_lab_all'}{'_' + args.status if args.status else ''}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    res = build_zip(db.connect(data / "speech_lab.db"), data, out, lambda v: find_source(data, v),
                    [args.video] if args.video else None, args.status)
    print(f"{out}: видео {res['videos']}, сегментов {res['segments']}, {out.stat().st_size / 1e6:.1f} MB")
