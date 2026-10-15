"""MP4 -> wav 16 kHz mono -> VoiceLab STT (текст + спикеры) -> сегменты 1.5–15 с -> SQLite + файлы.

Пример:
    python -m pipeline.prepare --videos data/videos --out data

Что получается в --out:
    audio/<video>.wav                  полное аудио, 16 kHz mono 16-bit PCM
    segments/<video>/<video>_<start_ms>_<end_ms>.wav нарезанные реплики
    transcripts/<video>.voicelab.json  сырой ответ VoiceLab (кэш, повторно не оплачивается)
    transcripts/<video>.txt            расшифровка со спикерами и таймкодами
    transcripts/<video>.srt            субтитры
    manifest.jsonl                     все сегменты из БД (audio_filepath, text, speaker, ...)
    speech_lab.db                      SQLite, таблица segments
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import db  # noqa: E402
from pipeline import voicelab  # noqa: E402

MEDIA_EXT = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4a", ".mp3", ".wav", ".ogg", ".flac"}
SAMPLE_RATE = 16000
PAD = 0.05          # запас по краям сегмента, с
OVERLAP_MIN = 0.1   # пересечение с другим спикером, после которого сегмент считается наложением, с


def safe_name(stem: str) -> str:
    return re.sub(r"[^\w\-]+", "_", stem, flags=re.UNICODE).strip("_") or "video"


def to_wav(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn",
                    "-ac", "1", "-ar", str(SAMPLE_RATE), "-sample_fmt", "s16", str(dst)],
                   check=True)


def upload_file(wav: Path, fmt: str, tmpdir: Path) -> Path:
    """Файл для отправки в VoiceLab: wav как есть или сжатый flac/mp3 (меньше трафика)."""
    if fmt == "wav":
        return wav
    out = tmpdir / f"{wav.stem}.{fmt}"
    codec = ["-c:a", "flac"] if fmt == "flac" else ["-c:a", "libmp3lame", "-b:a", "64k"]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), *codec, str(out)], check=True)
    return out


def get_transcription(wav: Path, cache: Path, args) -> dict:
    if cache.exists() and not args.force:
        print(f"  кэш: {cache.name}")
        return json.loads(cache.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        f = upload_file(wav, args.upload_format, Path(tmp))
        print(f"  VoiceLab: отправка {f.name} ({f.stat().st_size / 1e6:.1f} MB) ...")
        result = voicelab.transcribe(f, language=args.language,
                                     include_speakers=not args.no_speakers)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def raw_utterances(result: dict, video: str) -> list[dict]:
    """Сегменты VoiceLab -> [{start, end, text, speaker, words}] с speaker_id = <video>_SPEAKER_xx."""
    mapping: dict[str, str] = {}
    out = []
    for seg in sorted(result.get("segments") or [], key=lambda s: s["start_ms"]):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        raw_spk = str(seg.get("speaker") or "0")
        if raw_spk not in mapping:
            mapping[raw_spk] = f"SPEAKER_{len(mapping):02d}"
        out.append({
            "start": seg["start_ms"] / 1000, "end": seg["end_ms"] / 1000, "text": text,
            "speaker": f"{video}_{mapping[raw_spk]}",
            "words": [(w["start_ms"] / 1000, w["end_ms"] / 1000, w["text"]) for w in seg.get("words") or []],
        })
    return out


def mark_overlaps(utts: list[dict]) -> None:
    for u in utts:
        u["overlap"] = any(
            o is not u and o["speaker"] != u["speaker"]
            and min(u["end"], o["end"]) - max(u["start"], o["start"]) > OVERLAP_MIN
            for o in utts)


def split_long(u: dict, max_dur: float) -> list[dict]:
    """Режет реплику длиннее max_dur по границам слов; без слов оставляет как есть."""
    if u["end"] - u["start"] <= max_dur or not u["words"]:
        return [u]
    parts, cur = [], []
    for w in u["words"]:
        if cur and w[1] - cur[0][0] > max_dur:
            parts.append(cur)
            cur = []
        cur.append(w)
    if cur:
        parts.append(cur)
    return [{**u, "start": p[0][0], "end": p[-1][1], "text": " ".join(w[2] for w in p).strip(),
             "words": p} for p in parts]


def merge(utts: list[dict], max_dur: float, max_gap: float) -> list[dict]:
    """Склеивает подряд идущие реплики одного спикера, пока сегмент не длиннее max_dur."""
    out: list[dict] = []
    for u in utts:
        prev = out[-1] if out else None
        if (prev and prev["speaker"] == u["speaker"] and not prev["overlap"] and not u["overlap"]
                and u["start"] - prev["end"] <= max_gap and u["end"] - prev["start"] <= max_dur):
            prev["end"] = u["end"]
            prev["text"] = f'{prev["text"]} {u["text"]}'
            prev["words"] = prev["words"] + u["words"]
        else:
            out.append(dict(u))
    return out


def cut_wav(src: Path, dst: Path, start: float, end: float) -> None:
    with wave.open(str(src), "rb") as r:
        sr = r.getframerate()
        r.setpos(int(start * sr))
        frames = r.readframes(int((end - start) * sr))
        params = r.getparams()
    dst.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dst), "wb") as w:
        w.setparams(params)
        w.writeframes(frames)


def fmt_ts(t: float, sep: str = ",") -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def write_transcripts(utts: list[dict], base: Path) -> None:
    lines = [f'[{fmt_ts(u["start"], ".")} - {fmt_ts(u["end"], ".")}] {u["speaker"]}: {u["text"]}' for u in utts]
    base.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    srt = [f'{i}\n{fmt_ts(u["start"])} --> {fmt_ts(u["end"])}\n{u["speaker"]}: {u["text"]}\n'
           for i, u in enumerate(utts, 1)]
    base.with_suffix(".srt").write_text("\n".join(srt), encoding="utf-8")


def process(src: Path, out: Path, conn, args) -> dict:
    video = safe_name(src.stem)
    print(f"[{video}] {src.name}")
    wav = out / "audio" / f"{video}.wav"
    if not wav.exists() or args.force:
        to_wav(src, wav)
    with wave.open(str(wav), "rb") as r:
        total = r.getnframes() / r.getframerate()

    result = get_transcription(wav, out / "transcripts" / f"{video}.voicelab.json", args)
    utts = raw_utterances(result, video)
    write_transcripts(utts, out / "transcripts" / video)

    mark_overlaps(utts)
    utts = [p for u in utts for p in split_long(u, args.max_dur)]
    utts = merge(utts, args.max_dur, args.max_gap)

    rows, stats = [], {"short": 0, "long": 0, "overlap_dropped": 0}
    for u in utts:
        dur = u["end"] - u["start"]
        if dur < args.min_dur:
            stats["short"] += 1
            continue
        if u["overlap"] and not args.keep_overlap:
            stats["overlap_dropped"] += 1
            continue
        notes = None
        if dur > args.max_dur:
            stats["long"] += 1
            notes = f"длиннее {args.max_dur} с: VoiceLab не вернул слова для разбиения"
        start, end = max(0.0, u["start"] - PAD), min(total, u["end"] + PAD)
        seg_path = out / "segments" / video / f"{video}_{int(start * 1000):08d}_{int(end * 1000):08d}.wav"
        cut_wav(wav, seg_path, start, end)
        rows.append({
            "video": video, "audio_path": seg_path.relative_to(out).as_posix(),
            "start": round(start, 3), "end": round(end, 3), "duration": round(end - start, 3),
            "speaker": u["speaker"], "auto_text": u["text"], "text": u["text"],
            "status": "overlap" if u["overlap"] else "pending", "notes": notes,
        })
    added = db.insert_segments(conn, rows)
    print(f"  сегментов: {len(rows)} (новых в БД: {added}), "
          f"отброшено коротких: {stats['short']}, наложений: {stats['overlap_dropped']}")
    return {"video": video, "segments": len(rows), "hours": sum(r["duration"] for r in rows) / 3600, **stats}


def write_manifest(conn, out: Path) -> int:
    rows = conn.execute("SELECT * FROM segments ORDER BY video, start").fetchall()
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({
                "audio_filepath": r["audio_path"], "duration": r["duration"],
                "text": r["text"] or r["auto_text"] or "", "auto_text": r["auto_text"],
                "speaker": r["speaker"], "video": r["video"], "start": r["start"], "end": r["end"],
                "status": r["status"],
            }, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", default="data/videos", help="папка с MP4 (или один файл)")
    ap.add_argument("--out", default="data", help="куда сохранять аудио, сегменты и расшифровки")
    ap.add_argument("--db", default=None, help="путь к SQLite (по умолчанию <out>/speech_lab.db)")
    ap.add_argument("--env", default=str(ROOT / ".env"), help="файл с VOICELAB_API_KEY")
    ap.add_argument("--language", default="uz")
    ap.add_argument("--min-dur", type=float, default=1.5)
    ap.add_argument("--max-dur", type=float, default=15.0)
    ap.add_argument("--max-gap", type=float, default=0.5, help="макс. пауза при склейке реплик, с")
    ap.add_argument("--keep-overlap", action="store_true", help="сохранять наложения со статусом overlap")
    ap.add_argument("--no-speakers", action="store_true", help="не просить VoiceLab разделять спикеров")
    ap.add_argument("--upload-format", choices=["wav", "flac", "mp3"], default="wav",
                    help="в каком формате отправлять аудио в VoiceLab")
    ap.add_argument("--force", action="store_true", help="заново конвертировать и транскрибировать")
    args = ap.parse_args()

    voicelab.load_env(args.env)
    src = Path(args.videos)
    files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.suffix.lower() in MEDIA_EXT)
    if not files:
        sys.exit(f"Нет видео в {src}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    conn = db.connect(args.db or out / "speech_lab.db")

    summary = []
    for f in files:
        try:
            summary.append(process(f, out, conn, args))
        except (voicelab.VoiceLabError, subprocess.CalledProcessError) as e:
            print(f"  ОШИБКА: {e}", file=sys.stderr)
    n = write_manifest(conn, out)
    hours = sum(s["hours"] for s in summary)
    print(f"\nГотово: видео {len(summary)}/{len(files)}, сегментов в этом запуске "
          f"{sum(s['segments'] for s in summary)} ({hours:.2f} ч), всего в manifest.jsonl: {n}")


if __name__ == "__main__":
    main()
