"""MP4 -> wav 16 kHz mono -> STT -> сегменты 1.5–15 с -> SQLite + файлы.

STT выбирается переменной STT_PROVIDER в .env (или флагом --provider):
    xonsaroy (по умолчанию)  локально pyannote делит речь по спикерам (pipeline/diarize.py), реплики режутся
                             на куски 1.5–15 с, каждый распознаётся отдельно (pipeline/xonsaroy.py); без pyannote
                             или с --no-diarization нарезка по паузам и один спикер на видео
    voicelab                 весь файл целиком, VoiceLab сам режет на реплики и делит спикеров

Пример:
    python -m pipeline.prepare --videos data/videos --out data

Что получается в --out:
    audio/<video>.wav                  полное аудио, 16 kHz mono 16-bit PCM
    segments/<video>/<video>_<start_ms>_<end_ms>.wav нарезанные реплики
    transcripts/<video>.voicelab.json  сегменты от STT (кэш, повторно не отправляется; имя файла историческое,
                                       поле provider говорит, какой сервис его сделал)
    transcripts/<video>.diarization.json  реплики спикеров от pyannote (кэш, только xonsaroy)
    transcripts/<video>.txt            расшифровка со спикерами и таймкодами
    transcripts/<video>.srt            субтитры
    manifest.jsonl                     все сегменты из БД (audio_filepath, text, speaker, ...)
    speech_lab.db                      SQLite, таблица segments
"""
import argparse
import array
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import db  # noqa: E402
from pipeline import diarize, usage, voicelab, xonsaroy  # noqa: E402

MEDIA_EXT = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4a", ".mp3", ".wav", ".ogg", ".flac"}
SAMPLE_RATE = 16000
PAD = 0.05          # запас по краям сегмента, с
OVERLAP_MIN = 0.1   # пересечение с другим спикером, после которого сегмент считается наложением, с
PROVIDERS = ("xonsaroy", "voicelab")
STTError = (voicelab.VoiceLabError, xonsaroy.XonsaroyError)


def stt_provider(args) -> str:
    """--provider, иначе STT_PROVIDER из окружения/.env, иначе xonsaroy."""
    p = (getattr(args, "provider", None) or os.environ.get("STT_PROVIDER") or "xonsaroy").strip().lower()
    if p not in PROVIDERS:
        raise xonsaroy.XonsaroyError(f"Неизвестный STT_PROVIDER={p}: допустимо {', '.join(PROVIDERS)}")
    return p


def safe_name(stem: str) -> str:
    return re.sub(r"[^\w\-]+", "_", stem, flags=re.UNICODE).strip("_") or "video"


class MediaError(RuntimeError):
    pass


def to_wav(src: Path, dst: Path) -> None:
    """ffmpeg -> wav 16 kHz mono. При ошибке в сообщение попадает причина от ffmpeg, а не только код выхода."""
    if not src.is_file():
        raise MediaError(f"Файл {src.name} не найден в {src.parent}: загрузите его заново")
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(src), "-vn",
                        "-ac", "1", "-ar", str(SAMPLE_RATE), "-sample_fmt", "s16", str(dst)],
                       capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        dst.unlink(missing_ok=True)  # недописанный wav не должен считаться готовым при повторе
        why = " ".join(r.stderr.strip().splitlines()[-3:]) or f"код {r.returncode}"
        if "no audio" in why.lower() or "does not contain any stream" in why.lower() or "matches no streams" in why.lower():
            why = f"в файле нет звуковой дорожки ({why})"
        raise MediaError(f"ffmpeg не смог прочитать {src.name}: {why}")


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
    if stt_provider(args) == "xonsaroy":
        return get_xonsaroy(wav, cache, args)
    with tempfile.TemporaryDirectory() as tmp:
        f = upload_file(wav, args.upload_format, Path(tmp))
        print(f"  VoiceLab: отправка {f.name} ({f.stat().st_size / 1e6:.1f} MB) ...")
        with wave.open(str(wav), "rb") as r:
            duration = r.getnframes() / r.getframerate()
        bill: dict = {}
        try:
            result = voicelab.transcribe(f, language=args.language,
                                         include_speakers=not args.no_speakers, duration_s=duration,
                                         state_file=cache.with_suffix(".pending.json"), billing=bill)
        except voicelab.VoiceLabError:
            if bill.get("job_id"):  # задача была создана и упала: резерв кредитов VoiceLab возвращает
                usage.record(cache.parent.parent, wav.stem, "failed", duration, 0, f.name, bill["job_id"])
            raise
    dur = (bill.get("duration_ms") or duration * 1000) / 1000
    usage.record(cache.parent.parent, wav.stem, "charged", dur, bill.get("credits"), f.name, bill.get("job_id"))
    print(f"  VoiceLab: {dur / 60:.1f} мин, кредитов ≈{bill.get('credits') if bill.get('credits') is not None else '?'}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def get_turns(wav: Path, cache: Path, args) -> list[dict] | None:
    """Реплики спикеров от pyannote (кэш transcripts/<video>.diarization.json) или None, если диаризация выключена
    или недоступна: тогда Xonsaroy режет по паузам, и всё видео получает одного спикера."""
    args.diarization_error = None
    if getattr(args, "no_diarization", False):
        return None
    path = cache.parent / f"{wav.stem}.diarization.json"
    if path.exists() and not args.force:
        return json.loads(path.read_text(encoding="utf-8"))["turns"]
    on_progress = getattr(args, "on_progress", None)
    device = getattr(args, "device", None)
    try:  # видеокарта есть: pyannote на ней; нет: на процессоре, в 5-10 раз медленнее
        where = diarize.pick_device(device)
    except Exception:  # noqa: BLE001 - torch не ставится или сломан: причину объяснит diarize() ниже
        where = "?"
    where = "на видеокарте" if where.startswith("cuda") else "на процессоре, медленнее" if where == "cpu" else where
    if on_progress:
        on_progress(f"Диаризация (pyannote, {where}): кто и когда говорит ...")
    print(f"  pyannote: диаризация ({where}) ...")
    try:
        try:
            turns = diarize.diarize(wav, device=device, num_speakers=getattr(args, "num_speakers", None))
        except RuntimeError as e:  # нехватка памяти видеокарты (4 GB на длинном видео): повтор на cpu
            if "out of memory" not in str(e).lower() or device == "cpu":
                raise
            print("  pyannote: не хватило памяти видеокарты, повтор на cpu")
            if on_progress:
                on_progress("Диаризация (pyannote, на процессоре: не хватило памяти видеокарты, медленнее) ...")
            diarize.torch_empty_cache()
            turns = diarize.diarize(wav, device="cpu", num_speakers=getattr(args, "num_speakers", None))
    except Exception as e:  # noqa: BLE001 - без диаризации видео всё равно распознаётся, просто одним спикером
        print(f"  ВНИМАНИЕ: диаризация недоступна ({e}); режу по паузам, все реплики получат SPEAKER_00. "
              f"Когда pyannote заработает, перезапустите видео с --force", file=sys.stderr)
        args.diarization_error = str(e)[:300]  # веб-сервер покажет причину в задаче
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": diarize.MODEL, "turns": turns}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  pyannote: спикеров {len({t['speaker'] for t in turns})}, реплик {len(turns)}")
    return turns


def get_xonsaroy(wav: Path, cache: Path, args) -> dict:
    """Xonsaroy без кредитов: в журнал расхода пишутся минуты аудио и число запросов за этот запуск
    (как в дашборде Xonsaroy). Упавший посередине запуск тоже пишется: отправленные куски уже потрачены."""
    turns = get_turns(wav, cache, args)
    print("  Xonsaroy: " + ("распознавание реплик спикеров ..." if turns is not None else "поиск пауз и распознавание кусков ..."))
    last, bill = [0.0], {}
    on_progress = getattr(args, "on_progress", None)  # веб-сервер показывает ход в сообщении задачи

    def progress(done: int, total: int, b: dict) -> None:
        if on_progress:
            on_progress(f"Распознавание: {done}/{total} кусков, отправлено {b.get('audio_seconds', 0) / 60:.1f} мин")
        if time.time() - last[0] > 10 or done == total:
            last[0] = time.time()
            print(f"  Xonsaroy: {done}/{total} кусков")

    def record(event: str) -> None:
        if bill.get("requests"):
            usage.record(cache.parent.parent, wav.stem, event, bill["audio_seconds"], None, wav.name,
                         provider="xonsaroy", requests=bill["requests"])

    try:
        result = xonsaroy.transcribe(wav, min_dur=args.min_dur, max_dur=args.max_dur, billing=bill,
                                     state_file=cache.parent / f"{wav.stem}.xonsaroy.partial.json", progress=progress,
                                     turns=turns)
    except xonsaroy.XonsaroyError:
        record("partial")
        raise
    record("charged")
    if getattr(args, "diarization_error", None):
        result["diarization_error"] = args.diarization_error
    print(f"  Xonsaroy: отправлено {bill.get('audio_seconds', 0) / 60:.1f} мин, запросов {bill.get('requests', 0)}, "
          f"распознано кусков {len(result['segments'])}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


FP_STEP = 0.5  # шаг огибающей для отпечатка звука, с


def audio_fingerprint(wav: Path) -> dict:
    """Отпечаток звука: длительность и огибающая средней громкости по 0.5 с (0..255).

    Не зависит от имени файла и контейнера: тот же ролик в MP4 и MKV, перекодированный или
    скачанный заново, даёт почти ту же огибающую, хотя байты файла и wav различаются.
    """
    with wave.open(str(wav), "rb") as r:
        rate, n = r.getframerate(), r.getnframes()
        step = int(rate * FP_STEP)
        # отсчёт от первого звука: разные контейнеры и кодеки добавляют в начало разную тишину
        head = array.array("h", r.readframes(rate * 10))
        r.setpos(next((i for i, v in enumerate(head) if abs(v) > 300), 0))
        peaks = []
        while chunk := r.readframes(step):
            a = array.array("h", chunk)
            peaks.append(sum(map(abs, a)) // len(a) if a else 0)  # средняя громкость окна
    top = max(peaks, default=0) or 1
    return {"dur": round(n / rate, 2), "env": bytes(min(255, p * 255 // top) for p in peaks).hex()}


def _corr(x: list[int], y: list[int]) -> float:
    n = min(len(x), len(y))
    if n < 4:
        return 0.0
    x, y = x[:n], y[:n]
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx, syy = sum((a - mx) ** 2 for a in x), sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:  # тишина или ровный тон: совпадает, только если уровни равны
        return 1.0 if abs(mx - my) < 3 else 0.0
    return sxy / (sxx * syy) ** 0.5


def same_audio(a: dict, b: dict, min_corr: float = 0.97) -> bool:
    """Тот же звук: длительность отличается меньше чем на 1 с, огибающие совпадают со сдвигом до ±1 с."""
    if abs(a["dur"] - b["dur"]) > 1.0:
        return False
    x, y = list(bytes.fromhex(a["env"])), list(bytes.fromhex(b["env"]))
    lags = int(1 / FP_STEP)
    return max(_corr(x[max(0, k):], y[max(0, -k):]) for k in range(-lags, lags + 1)) >= min_corr


def audio_index(out: Path) -> dict:
    """{видео: отпечаток} для уже расшифрованных видео; старые видео дописываются при первом обращении."""
    path = out / "transcripts" / "audio_index.json"
    index = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    changed = False
    for wav in sorted((out / "audio").glob("*.wav")):
        if wav.stem not in index and (out / "transcripts" / f"{wav.stem}.voicelab.json").exists():
            index[wav.stem] = audio_fingerprint(wav)
            changed = True
    if changed:
        save_audio_index(out, index)
    return index


def find_duplicate(out: Path, video: str, fp: dict, index: dict) -> str | None:
    for other, ofp in index.items():
        if other != video and (out / "transcripts" / f"{other}.voicelab.json").exists() and same_audio(fp, ofp):
            return other
    return None


def save_audio_index(out: Path, index: dict) -> None:
    path = out / "transcripts" / "audio_index.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")


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
    created = not wav.exists() or args.force
    if created:
        to_wav(src, wav)
    with wave.open(str(wav), "rb") as r:
        total = r.getnframes() / r.getframerate()

    # тот же звук уже расшифрован под другим именем: не отправляем в STT повторно и не дублируем датасет
    fp, index = audio_fingerprint(wav), audio_index(out)
    owner = find_duplicate(out, video, fp, index)
    cache = out / "transcripts" / f"{video}.voicelab.json"
    if owner:
        if not args.allow_duplicates:
            if created:
                wav.unlink(missing_ok=True)
            print(f"  дубликат: тот же звук, что у {owner}; STT не вызывался")
            usage.record(out, video, "duplicate", total, None, src.name, provider=stt_provider(args))
            return {"video": video, "duplicate_of": owner, "segments": 0, "hours": 0.0,
                    "short": 0, "long": 0, "overlap_dropped": 0}
        if not cache.exists():  # дубль разрешён: берём готовую расшифровку, а не платим ещё раз
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes((out / "transcripts" / f"{owner}.voicelab.json").read_bytes())

    result = get_transcription(wav, cache, args)
    if index.get(video) != fp:
        index[video] = fp
        save_audio_index(out, index)
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
            notes = f"длиннее {args.max_dur} с: STT не вернул слова для разбиения"
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
    return {"video": video, "segments": len(rows), "hours": sum(r["duration"] for r in rows) / 3600,
            "speakers": len({r["speaker"] for r in rows}), "diarization": result.get("diarization"),
            "diarization_error": result.get("diarization_error"), **stats}


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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", default="data/videos", help="папка с MP4 (или один файл)")
    ap.add_argument("--out", default="data", help="куда сохранять аудио, сегменты и расшифровки")
    ap.add_argument("--db", default=None, help="путь к SQLite (по умолчанию <out>/speech_lab.db)")
    ap.add_argument("--env", default=str(ROOT / ".env"), help="файл с ключами (XONSAROY_API_KEY, HF_TOKEN, VOICELAB_API_KEY)")
    ap.add_argument("--provider", choices=PROVIDERS, default=None,
                    help="сервис STT (по умолчанию STT_PROVIDER из .env, иначе xonsaroy)")
    ap.add_argument("--language", default="uz")
    ap.add_argument("--min-dur", type=float, default=1.5)
    ap.add_argument("--max-dur", type=float, default=15.0)
    ap.add_argument("--max-gap", type=float, default=0.5, help="макс. пауза при склейке реплик, с")
    ap.add_argument("--keep-overlap", action="store_true", help="сохранять наложения со статусом overlap")
    ap.add_argument("--no-speakers", action="store_true", help="не просить VoiceLab разделять спикеров (только voicelab)")
    ap.add_argument("--no-diarization", action="store_true",
                    help="не делить спикеров через pyannote, резать по паузам (только xonsaroy)")
    ap.add_argument("--device", default=None,
                    help="устройство для pyannote: auto (cuda, если есть), cuda или cpu; по умолчанию DIARIZATION_DEVICE из .env")
    ap.add_argument("--num-speakers", type=int, default=None,
                    help="точное число спикеров в видео, если известно (подсказка pyannote)")
    ap.add_argument("--upload-format", choices=["wav", "flac", "mp3"], default="wav",
                    help="в каком формате отправлять аудио в VoiceLab (только voicelab)")
    ap.add_argument("--force", action="store_true", help="заново конвертировать и транскрибировать")
    ap.add_argument("--allow-duplicates", action="store_true",
                    help="обрабатывать видео с уже расшифрованным звуком (расшифровка берётся из кэша)")
    return ap


def main() -> None:
    args = build_parser().parse_args()

    voicelab.load_env(args.env)
    src = Path(args.videos)
    files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.suffix.lower() in MEDIA_EXT)
    if not files:
        sys.exit(f"Нет видео в {src}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    conn = db.connect(args.db or out / "speech_lab.db")

    summary, seen = [], len(usage.load(out))
    for f in files:
        try:
            summary.append(process(f, out, conn, args))
        except (*STTError, MediaError, subprocess.CalledProcessError) as e:
            print(f"  ОШИБКА: {e}", file=sys.stderr)
    dups = [s for s in summary if s.get("duplicate_of")]
    for s in dups:
        print(f"Пропущен дубликат: {s['video']} = {s['duplicate_of']}")
    n = write_manifest(conn, out)
    hours = sum(s["hours"] for s in summary)
    print(f"\nГотово: видео {len(summary)}/{len(files)}, сегментов в этом запуске "
          f"{sum(s['segments'] for s in summary)} ({hours:.2f} ч), всего в manifest.jsonl: {n}")
    run = usage.load(out)[seen:]
    sent = [r for r in run if r["event"] == "charged"]
    print(f"STT в этом запуске: файлов {len(sent)}, {sum(r['duration_s'] for r in sent) / 60:.1f} мин аудио, "
          f"кредитов VoiceLab {sum(r['credits'] or 0 for r in sent)}; всего: {usage.fmt(usage.summary(out)['total'])}")


if __name__ == "__main__":
    main()
