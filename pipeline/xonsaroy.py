"""Клиент Xonsaroy STT (GigaAM-Multilingual) по документации STT-API-qollanma.

POST {XONSAROY_BASE_URL}/transcriptions  (multipart: file; Bearer xs_stt_...) -> {text, audio_seconds, ms}

API возвращает только сплошной текст: без таймкодов и без спикеров. Поэтому границы реплик ищутся
локально (VAD по громкости в wav 16 kHz), каждый кусок 1.5–15 с отправляется отдельно, а ответ
собирается в том же виде, что у VoiceLab: segments[{start_ms, end_ms, text, speaker}], и дальше
pipeline.prepare работает как раньше. Если передана диаризация pyannote (turns), куски режутся внутри
реплик каждого спикера и получают его метку; иначе все куски одного видео получают спикера "0", и разделить
людей редактор может в студии (переназначение спикера).

Ограничения API: 2 одновременных запроса (иначе 429 + Retry-After), 400 MB, 2 часа на файл.
Повтор только на 429 (ждём Retry-After), 502/503 и обрыв связи; остальные ошибки сразу.
Готовые куски пишутся в state_file, поэтому после сбоя уже распознанные куски заново не отправляются.
"""
import array
import json
import math
import os
import tempfile
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

DEFAULT_BASE_URL = "https://xonsaroy.adiz.uz/api/v1"
MAX_CONCURRENT = 2      # лимит сервера на один ключ
FRAME = 0.03            # окно VAD, с
MIN_SILENCE = 0.3       # пауза короче этого не разрывает речь, с
MIN_SPEECH = 0.2        # всплеск короче этого считается шумом, с
JOIN_GAP = 1.5          # паузы до JOIN_GAP не обязательная граница: кусок режется в них, только если он длиннее max_dur, с
EDGE = 0.1              # запас тишины по краям куска, отправляемого в API, с


class XonsaroyError(RuntimeError):
    pass


def get_api_key() -> str:
    key = os.environ.get("XONSAROY_API_KEY", "").strip()
    if not key:
        raise XonsaroyError("XONSAROY_API_KEY не задан: добавьте его в .env")
    return key


def base_url() -> str:
    """Читается при каждом запросе: .env подхватывается уже после импорта модуля."""
    return (os.environ.get("XONSAROY_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def _error(resp: requests.Response) -> XonsaroyError:
    try:
        err = resp.json().get("error") or {}
        code, msg = err.get("code", "unknown_error"), err.get("message", resp.text)
    except ValueError:
        code, msg = "unknown_error", resp.text
    rid = resp.headers.get("X-Request-Id")
    return XonsaroyError(f"Xonsaroy HTTP {resp.status_code} [{code}]: {msg}" + (f" (X-Request-Id {rid})" if rid else ""))


def post_file(path: Path, attempts: int = 5, max_rate_limited: float = 600) -> dict:
    """POST /transcriptions с повторами: 429 ждёт Retry-After (до max_rate_limited с суммарно),
    502/503 и обрыв связи повторяются до attempts раз с паузой 1, 2, 4... с."""
    fails, waited, last = 0, 0.0, None
    while True:
        try:
            with path.open("rb") as f:
                resp = requests.post(f"{base_url()}/transcriptions", headers={"Authorization": f"Bearer {get_api_key()}"},
                                     files={"file": (path.name, f, "audio/wav")}, timeout=960)
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as e:
            resp, last = None, e
        if resp is not None and resp.status_code == 429:
            try:
                delay = float(resp.headers.get("Retry-After") or 1)
            except ValueError:
                delay = 1.0
            if waited >= max_rate_limited:
                raise XonsaroyError("Xonsaroy занят (429) слишком долго: попробуйте позже")
            time.sleep(delay)
            waited += delay
            continue
        if resp is None or resp.status_code in (502, 503):
            fails += 1
            if fails >= attempts:
                raise (_error(resp) if resp is not None else XonsaroyError(f"Xonsaroy недоступен: {last}"))
            time.sleep(2 ** (fails - 1))
            continue
        if not resp.ok:
            raise _error(resp)
        return resp.json()


def _energies(wav: Path) -> tuple[list[int], int]:
    """Средняя громкость (mean |x|) по окнам FRAME и частота дискретизации."""
    with wave.open(str(wav), "rb") as r:
        rate = r.getframerate()
        step = int(rate * FRAME)
        out = []
        while chunk := r.readframes(step):
            a = array.array("h", chunk)
            out.append(sum(map(abs, a)) // len(a) if a else 0)
    return out, rate


def _smooth(energy: list[int], a: int, b: int, w: int = 17) -> list[float]:
    """Скользящее среднее громкости (~0.5 с) на [a, b): короткая тихая точка внутри слова или фразы не перебивает настоящую паузу."""
    h = w // 2
    pre = [0]
    for v in energy[a:b]:
        pre.append(pre[-1] + v)
    n = b - a
    return [(pre[min(n, k + h + 1)] - pre[max(0, k - h)]) / (min(n, k + h + 1) - max(0, k - h)) for k in range(n)]


def speech_regions(energy: list[int]) -> list[tuple[int, int]]:
    """Отрезки речи в номерах окон [start, end). Порог от уровня шума (10-й перцентиль).

    Решение принимается по громкости, сглаженной на ~0.15 с: в шумной записи отдельные громкие окна
    шума не разрывают паузу между фразами, а тихие окна внутри слова не разрывают слово.
    """
    if not energy:
        return []
    sm = _smooth(energy, 0, len(energy), 5)
    srt = sorted(sm)
    floor, median = srt[len(srt) // 10], srt[len(srt) // 2]
    thr = max(100, min(floor * 2, median))
    flags = [e >= thr for e in sm]
    regions, i, n = [], 0, len(flags)
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = i
        while j < n and flags[j]:
            j += 1
        if regions and (i - regions[-1][1]) * FRAME < MIN_SILENCE:
            regions[-1] = (regions[-1][0], j)  # короткая пауза внутри фразы
        else:
            regions.append((i, j))
        i = j
    return [(a, b) for a, b in regions if (b - a) * FRAME >= MIN_SPEECH]


def _best_split(regions: list[tuple[int, int]], energy: list[int], max_len: int, min_len: int) -> list[tuple[int, int]]:
    """Делит группу отрезков речи (паузы между ними до JOIN_GAP) на куски не длиннее max_len.

    Группа, которая влезает в max_len, остаётся одним куском, даже с паузами внутри: фраза не рвётся.
    Иначе режем в самой длинной паузе группы, где обе части не короче min_len (паузу, после которой
    остаётся обрывок, берём только если других нет), и повторяем для каждой половины. Сплошная речь
    без пауз режется в самой тихой точке сглаженной громкости, подальше от краёв.
    """
    a, b = regions[0][0], regions[-1][1]
    if b - a <= max_len:
        return [(a, b)]
    if len(regions) > 1:
        def score(i: int) -> tuple:
            left, right = regions[i][1] - a, b - regions[i + 1][0]
            return (left >= min_len and right >= min_len, regions[i + 1][0] - regions[i][1])
        i = max(range(len(regions) - 1), key=score)
        return (_best_split(regions[:i + 1], energy, max_len, min_len)
                + _best_split(regions[i + 1:], energy, max_len, min_len))
    # разрез, после которого обе части не длиннее max_len (одного разреза хватит), иначе в первом max_len
    lo, hi = max(a + min_len, b - max_len), min(b - min_len, a + max_len)
    if lo >= hi:
        lo, hi = a + max_len // 2, a + max_len
    sm = _smooth(energy, lo, hi)
    cut = lo + min(range(len(sm)), key=lambda k: sm[k])
    return (_best_split([(a, cut)], energy, max_len, min_len)
            + _best_split([(cut, b)], energy, max_len, min_len))


def plan_chunks(energy: list[int], min_dur: float, max_dur: float) -> list[tuple[float, float]]:
    """Куски для отправки в API, секунды [start, end): длиной не больше max_dur, короче min_dur отбрасываются.

    Паузы длиннее JOIN_GAP всегда граница куска; более короткие паузы режут кусок, только если без этого
    он длиннее max_dur, и тогда выбирается самая длинная из них (см. _best_split).
    """
    max_len = int((max_dur - 2 * EDGE) / FRAME)
    min_len = int(min_dur / FRAME)
    groups: list[list[tuple[int, int]]] = []
    for r in speech_regions(energy):
        if groups and (r[0] - groups[-1][-1][1]) * FRAME <= JOIN_GAP:
            groups[-1].append(r)
        else:
            groups.append([r])
    chunks = [list(c) for g in groups for c in _best_split(g, energy, max_len, min_len)]
    # запас EDGE по краям, но не больше половины паузы до соседа: на разрезе внутри речи куски не пересекаются
    bounds = [0, *(x for c in chunks for x in c), len(energy)]
    out = []
    for i, (a, b) in enumerate(chunks):
        prev_end, next_start = bounds[2 * i], bounds[2 * i + 3]
        start = max(a * FRAME - EDGE, (a + prev_end) / 2 * FRAME if i else 0.0)
        end = min(b * FRAME + EDGE, (b + next_start) / 2 * FRAME)
        if end - start >= min_dur:
            out.append((round(start, 3), round(end, 3)))
    return out


def plan_turn_chunks(energy: list[int], turns: list[dict], min_dur: float,
                     max_dur: float) -> list[tuple[float, float, str]]:
    """Куски по репликам спикеров из диаризации: (start, end, speaker), каждый целиком внутри реплики одного человека.

    Подряд идущие реплики одного спикера с паузой до JOIN_GAP, между которыми никто другой не говорит, считаются
    одной. Внутри реплики куски режутся так же, как без диаризации (паузы по громкости, _best_split), а запас EDGE
    по краям не заходит в реплику другого спикера. Реплики разных спикеров могут пересекаться: такие куски
    prepare помечает как наложение.
    """
    max_len = int((max_dur - 2 * EDGE) / FRAME)
    min_len = int(min_dur / FRAME)
    n, total = len(energy), len(energy) * FRAME
    regions = speech_regions(energy)
    merged: list[dict] = []
    for t in sorted(turns, key=lambda t: t["start"]):
        p = next((m for m in reversed(merged) if m["speaker"] == t["speaker"]), None)
        if p and t["start"] - p["end"] <= JOIN_GAP and not any(
                o["speaker"] != t["speaker"] and o["end"] > p["end"] and o["start"] < t["start"] for o in turns):
            p["end"] = max(p["end"], t["end"])
        else:
            merged.append(dict(t))
    out = []
    for m in merged:
        a, b = int(m["start"] / FRAME), min(n, math.ceil(m["end"] / FRAME))
        if b <= a:
            continue
        # речь внутри реплики по VAD; тихую речь, которую VAD не заметил, берём всей репликой
        rs = [(max(a, x), min(b, y)) for x, y in regions if x < b and y > a] or [(a, b)]
        groups: list[list[tuple[int, int]]] = []
        for r in rs:
            if groups and (r[0] - groups[-1][-1][1]) * FRAME <= JOIN_GAP:
                groups[-1].append(r)
            else:
                groups.append([r])
        chunks = [c for g in groups for c in _best_split(g, energy, max_len, min_len)]
        others = [o for o in turns if o["speaker"] != m["speaker"]]
        lo = max([o["end"] for o in others if m["start"] - EDGE < o["end"] <= m["start"]], default=m["start"] - EDGE)
        hi = min([o["start"] for o in others if m["end"] <= o["start"] < m["end"] + EDGE], default=m["end"] + EDGE)
        for i, (x, y) in enumerate(chunks):
            start = max(x * FRAME - EDGE, (x + chunks[i - 1][1]) / 2 * FRAME if i else lo, 0.0)
            end = min(y * FRAME + EDGE, (y + chunks[i + 1][0]) / 2 * FRAME if i + 1 < len(chunks) else hi, total)
            if end - start >= min_dur:
                out.append((round(start, 3), round(end, 3), m["speaker"]))
    return sorted(out)


def _cut(src: Path, dst: Path, start: float, end: float) -> None:
    with wave.open(str(src), "rb") as r:
        sr = r.getframerate()
        r.setpos(int(start * sr))
        frames = r.readframes(int((end - start) * sr))
        params = r.getparams()
    with wave.open(str(dst), "wb") as w:
        w.setparams(params)
        w.writeframes(frames)


def transcribe(wav: str | Path, min_dur: float = 1.5, max_dur: float = 15.0,
               state_file: str | Path | None = None, concurrency: int | None = None,
               progress=None, billing: dict | None = None, turns: list[dict] | None = None,
               post=None, provider: str = "xonsaroy") -> dict:
    """Режет wav по паузам (или по репликам спикеров turns из pyannote) и распознаёт каждый кусок
    (не больше concurrency запросов одновременно).

    Возвращает {provider, transcript, duration_ms, audio_seconds, segments[{start_ms, end_ms, text, speaker}]}.
    Куски, где ничего не распознано (пустой text), в segments не попадают.
    billing (если передан) по ходу работы заполняется: requests и audio_seconds (по ответам API, как их
    считает дашборд Xonsaroy) только за этот запуск, без кусков, распознанных в прошлых запусках.
    """
    wav = Path(wav)
    energy, _ = _energies(wav)
    if turns is None:
        chunks = [(a, b, "0") for a, b in plan_chunks(energy, min_dur, max_dur)]
    else:
        chunks = plan_turn_chunks(energy, turns, min_dur, max_dur)
    opts = {"file": wav.name, "min_dur": min_dur, "max_dur": max_dur, "chunks": len(chunks)}
    if turns is not None:
        opts["turns"] = len(turns)

    def key(c: tuple[float, float, str]) -> str:  # без диаризации ключ прежний: старые partial-файлы подходят
        return f"{c[0]}-{c[1]}" if turns is None else f"{c[0]}-{c[1]}-{c[2]}"
    state_path = Path(state_file) if state_file else None
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path and state_path.exists() else {}
    if state.get("opts") != opts:
        state = {"opts": opts, "done": {}}
    lock = threading.Lock()
    todo = [c for c in chunks if key(c) not in state["done"]]
    workers = max(1, min(concurrency or int(os.environ.get("XONSAROY_MAX_CONCURRENT", MAX_CONCURRENT)), MAX_CONCURRENT))
    post = post or post_file  # другой распознаватель кусков (локальный Whisper): тот же ответ {text, audio_seconds}

    with tempfile.TemporaryDirectory() as tmp:
        def work(c: tuple[float, float, str]) -> None:
            part = Path(tmp) / f"{wav.stem}_{int(c[0] * 1000):08d}_{int(c[1] * 1000):08d}_{c[2]}.wav"
            _cut(wav, part, c[0], c[1])
            res = post(part)
            part.unlink(missing_ok=True)
            with lock:
                if billing is not None:
                    billing["requests"] = billing.get("requests", 0) + 1
                    billing["audio_seconds"] = billing.get("audio_seconds", 0.0) + (res.get("audio_seconds") or c[1] - c[0])
                state["done"][key(c)] = {"text": (res.get("text") or "").strip(),
                                                   "audio_seconds": res.get("audio_seconds")}
                if state_path:
                    state_path.parent.mkdir(parents=True, exist_ok=True)
                    state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
                if progress:
                    progress(len(state["done"]), len(chunks), billing)

        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            for fut in [pool.submit(work, c) for c in todo]:
                fut.result()  # первая ошибка прерывает распознавание; готовые куски уже в state_file
        finally:
            pool.shutdown(wait=True, cancel_futures=True)  # после ошибки остальные куски не отправляем

    segments = []
    for c in chunks:
        text = state["done"][key(c)]["text"]
        if text:
            segments.append({"start_ms": int(c[0] * 1000), "end_ms": int(c[1] * 1000), "text": text, "speaker": c[2]})
    if state_path:
        state_path.unlink(missing_ok=True)
    return {"provider": provider, "diarization": "vad" if turns is None else "pyannote",
            "transcript": " ".join(s["text"] for s in segments),
            "duration_ms": int(len(energy) * FRAME * 1000),
            "audio_seconds": round(sum(c[1] - c[0] for c in chunks), 3), "segments": segments}
