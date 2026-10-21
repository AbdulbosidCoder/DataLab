"""Локальный Whisper (faster-whisper) вместо Xonsaroy: STT_PROVIDER=whisper.

Куски режутся так же, как для Xonsaroy (по репликам pyannote или по паузам, см. xonsaroy.transcribe), только каждый
кусок распознаётся на своей машине, без сети и без лимитов. По умолчанию large-v3-turbo в int8_float16:
около 1.5 ГБ видеопамяти (GTX 1650 4 ГБ хватает с запасом). Без видеокарты int8 на процессоре, медленно.

.env: WHISPER_MODEL (large-v3-turbo | large-v3 | medium ...), WHISPER_COMPUTE (int8_float16 | int8 | float16),
WHISPER_DEVICE (auto | cuda | cpu), WHISPER_BEAM (1 = меньше памяти, 5 = чуть точнее).
Модель скачивается с Hugging Face один раз (в Docker: data/.cache/huggingface).
"""
import os
import wave
from pathlib import Path

DEFAULT_MODEL = "large-v3-turbo"
_model: dict = {}  # модель грузится один раз на процесс


class WhisperError(RuntimeError):
    pass


def _device() -> str:
    d = (os.environ.get("WHISPER_DEVICE") or "auto").strip().lower()
    if d != "auto":
        return d
    try:
        import ctranslate2
        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:  # noqa: BLE001 - нет ctranslate2 или драйвера: работаем на процессоре
        return "cpu"


def load():
    if "m" in _model:
        return _model["m"]
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise WhisperError(f"faster-whisper не установлен ({e}): pip install -r requirements.txt")
    name = os.environ.get("WHISPER_MODEL") or DEFAULT_MODEL
    dev = _device()
    compute = os.environ.get("WHISPER_COMPUTE") or ("int8_float16" if dev.startswith("cuda") else "int8")
    try:
        _model["m"] = WhisperModel(name, device=dev, compute_type=compute)
    except Exception as e:  # нет сети, не хватает памяти видеокарты, нет cuDNN
        raise WhisperError(f"не удалось загрузить Whisper {name} ({dev}, {compute}): {e}")
    print(f"  Whisper: {name} на {dev} ({compute})")
    return _model["m"]


def post_file(path: str | Path) -> dict:
    """Тот же ответ, что у xonsaroy.post_file: {text, audio_seconds}."""
    with wave.open(str(path), "rb") as r:
        dur = r.getnframes() / r.getframerate()
    try:
        segs, _ = load().transcribe(str(path), language="uz", beam_size=int(os.environ.get("WHISPER_BEAM", 1)),
                                    condition_on_previous_text=False, vad_filter=False)
        text = " ".join(s.text.strip() for s in segs)
    except WhisperError:
        raise
    except Exception as e:  # например нехватка видеопамяти на длинном куске
        raise WhisperError(f"Whisper не распознал {Path(path).name}: {e}")
    return {"text": text.strip(), "audio_seconds": round(dur, 3)}
