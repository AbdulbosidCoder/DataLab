"""Локальная диаризация pyannote/speaker-diarization-3.1: кто и когда говорит.

Xonsaroy возвращает только текст, поэтому людей делит pyannote на своём компьютере, а в Xonsaroy уходит
каждая реплика спикера отдельно (см. xonsaroy.transcribe(turns=...)).

Нужно: pip install -r requirements-diarization.txt (torch + pyannote.audio), HF_TOKEN в .env и принятые условия
на https://hf.co/pyannote/speaker-diarization-3.1 и https://hf.co/pyannote/segmentation-3.0.
Устройство: cuda, если torch видит видеокарту (GTX 1650 4 GB хватает), иначе cpu (в несколько раз медленнее).
Если pyannote не установлен или нет токена, бросается DiarizationUnavailable, и prepare режет по паузам, как раньше.
"""
import array
import os
import wave
from pathlib import Path

MODEL = "pyannote/speaker-diarization-3.1"
_pipelines: dict = {}  # модель грузится один раз на процесс (веб-сервер обрабатывает много видео)


class DiarizationUnavailable(RuntimeError):
    pass


def hf_token() -> str:
    return (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or "").strip()


def pick_device(device: str | None = None) -> str:
    """auto (по умолчанию): cuda, если доступна, иначе cpu. DIARIZATION_DEVICE в .env или --device."""
    import torch
    d = (device or os.environ.get("DIARIZATION_DEVICE") or "auto").strip().lower()
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if d.startswith("cuda") and not torch.cuda.is_available():
        print("  pyannote: CUDA недоступна (torch без CUDA или нет драйвера), работаю на cpu")
        return "cpu"
    return d


def load_pipeline(device: str | None = None):
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as e:
        raise DiarizationUnavailable(f"pyannote.audio не установлен ({e}): pip install -r requirements-diarization.txt")
    token = hf_token()
    if not token:
        raise DiarizationUnavailable("HF_TOKEN не задан: добавьте токен Hugging Face в .env")
    dev = pick_device(device)
    # torch>=2.6 по умолчанию грузит чекпойнты с weights_only=True, а чекпойнты pyannote 3.x так не читаются
    # (UnpicklingError). Модель официальная, с Hugging Face, поэтому возвращаем прежнее поведение torch.load.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    if dev not in _pipelines:
        try:
            pipe = Pipeline.from_pretrained(MODEL, use_auth_token=token)
        except TypeError:  # pyannote.audio 4.x переименовал аргумент
            pipe = Pipeline.from_pretrained(MODEL, token=token)
        except Exception as e:  # нет сети, не приняты условия модели, неверный токен
            raise DiarizationUnavailable(f"не удалось загрузить {MODEL}: {e}")
        if pipe is None:  # 3.x возвращает None, если доступ к модели закрыт
            raise DiarizationUnavailable(f"нет доступа к {MODEL}: примите условия модели на Hugging Face для этого токена")
        pipe.to(torch.device(dev))
        _pipelines[dev] = pipe
        print(f"  pyannote: {MODEL} на {dev}")
    return _pipelines[dev]


def _waveform(wav: Path):
    """wav 16-bit mono -> тензор (1, n) float32: без torchaudio/ffmpeg-бэкенда, который часто ломается на Windows."""
    import torch
    with wave.open(str(wav), "rb") as r:
        rate = r.getframerate()
        a = array.array("h", r.readframes(r.getnframes()))
    return {"waveform": torch.tensor(a, dtype=torch.float32).div_(32768.0).unsqueeze(0), "sample_rate": rate}


def diarize(wav: str | Path, device: str | None = None, num_speakers: int | None = None) -> list[dict]:
    """[{start, end, speaker}] в секундах, по времени. speaker как у pyannote (SPEAKER_00, ...), реплики разных
    спикеров могут пересекаться (наложения)."""
    pipe = load_pipeline(device)
    out = pipe(_waveform(Path(wav)), **({"num_speakers": num_speakers} if num_speakers else {}))
    ann = getattr(out, "speaker_diarization", out)  # 4.x возвращает объект с полем speaker_diarization
    turns = [{"start": round(t.start, 3), "end": round(t.end, 3), "speaker": str(spk)}
             for t, _, spk in ann.itertracks(yield_label=True) if t.end > t.start]
    return sorted(turns, key=lambda t: (t["start"], t["end"]))


def torch_empty_cache() -> None:
    import torch
    _pipelines.pop("cuda", None)  # модель на видеокарте больше не нужна: освобождаем её память
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
