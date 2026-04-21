from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import soundfile as sf
import torch
from torchaudio.transforms import Resample

from .buckets import TARGET_SAMPLE_RATE

PREPROCESS_VERSION = "soundfile-first-channel-resample16k-v2"


class PathAccessError(ValueError):
    pass


class AudioDecodingError(ValueError):
    pass


class AudioTooLongError(ValueError):
    pass


def file_signature(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return str(path), int(stat.st_size), int(stat.st_mtime_ns)


@dataclass(frozen=True, slots=True)
class LoadedAudio:
    path: Path
    waveform: torch.Tensor
    sample_rate: int
    signature: tuple[str, int, int]

    @property
    def num_samples(self) -> int:
        return int(self.waveform.numel())


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class AudioLoader:
    def __init__(
        self,
        allowed_roots: Iterable[Path] | None = None,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
        max_samples: int | None = None,
        audio_backend: str = "soundfile",
    ):
        if audio_backend not in {"soundfile", "librosa"}:
            raise ValueError(f"Unsupported audio_backend: {audio_backend}")
        self.allowed_roots = tuple(Path(root).resolve() for root in (allowed_roots or ()))
        self.target_sample_rate = target_sample_rate
        self.max_samples = max_samples
        self.audio_backend = audio_backend
        self._resamplers: dict[tuple[int, int], Resample] = {}

    def resolve_path(self, path_like: str | os.PathLike[str]) -> Path:
        path = Path(path_like)
        if not path.is_absolute():
            raise PathAccessError(f"Expected an absolute path, got {path}")
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Missing audio file: {path}") from exc

        if self.allowed_roots and not any(_is_relative_to(resolved, root) for root in self.allowed_roots):
            roots = ", ".join(str(root) for root in self.allowed_roots)
            raise PathAccessError(f"Path {resolved} is outside allowed roots: {roots}")
        return resolved

    def _get_resampler(self, sample_rate: int) -> Resample:
        key = (int(sample_rate), int(self.target_sample_rate))
        if key not in self._resamplers:
            self._resamplers[key] = Resample(orig_freq=key[0], new_freq=key[1])
        return self._resamplers[key]

    def _decode_with_soundfile(self, path: Path) -> tuple[np.ndarray, int]:
        waveform, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        return np.asarray(waveform[:, 0], dtype=np.float32), int(sample_rate)

    def _decode_with_librosa(self, path: Path) -> tuple[np.ndarray, int]:
        waveform, sample_rate = librosa.load(str(path), sr=None, mono=False)
        if waveform.ndim == 2:
            waveform = waveform[0]
        return np.asarray(waveform, dtype=np.float32), int(sample_rate)

    def _decode(self, path: Path) -> tuple[np.ndarray, int]:
        if self.audio_backend == "librosa":
            return self._decode_with_librosa(path)
        try:
            return self._decode_with_soundfile(path)
        except Exception:
            return self._decode_with_librosa(path)

    def load(self, path_like: str | os.PathLike[str]) -> LoadedAudio:
        path = self.resolve_path(path_like)
        try:
            waveform, sample_rate = self._decode(path)
        except Exception as exc:
            raise AudioDecodingError(f"Failed to decode audio from {path}") from exc

        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim != 1:
            raise AudioDecodingError(f"Expected mono or channel-first audio, got shape {waveform.shape}")
        if waveform.size == 0:
            raise AudioDecodingError(f"Decoded empty audio from {path}")

        wav = torch.from_numpy(np.ascontiguousarray(waveform)).contiguous()
        if sample_rate != self.target_sample_rate:
            wav = self._get_resampler(sample_rate)(wav.unsqueeze(0)).squeeze(0)
            wav = wav.contiguous()

        if self.max_samples is not None and wav.numel() > self.max_samples:
            raise AudioTooLongError(
                f"Audio {path} has {wav.numel()} samples after resampling, "
                f"which exceeds the configured limit {self.max_samples}"
            )

        return LoadedAudio(
            path=path,
            waveform=wav.to(dtype=torch.float32),
            sample_rate=self.target_sample_rate,
            signature=file_signature(path),
        )
