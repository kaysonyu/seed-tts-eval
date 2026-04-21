from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .audio import PREPROCESS_VERSION, AudioLoader, LoadedAudio
from .backends import EmbeddingBackend
from .buckets import BucketPolicy, BucketSpec


def make_reference_cache_key(
    audio: LoadedAudio,
    backend_fingerprint: str,
    bucket_policy_version: str,
) -> tuple[str, int, int, str, str, str]:
    return (
        audio.signature[0],
        audio.signature[1],
        audio.signature[2],
        backend_fingerprint,
        PREPROCESS_VERSION,
        bucket_policy_version,
    )


class ReferenceEmbeddingCache:
    def __init__(self, max_items: int = 10000):
        self.max_items = max_items
        self._cache: OrderedDict[tuple[str, int, int, str, str, str], torch.Tensor] = OrderedDict()

    def get(self, key: tuple[str, int, int, str, str, str]) -> torch.Tensor | None:
        if key not in self._cache:
            return None
        value = self._cache.pop(key)
        self._cache[key] = value
        return value.clone()

    def put(self, key: tuple[str, int, int, str, str, str], value: torch.Tensor) -> None:
        if key in self._cache:
            self._cache.pop(key)
        self._cache[key] = value.detach().cpu().clone()
        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)


@dataclass(frozen=True, slots=True)
class ScoreResult:
    request_id: str | None
    backend: str
    ref_cache_hit: bool
    ref_bucket_seconds: int
    generated_bucket_seconds: list[int]
    scores: list[float]


class SimilarityService:
    def __init__(
        self,
        audio_loader: AudioLoader,
        bucket_policy: BucketPolicy,
        backend: EmbeddingBackend,
        ref_cache: ReferenceEmbeddingCache | None = None,
    ):
        self.audio_loader = audio_loader
        self.bucket_policy = bucket_policy
        self.backend = backend
        self.ref_cache = ref_cache or ReferenceEmbeddingCache()

    @property
    def is_ready(self) -> bool:
        return self.backend.is_ready

    def load_audios(self, paths: Sequence[str]) -> list[LoadedAudio]:
        return [self.audio_loader.load(path) for path in paths]

    def _embed_loaded_audio(self, audio: LoadedAudio) -> tuple[torch.Tensor, BucketSpec]:
        bucket = self.bucket_policy.bucket_for_num_samples(audio.num_samples)
        x, lengths = self.bucket_policy.build_padded_batch([audio.waveform], bucket=bucket)
        emb = self.backend.embed_padded_batch(x, lengths)[0]
        return emb, bucket

    def _embed_loaded_audios(self, audios: Sequence[LoadedAudio]) -> tuple[list[torch.Tensor], list[BucketSpec]]:
        if not audios:
            return [], []
        outputs: list[torch.Tensor | None] = [None] * len(audios)
        bucket_by_index: list[BucketSpec | None] = [None] * len(audios)
        groups = self.bucket_policy.group_indices_by_bucket([audio.num_samples for audio in audios])
        for bucket, indices in groups.items():
            for chunk_start in range(0, len(indices), bucket.max_batch):
                chunk_indices = indices[chunk_start : chunk_start + bucket.max_batch]
                waveforms = [audios[idx].waveform for idx in chunk_indices]
                x, lengths = self.bucket_policy.build_padded_batch(waveforms, bucket=bucket)
                emb = self.backend.embed_padded_batch(x, lengths)
                for idx, emb_vec in zip(chunk_indices, emb):
                    outputs[idx] = emb_vec
                    bucket_by_index[idx] = bucket
        return [tensor for tensor in outputs if tensor is not None], [bucket for bucket in bucket_by_index if bucket is not None]

    def _get_or_compute_reference_embedding(self, audio: LoadedAudio) -> tuple[torch.Tensor, BucketSpec, bool]:
        key = make_reference_cache_key(audio, self.backend.fingerprint, self.bucket_policy.version)
        cached = self.ref_cache.get(key)
        if cached is not None:
            bucket = self.bucket_policy.bucket_for_num_samples(audio.num_samples)
            return cached, bucket, True
        emb, bucket = self._embed_loaded_audio(audio)
        self.ref_cache.put(key, emb)
        return emb, bucket, False

    def score_rollouts(
        self,
        ref_audio_path: str,
        generated_audio_paths: Sequence[str],
        request_id: str | None = None,
    ) -> ScoreResult:
        if not generated_audio_paths:
            raise ValueError("generated_audio_paths must not be empty")
        if len(generated_audio_paths) > 16:
            raise ValueError("generated_audio_paths must contain at most 16 paths")

        ref_audio = self.audio_loader.load(ref_audio_path)
        gen_audios = self.load_audios(generated_audio_paths)

        ref_emb, ref_bucket, ref_cache_hit = self._get_or_compute_reference_embedding(ref_audio)
        gen_embs, gen_buckets = self._embed_loaded_audios(gen_audios)
        if len(gen_embs) != len(gen_audios):
            raise RuntimeError("Internal embedding alignment failure")

        gen_emb_batch = torch.stack(gen_embs, dim=0)
        ref_emb_batch = ref_emb.unsqueeze(0).expand(gen_emb_batch.size(0), -1)
        scores = F.cosine_similarity(gen_emb_batch, ref_emb_batch, dim=-1)
        return ScoreResult(
            request_id=request_id,
            backend=self.backend.name,
            ref_cache_hit=ref_cache_hit,
            ref_bucket_seconds=ref_bucket.seconds,
            generated_bucket_seconds=[bucket.seconds for bucket in gen_buckets],
            scores=[float(score.item()) for score in scores],
        )
