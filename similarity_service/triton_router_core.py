from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from .audio import PREPROCESS_VERSION, AudioLoader, LoadedAudio, file_signature
from .buckets import BucketPolicy, BucketSpec


BucketEmbedFn = Callable[[BucketSpec, torch.Tensor, torch.Tensor], torch.Tensor]
EmbeddingCacheKey = tuple[str, int, int, str, str, str]


@dataclass(frozen=True, slots=True)
class PathEmbeddingExecutionResult:
    embeddings: list[torch.Tensor | None]
    bucket_seconds: list[int | None]
    errors: list[Exception | None]


class PathEmbeddingCache:
    def __init__(self, max_items: int = 4096):
        self.max_items = max(0, int(max_items))
        self._cache: OrderedDict[EmbeddingCacheKey, tuple[torch.Tensor, int]] = OrderedDict()

    def get(self, key: EmbeddingCacheKey) -> tuple[torch.Tensor, int] | None:
        if self.max_items <= 0 or key not in self._cache:
            return None
        value = self._cache.pop(key)
        self._cache[key] = value
        embedding, bucket_seconds = value
        return embedding.clone(), bucket_seconds

    def put(self, key: EmbeddingCacheKey, embedding: torch.Tensor, bucket_seconds: int) -> None:
        if self.max_items <= 0:
            return
        if key in self._cache:
            self._cache.pop(key)
        self._cache[key] = (embedding.detach().cpu().clone(), int(bucket_seconds))
        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)


class PathEmbeddingRouterCore:
    def __init__(
        self,
        audio_loader: AudioLoader,
        bucket_policy: BucketPolicy,
        normalize_embeddings: bool = True,
        embedding_cache_items: int = 4096,
        cache_namespace: str = "wavlm_ecapa",
    ):
        self.audio_loader = audio_loader
        self.bucket_policy = bucket_policy
        self.normalize_embeddings = normalize_embeddings
        self.cache_namespace = cache_namespace
        self.embedding_cache = PathEmbeddingCache(embedding_cache_items)

    def _cache_key_for_resolved_path(self, path) -> EmbeddingCacheKey:
        signature = file_signature(path)
        return (
            signature[0],
            signature[1],
            signature[2],
            PREPROCESS_VERSION,
            self.bucket_policy.version,
            self.cache_namespace,
        )

    def embed_paths(
        self,
        audio_paths: Sequence[str],
        bucket_embed_fn: BucketEmbedFn,
    ) -> PathEmbeddingExecutionResult:
        embeddings: list[torch.Tensor | None] = [None] * len(audio_paths)
        bucket_seconds: list[int | None] = [None] * len(audio_paths)
        errors: list[Exception | None] = [None] * len(audio_paths)

        loaded_audios: list[LoadedAudio | None] = [None] * len(audio_paths)
        groups: dict[BucketSpec, list[int]] = {}
        for idx, path in enumerate(audio_paths):
            try:
                resolved_path = self.audio_loader.resolve_path(path)
                cache_key = self._cache_key_for_resolved_path(resolved_path)
                cached = self.embedding_cache.get(cache_key)
                if cached is not None:
                    embedding, cached_bucket_seconds = cached
                    embeddings[idx] = embedding
                    bucket_seconds[idx] = cached_bucket_seconds
                    continue
                loaded = self.audio_loader.load(path)
                loaded_audios[idx] = loaded
                bucket = self.bucket_policy.bucket_for_num_samples(loaded.num_samples)
                bucket_seconds[idx] = bucket.seconds
                groups.setdefault(bucket, []).append(idx)
            except Exception as exc:  # pragma: no cover - exercised via result handling
                errors[idx] = exc

        for bucket, indices in groups.items():
            for chunk_start in range(0, len(indices), bucket.max_batch):
                chunk_indices = indices[chunk_start : chunk_start + bucket.max_batch]
                waveforms = [
                    loaded_audios[idx].waveform
                    for idx in chunk_indices
                    if loaded_audios[idx] is not None
                ]
                x, lengths = self.bucket_policy.build_padded_batch(waveforms, bucket=bucket)
                raw_embeddings = bucket_embed_fn(bucket, x, lengths)
                raw_embeddings = raw_embeddings.detach().cpu().to(dtype=torch.float32)
                if self.normalize_embeddings:
                    raw_embeddings = F.normalize(raw_embeddings, dim=-1)
                for idx, embedding in zip(chunk_indices, raw_embeddings):
                    embeddings[idx] = embedding
                    loaded = loaded_audios[idx]
                    if loaded is not None:
                        cache_key = (
                            loaded.signature[0],
                            loaded.signature[1],
                            loaded.signature[2],
                            PREPROCESS_VERSION,
                            self.bucket_policy.version,
                            self.cache_namespace,
                        )
                        self.embedding_cache.put(cache_key, embedding, bucket.seconds)

        return PathEmbeddingExecutionResult(
            embeddings=embeddings,
            bucket_seconds=bucket_seconds,
            errors=errors,
        )
