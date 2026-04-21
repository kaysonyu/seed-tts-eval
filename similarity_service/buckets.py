from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import torch


TARGET_SAMPLE_RATE = 16000


def default_max_batch_for_bucket_seconds(seconds: int) -> int:
    if seconds <= 8:
        return 16
    if seconds <= 16:
        return 8
    if seconds <= 30:
        return 4
    return 2


@dataclass(frozen=True, slots=True)
class BucketSpec:
    seconds: int
    max_batch: int
    sample_rate: int = TARGET_SAMPLE_RATE

    @property
    def num_samples(self) -> int:
        return self.seconds * self.sample_rate

    @property
    def key(self) -> str:
        return f"{self.seconds}s"


DEFAULT_BUCKET_SPECS = tuple(
    BucketSpec(seconds=seconds, max_batch=default_max_batch_for_bucket_seconds(seconds))
    for seconds in (4, 8, 12, 16, 20, 24, 30, 45, 60, 90)
)


class BucketPolicy:
    def __init__(
        self,
        bucket_specs: Sequence[BucketSpec] = DEFAULT_BUCKET_SPECS,
        sample_rate: int = TARGET_SAMPLE_RATE,
    ):
        if not bucket_specs:
            raise ValueError("bucket_specs must not be empty")
        self.sample_rate = sample_rate
        self.bucket_specs = tuple(sorted(bucket_specs, key=lambda spec: spec.num_samples))
        if any(spec.sample_rate != self.sample_rate for spec in self.bucket_specs):
            raise ValueError("All bucket_specs must use the same sample rate as the policy")
        if len({spec.seconds for spec in self.bucket_specs}) != len(self.bucket_specs):
            raise ValueError("bucket_specs must not contain duplicate second values")

    @property
    def max_samples(self) -> int:
        return self.bucket_specs[-1].num_samples

    @property
    def version(self) -> str:
        return "-".join(str(spec.seconds) for spec in self.bucket_specs)

    def bucket_for_num_samples(self, num_samples: int) -> BucketSpec:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        for spec in self.bucket_specs:
            if num_samples <= spec.num_samples:
                return spec
        raise ValueError(
            f"num_samples={num_samples} exceeds the largest supported bucket "
            f"({self.bucket_specs[-1].num_samples})"
        )

    def bucket_for_waveform(self, waveform: torch.Tensor) -> BucketSpec:
        if waveform.dim() != 1:
            raise ValueError(f"Expected 1-D waveform, got shape {tuple(waveform.shape)}")
        return self.bucket_for_num_samples(int(waveform.numel()))

    def group_indices_by_bucket(self, lengths: Sequence[int]) -> Dict[BucketSpec, list[int]]:
        groups: Dict[BucketSpec, list[int]] = {}
        for idx, length in enumerate(lengths):
            bucket = self.bucket_for_num_samples(int(length))
            groups.setdefault(bucket, []).append(idx)
        return groups

    def build_padded_batch(
        self,
        waveforms: Sequence[torch.Tensor],
        bucket: BucketSpec | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not waveforms:
            raise ValueError("waveforms must not be empty")
        if bucket is None:
            bucket = self.bucket_for_waveform(waveforms[0])
        if len(waveforms) > bucket.max_batch:
            raise ValueError(
                f"Bucket {bucket.key} allows at most {bucket.max_batch} waveforms, "
                f"got {len(waveforms)}"
            )
        x = torch.zeros((len(waveforms), bucket.num_samples), dtype=torch.float32)
        lengths = torch.empty(len(waveforms), dtype=torch.long)
        for idx, waveform in enumerate(waveforms):
            if waveform.dim() != 1:
                raise ValueError(f"Expected 1-D waveform, got shape {tuple(waveform.shape)}")
            waveform = waveform.detach().cpu().to(dtype=torch.float32)
            length = int(waveform.numel())
            if length > bucket.num_samples:
                raise ValueError(
                    f"Waveform length {length} exceeds bucket {bucket.key} "
                    f"({bucket.num_samples})"
                )
            x[idx, :length] = waveform
            lengths[idx] = length
        return x, lengths

    def bucket_specs_from_seconds(self, seconds: Iterable[int]) -> tuple[BucketSpec, ...]:
        spec_by_seconds = {spec.seconds: spec for spec in self.bucket_specs}
        selected = []
        for second in seconds:
            if second in spec_by_seconds:
                selected.append(spec_by_seconds[second])
            else:
                selected.append(
                    BucketSpec(
                        seconds=second,
                        max_batch=default_max_batch_for_bucket_seconds(second),
                        sample_rate=self.sample_rate,
                    )
                )
        return tuple(selected)
