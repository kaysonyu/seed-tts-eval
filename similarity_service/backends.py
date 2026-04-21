from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from wavlm_large_ecapa_tdnn import load_wavlm_large_ecapa_tdnn

try:
    import tensorrt as trt
except ImportError:  # pragma: no cover - handled by runtime checks
    trt = None


DEFAULT_CHECKPOINT = Path(
    "/inspire/hdd/project/embodied-multimodality/public/btjiang/tts/checkpoint/Seed-Similarity/wavlm_large_finetune.pth"
)


def fingerprint_checkpoint(checkpoint_path: Path) -> str:
    hasher = hashlib.sha256()
    with checkpoint_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def trt_dtype_to_torch(dtype):
    return {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int64: torch.int64,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }[dtype]


class EmbeddingBackend:
    name = "unknown"

    @property
    def is_ready(self) -> bool:
        return True

    @property
    def fingerprint(self) -> str:
        return "unknown"

    def embed_padded_batch(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface only
        raise NotImplementedError


class PyTorchEmbeddingBackend(EmbeddingBackend):
    name = "pytorch_eager"

    def __init__(self, checkpoint_path: Path = DEFAULT_CHECKPOINT, device: str = "cuda:0"):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self._fingerprint = fingerprint_checkpoint(self.checkpoint_path)
        model = load_wavlm_large_ecapa_tdnn(
            checkpoint_path=str(self.checkpoint_path),
            map_location="cpu",
        )
        self.model = model.eval().to(self.device)

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def embed_padded_batch(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.device, dtype=torch.float32, non_blocking=True)
        lengths = lengths.to(device=self.device, dtype=torch.long, non_blocking=True)
        with torch.no_grad():
            emb = self.model(x, lengths)
        return emb.detach().cpu()


@dataclass(frozen=True, slots=True)
class TensorRTBucketArtifact:
    seconds: int
    num_samples: int
    max_batch: int
    engine_path: Path
    onnx_path: Path | None = None
    trtexec_log: Path | None = None
    metrics: dict[str, Any] | None = None


class TensorRTEngineRunner:
    def __init__(self, artifact: TensorRTBucketArtifact, device: str = "cuda:0"):
        if trt is None:  # pragma: no cover - environment dependent
            raise ImportError("tensorrt is required for the TensorRT backend")
        self.artifact = artifact
        self.device = torch.device(device)
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        with artifact.engine_path.open("rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine {artifact.engine_path}")

    def infer(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if x.size(1) != self.artifact.num_samples:
            raise ValueError(
                f"Engine {self.artifact.engine_path} expects T={self.artifact.num_samples}, "
                f"got {x.size(1)}"
            )
        if x.size(0) > self.artifact.max_batch:
            raise ValueError(
                f"Engine {self.artifact.engine_path} supports batch <= {self.artifact.max_batch}, "
                f"got {x.size(0)}"
            )

        x = x.to(device=self.device, dtype=torch.float32, non_blocking=True).contiguous()
        lengths = lengths.to(device=self.device, dtype=torch.long, non_blocking=True).contiguous()
        context = self.engine.create_execution_context()
        tensors: dict[str, torch.Tensor] = {
            "x": x,
            "lengths": lengths,
        }

        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            if self.engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
                continue
            context.set_input_shape(name, tuple(tensors[name].shape))

        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            if self.engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
                continue
            tensors[name] = torch.empty(
                tuple(context.get_tensor_shape(name)),
                dtype=trt_dtype_to_torch(self.engine.get_tensor_dtype(name)),
                device=self.device,
            )

        for name, tensor in tensors.items():
            context.set_tensor_address(name, tensor.data_ptr())

        stream = torch.cuda.current_stream(device=self.device).cuda_stream
        if not context.execute_async_v3(stream):
            raise RuntimeError(f"TensorRT execute_async_v3 failed for {self.artifact.engine_path}")
        torch.cuda.synchronize(device=self.device)
        return tensors["emb"].detach().cpu()


class TensorRTEmbeddingBackend(EmbeddingBackend):
    name = "tensorrt_fp32"

    def __init__(self, manifest_path: Path, device: str = "cuda:0"):
        self.manifest_path = Path(manifest_path)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self._fingerprint = manifest.get("checkpoint_fingerprint", "unknown")
        self.backend_name = manifest.get("backend_name", self.name)
        self.device = device
        self.artifacts: dict[int, TensorRTBucketArtifact] = {}
        self.runners: dict[int, TensorRTEngineRunner] = {}
        for entry in manifest["buckets"]:
            artifact = TensorRTBucketArtifact(
                seconds=int(entry["seconds"]),
                num_samples=int(entry["num_samples"]),
                max_batch=int(entry["max_batch"]),
                engine_path=Path(entry["engine_path"]),
                onnx_path=Path(entry["onnx_path"]) if entry.get("onnx_path") else None,
                trtexec_log=Path(entry["trtexec_log"]) if entry.get("trtexec_log") else None,
                metrics=entry.get("metrics"),
            )
            self.artifacts[artifact.num_samples] = artifact
            self.runners[artifact.num_samples] = TensorRTEngineRunner(artifact, device=device)

    @property
    def name(self) -> str:
        return self.backend_name

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def is_ready(self) -> bool:
        return all(artifact.engine_path.exists() for artifact in self.artifacts.values())

    def embed_padded_batch(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        num_samples = int(x.size(1))
        if num_samples not in self.runners:
            supported = ", ".join(str(key) for key in sorted(self.runners))
            raise ValueError(f"No TensorRT engine for T={num_samples}. Supported: {supported}")
        return self.runners[num_samples].infer(x, lengths)
