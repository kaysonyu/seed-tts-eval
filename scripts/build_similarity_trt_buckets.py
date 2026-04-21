 #!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import onnxruntime as ort
except ImportError:  # ONNX Runtime validation is optional in minimal Triton images.
    ort = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from similarity_service import (
    BucketPolicy,
    DEFAULT_BUCKET_SPECS,
    DEFAULT_CHECKPOINT,
    fingerprint_checkpoint,
)
from wavlm_large_ecapa_tdnn import load_wavlm_large_ecapa_tdnn

DEFAULT_TRTEXEC = Path("/usr/src/tensorrt/bin/trtexec")


def compare_outputs(reference: torch.Tensor, output: torch.Tensor) -> dict[str, float]:
    reference = reference.detach().cpu().to(dtype=torch.float32)
    output = output.detach().cpu().to(dtype=torch.float32)
    diff = (reference - output).abs()
    return {
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "cosine_mean": F.cosine_similarity(reference, output, dim=-1).mean().item(),
    }


def trt_dtype_to_torch(dtype) -> torch.dtype:
    import tensorrt as trt

    return {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int64: torch.int64,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }[dtype]


def run_tensorrt_engine(engine_path: Path, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    tensors = {
        "x": x.contiguous(),
        "lengths": lengths.contiguous(),
    }

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
            continue
        if -1 in tuple(context.get_tensor_shape(name)):
            context.set_input_shape(name, tuple(tensors[name].shape))

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
            continue
        tensors[name] = torch.empty(
            tuple(context.get_tensor_shape(name)),
            dtype=trt_dtype_to_torch(engine.get_tensor_dtype(name)),
            device=x.device,
        )

    for name, tensor in tensors.items():
        context.set_tensor_address(name, tensor.data_ptr())

    stream = torch.cuda.current_stream(device=x.device).cuda_stream
    ok = context.execute_async_v3(stream)
    if not ok:
        raise RuntimeError("TensorRT execute_async_v3 failed")
    torch.cuda.synchronize(device=x.device)
    return tensors["emb"].detach().cpu()


def build_examples(bucket_samples: int, max_batch: int) -> dict[str, dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    full = {
        "x": torch.randn(1, bucket_samples, dtype=torch.float32),
        "lengths": torch.tensor([bucket_samples], dtype=torch.int64),
    }
    partial_batch = min(max_batch, 2)
    partial_lengths = [bucket_samples]
    if partial_batch == 2:
        partial_lengths.append(max(bucket_samples - 16000, bucket_samples // 2))
    partial = {
        "x": torch.randn(partial_batch, bucket_samples, dtype=torch.float32),
        "lengths": torch.tensor(partial_lengths, dtype=torch.int64),
    }

    sr = 16000
    t = torch.arange(0, bucket_samples, dtype=torch.float32) / sr
    sine = 0.2 * torch.sin(2 * math.pi * 220 * t)
    deterministic = {
        "x": sine.unsqueeze(0),
        "lengths": torch.tensor([bucket_samples], dtype=torch.int64),
    }
    return {
        "full": full,
        "partial": partial,
        "deterministic": deterministic,
    }


def export_onnx_dynamic_batch(model, example: dict[str, torch.Tensor], onnx_path: Path) -> float:
    start = time.perf_counter()
    torch.onnx.export(
        model,
        (example["x"], example["lengths"]),
        str(onnx_path),
        input_names=["x", "lengths"],
        output_names=["emb"],
        opset_version=18,
        dynamo=False,
        dynamic_axes={
            "x": {0: "batch"},
            "lengths": {0: "batch"},
            "emb": {0: "batch"},
        },
    )
    return time.perf_counter() - start


def run_onnxruntime(onnx_path: Path, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    if ort is None:
        raise RuntimeError("onnxruntime is not installed")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    output = session.run(
        None,
        {
            "x": x.detach().cpu().numpy(),
            "lengths": lengths.detach().cpu().numpy(),
        },
    )[0]
    return torch.from_numpy(output)


def build_tensorrt_engine_dynamic_batch(
    trtexec: Path,
    onnx_path: Path,
    engine_path: Path,
    log_path: Path,
    bucket_samples: int,
    max_batch: int,
    no_tf32: bool,
) -> tuple[float, list[str], int]:
    opt_batch = min(max_batch, 8)
    cmd = [
        str(trtexec),
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--minShapes=x:1x{bucket_samples},lengths:1",
        f"--optShapes=x:{opt_batch}x{bucket_samples},lengths:{opt_batch}",
        f"--maxShapes=x:{max_batch}x{bucket_samples},lengths:{max_batch}",
        "--skipInference",
    ]
    if no_tf32:
        cmd.append("--noTF32")
    start = time.perf_counter()
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    duration = time.perf_counter() - start
    log_path.write_text(proc.stdout, encoding="utf-8")
    return duration, cmd, opt_batch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export bucketed dynamic-batch ONNX models and TensorRT engines for the similarity service."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--outdir", type=Path, default=ROOT / "artifacts" / "similarity_trt_buckets")
    parser.add_argument("--bucket-seconds", type=int, nargs="*", help="Optional custom bucket second list.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trtexec", type=Path, default=DEFAULT_TRTEXEC)
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--skip-trt-build", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")
    if not args.skip_trt_build and not args.trtexec.exists():
        raise FileNotFoundError(f"Missing trtexec: {args.trtexec}")
    if not args.skip_trt_build and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TensorRT build and validation")

    policy = BucketPolicy(DEFAULT_BUCKET_SPECS)
    selected_specs = (
        policy.bucket_specs_from_seconds(args.bucket_seconds)
        if args.bucket_seconds
        else DEFAULT_BUCKET_SPECS
    )
    args.outdir.mkdir(parents=True, exist_ok=True)

    cpu_device = torch.device("cpu")
    gpu_device = torch.device(args.device)
    model = load_wavlm_large_ecapa_tdnn(
        checkpoint_path=str(args.checkpoint),
        map_location="cpu",
    ).eval().to(cpu_device)

    manifest = {
        "backend_name": (
            "onnx_export_only"
            if args.skip_trt_build
            else ("tensorrt_fp32_no_tf32" if args.no_tf32 else "tensorrt_fp32")
        ),
        "checkpoint": str(args.checkpoint),
        "checkpoint_fingerprint": fingerprint_checkpoint(args.checkpoint),
        "device": args.device,
        "bucket_policy_version": "-".join(str(spec.seconds) for spec in selected_specs),
        "no_tf32": args.no_tf32,
        "buckets": [],
    }

    for spec in selected_specs:
        bucket_dir = args.outdir / f"{spec.seconds}s"
        bucket_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = bucket_dir / "wavlm_large_ecapa_tdnn_dynamic.onnx"
        engine_path = bucket_dir / "wavlm_large_ecapa_tdnn_dynamic.plan"
        log_path = bucket_dir / "trtexec.log"
        report_path = bucket_dir / "report.json"

        examples = build_examples(spec.num_samples, spec.max_batch)
        export_duration = export_onnx_dynamic_batch(model, examples["partial"], onnx_path)

        ort_metrics = {}
        eager_outputs = {}
        ort_start = time.perf_counter()
        for name, example in examples.items():
            with torch.no_grad():
                eager = model(example["x"], example["lengths"]).detach().cpu()
            eager_outputs[name] = eager
            if ort is not None:
                ort_out = run_onnxruntime(onnx_path, example["x"], example["lengths"])
                ort_metrics[name] = compare_outputs(eager, ort_out)
        ort_duration = time.perf_counter() - ort_start

        bucket_report = {
            "seconds": spec.seconds,
            "num_samples": spec.num_samples,
            "max_batch": spec.max_batch,
            "onnx_path": str(onnx_path),
            "engine_path": str(engine_path),
            "trtexec_log": str(log_path),
            "durations_sec": {
                "export_onnx": export_duration,
                "validate_onnxruntime": ort_duration if ort is not None else 0.0,
            },
            "metrics": {
                "onnxruntime": ort_metrics
                if ort is not None
                else {"skipped": "onnxruntime Python package is not installed"},
            },
        }

        if not args.skip_trt_build:
            trt_build_duration, trtexec_cmd, opt_batch = build_tensorrt_engine_dynamic_batch(
                trtexec=args.trtexec,
                onnx_path=onnx_path,
                engine_path=engine_path,
                log_path=log_path,
                bucket_samples=spec.num_samples,
                max_batch=spec.max_batch,
                no_tf32=args.no_tf32,
            )
            trt_metrics = {}
            trt_validate_start = time.perf_counter()
            for name, example in examples.items():
                trt_out = run_tensorrt_engine(
                    engine_path=engine_path,
                    x=example["x"].to(gpu_device),
                    lengths=example["lengths"].to(gpu_device),
                )
                trt_metrics[name] = compare_outputs(eager_outputs[name], trt_out)
            trt_validate_duration = time.perf_counter() - trt_validate_start
            bucket_report["profile"] = {
                "min_batch": 1,
                "opt_batch": opt_batch,
                "max_batch": spec.max_batch,
            }
            bucket_report["commands"] = {"trtexec": trtexec_cmd}
            bucket_report["durations_sec"]["build_tensorrt"] = trt_build_duration
            bucket_report["durations_sec"]["validate_tensorrt"] = trt_validate_duration
            bucket_report["metrics"]["tensorrt"] = trt_metrics

        report_path.write_text(json.dumps(bucket_report, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest["buckets"].append(bucket_report)

    manifest_path = args.outdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
