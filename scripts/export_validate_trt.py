#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Dict

import onnxruntime as ort
import tensorrt as trt
import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "wavlm_large_ecapa_tdnn.py"
DEFAULT_CHECKPOINT = Path(
    "/inspire/hdd/project/embodied-multimodality/public/btjiang/tts/checkpoint/Seed-Similarity/wavlm_large_finetune.pth"
)
DEFAULT_TRTEXEC = Path("/opt/tensorrt/bin/trtexec")


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_model(module, checkpoint: Path, device: torch.device):
    state = torch.load(checkpoint, map_location="cpu")["model"]
    model = module.WavLMLargeECAPATDNN().eval().to(device)
    model.load_state_dict(state, strict=True)
    return model


def build_examples(device: torch.device) -> Dict[str, Dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    random_pad = {
        "x": torch.randn(2, 16000, dtype=torch.float32, device=device),
        "lengths": torch.tensor([16000, 12000], dtype=torch.int64, device=device),
    }

    sr = 16000
    t = torch.arange(0, sr, dtype=torch.float32, device=device) / sr
    sine_a = 0.2 * torch.sin(2 * math.pi * 220 * t)
    sine_b = 0.15 * torch.sin(2 * math.pi * 440 * t + 0.3)
    sine_b = sine_b + 0.05 * torch.cos(2 * math.pi * 90 * t)
    deterministic_full = {
        "x": torch.stack([sine_a, sine_b], dim=0),
        "lengths": torch.tensor([16000, 16000], dtype=torch.int64, device=device),
    }

    return {
        "random_pad": random_pad,
        "deterministic_full": deterministic_full,
    }


def export_onnx(model, x: torch.Tensor, lengths: torch.Tensor, onnx_path: Path) -> float:
    start = time.perf_counter()
    torch.onnx.export(
        model,
        (x, lengths),
        str(onnx_path),
        input_names=["x", "lengths"],
        output_names=["emb"],
        opset_version=18,
        dynamo=True,
    )
    return time.perf_counter() - start


def run_onnxruntime(onnx_path: Path, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    output = session.run(
        None,
        {
            "x": x.detach().cpu().numpy(),
            "lengths": lengths.detach().cpu().numpy(),
        },
    )[0]
    return torch.from_numpy(output)


def build_tensorrt_engine(
    trtexec: Path,
    onnx_path: Path,
    engine_path: Path,
    log_path: Path,
    no_tf32: bool,
) -> tuple[float, list[str]]:
    cmd = [
        str(trtexec),
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--skipInference",
    ]
    if no_tf32:
        cmd.append("--noTF32")

    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=True,
    )
    duration = time.perf_counter() - start
    log_path.write_text(proc.stdout, encoding="utf-8")
    return duration, cmd


def trt_dtype_to_torch(dtype):
    return {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int64: torch.int64,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }[dtype]


def run_tensorrt_engine(engine_path: Path, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
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


def compare_outputs(reference: torch.Tensor, output: torch.Tensor) -> Dict[str, float]:
    diff = (reference - output).abs()
    return {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "cosine_mean": float(
            torch.nn.functional.cosine_similarity(reference, output, dim=-1).mean().item()
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export the single-file WavLM+ECAPA model to ONNX, build a TensorRT engine, and validate ORT/TRT outputs."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--outdir", type=Path, default=ROOT / "artifacts" / "tensorrt_static")
    parser.add_argument("--trtexec", type=Path, default=DEFAULT_TRTEXEC)
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TensorRT validation")
    if not args.trtexec.exists():
        raise FileNotFoundError(f"Missing trtexec: {args.trtexec}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")

    cpu_device = torch.device("cpu")
    gpu_device = torch.device(args.device)
    args.outdir.mkdir(parents=True, exist_ok=True)

    module = load_module(MODEL_FILE, "wavlm_large_ecapa_tdnn_export_validate")

    load_start = time.perf_counter()
    model = load_model(module, args.checkpoint, cpu_device)
    load_duration = time.perf_counter() - load_start

    examples = build_examples(cpu_device)
    export_example = examples["random_pad"]

    onnx_name = "wavlm_large_ecapa_tdnn_static_no_tf32.onnx" if args.no_tf32 else "wavlm_large_ecapa_tdnn_static.onnx"
    plan_name = "wavlm_large_ecapa_tdnn_static_no_tf32.plan" if args.no_tf32 else "wavlm_large_ecapa_tdnn_static.plan"
    log_name = "trtexec_no_tf32.log" if args.no_tf32 else "trtexec.log"
    report_name = "report_no_tf32.json" if args.no_tf32 else "report.json"

    onnx_path = args.outdir / onnx_name
    engine_path = args.outdir / plan_name
    log_path = args.outdir / log_name
    report_path = args.outdir / report_name

    export_duration = export_onnx(
        model,
        export_example["x"],
        export_example["lengths"],
        onnx_path,
    )

    ort_start = time.perf_counter()
    ort_metrics = {}
    eager_outputs = {}
    for name, example in examples.items():
        with torch.no_grad():
            eager = model(example["x"], example["lengths"]).detach().cpu()
        eager_outputs[name] = eager
        ort_out = run_onnxruntime(onnx_path, example["x"], example["lengths"])
        ort_metrics[name] = compare_outputs(eager, ort_out)
    ort_duration = time.perf_counter() - ort_start

    trt_build_duration, trtexec_cmd = build_tensorrt_engine(
        trtexec=args.trtexec,
        onnx_path=onnx_path,
        engine_path=engine_path,
        log_path=log_path,
        no_tf32=args.no_tf32,
    )

    trt_validate_start = time.perf_counter()
    trt_metrics = {}
    for name, example in examples.items():
        trt_out = run_tensorrt_engine(
            engine_path,
            example["x"].to(gpu_device),
            example["lengths"].to(gpu_device),
        )
        trt_metrics[name] = compare_outputs(eager_outputs[name], trt_out)
    trt_validate_duration = time.perf_counter() - trt_validate_start

    total_duration = (
        load_duration
        + export_duration
        + ort_duration
        + trt_build_duration
        + trt_validate_duration
    )

    report = {
        "env": {
            "checkpoint": str(args.checkpoint),
            "device": str(gpu_device),
            "trtexec": str(args.trtexec),
            "no_tf32": args.no_tf32,
        },
        "artifacts": {
            "onnx": str(onnx_path),
            "engine": str(engine_path),
            "trtexec_log": str(log_path),
            "report": str(report_path),
        },
        "commands": {
            "trtexec": trtexec_cmd,
        },
        "durations_sec": {
            "load_model": load_duration,
            "export_onnx": export_duration,
            "validate_onnxruntime": ort_duration,
            "build_tensorrt": trt_build_duration,
            "validate_tensorrt": trt_validate_duration,
            "total": total_duration,
        },
        "metrics": {
            "onnxruntime": ort_metrics,
            "tensorrt": trt_metrics,
        },
    }

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
