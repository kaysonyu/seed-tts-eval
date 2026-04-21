#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild TensorRT engines from an existing bucket manifest using the local trtexec."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trtexec", type=Path, default=Path("/usr/src/tensorrt/bin/trtexec"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument(
        "--hardware-compatibility-level",
        choices=("none", "ampere+", "sameComputeCapability"),
        default="none",
    )
    parser.add_argument(
        "--version-compatible",
        action="store_true",
        help="Build a version-compatible TensorRT engine. This only helps loading on newer TensorRT runtimes, not older ones.",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="Optional output manifest path. Defaults to overwrite the input manifest.",
    )
    return parser.parse_args()


def build_engine_for_bucket(
    trtexec: Path,
    onnx_path: Path,
    engine_path: Path,
    log_path: Path,
    num_samples: int,
    max_batch: int,
    device: int,
    no_tf32: bool,
    hardware_compatibility_level: str,
    version_compatible: bool,
) -> tuple[list[str], float, int]:
    opt_batch = min(max_batch, 8)
    cmd = [
        str(trtexec),
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--minShapes=x:1x{num_samples},lengths:1",
        f"--optShapes=x:{opt_batch}x{num_samples},lengths:{opt_batch}",
        f"--maxShapes=x:{max_batch}x{num_samples},lengths:{max_batch}",
        f"--device={device}",
        "--skipInference",
    ]
    if no_tf32:
        cmd.append("--noTF32")
    if hardware_compatibility_level != "none":
        cmd.append(f"--hardwareCompatibilityLevel={hardware_compatibility_level}")
    if version_compatible:
        cmd.append("--versionCompatible")
    start = time.perf_counter()
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    duration = time.perf_counter() - start
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout, encoding="utf-8")
    return cmd, duration, opt_batch


def main():
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not args.trtexec.exists():
        raise FileNotFoundError(f"Missing trtexec: {args.trtexec}")

    for entry in manifest["buckets"]:
        seconds = int(entry["seconds"])
        onnx_path = Path(entry["onnx_path"]).resolve()
        if not onnx_path.exists():
            raise FileNotFoundError(f"Missing ONNX for bucket {seconds}s: {onnx_path}")

        engine_path = Path(entry["engine_path"]).resolve()
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        log_value = entry.get("trtexec_log")
        log_path = Path(log_value).resolve() if log_value else engine_path.parent / "trtexec.log"

        cmd, duration, opt_batch = build_engine_for_bucket(
            trtexec=args.trtexec,
            onnx_path=onnx_path,
            engine_path=engine_path,
            log_path=log_path,
            num_samples=int(entry["num_samples"]),
            max_batch=int(entry["max_batch"]),
            device=args.device,
            no_tf32=args.no_tf32,
            hardware_compatibility_level=args.hardware_compatibility_level,
            version_compatible=args.version_compatible,
        )

        entry["engine_path"] = str(engine_path)
        entry["trtexec_log"] = str(log_path)
        entry.setdefault("profile", {})
        entry["profile"]["min_batch"] = 1
        entry["profile"]["opt_batch"] = opt_batch
        entry["profile"]["max_batch"] = int(entry["max_batch"])
        entry.setdefault("commands", {})
        entry["commands"]["trtexec"] = cmd
        entry.setdefault("durations_sec", {})
        entry["durations_sec"]["build_tensorrt"] = duration

        print(
            json.dumps(
                {
                    "bucket_seconds": seconds,
                    "engine_path": str(engine_path),
                    "build_tensorrt_sec": duration,
                },
                ensure_ascii=False,
            )
        )
        sys.stdout.flush()

    backend_name = "tensorrt_fp32_no_tf32" if args.no_tf32 else "tensorrt_fp32"
    if args.version_compatible:
        backend_name = f"{backend_name}_vc"
    if args.hardware_compatibility_level != "none":
        backend_name = f"{backend_name}_{args.hardware_compatibility_level}"

    manifest["backend_name"] = backend_name
    manifest["device"] = f"cuda:{args.device}"
    manifest["no_tf32"] = bool(args.no_tf32)
    manifest["version_compatible"] = bool(args.version_compatible)
    manifest["hardware_compatibility_level"] = args.hardware_compatibility_level
    out_path = args.manifest_out.resolve() if args.manifest_out is not None else manifest_path
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
