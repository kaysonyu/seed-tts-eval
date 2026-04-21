#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
TRITON_TEMPLATE_DIR = ROOT / "triton_templates"


def preferred_batch_sizes(max_batch: int) -> list[int]:
    candidates = [2, 4, 8, 16, 32]
    selected = [candidate for candidate in candidates if candidate <= max_batch]
    return selected or [1]


def _escape_pbtxt_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def render_tensorrt_config(
    model_name: str,
    num_samples: int,
    max_batch: int,
    queue_delay_us: int = 8000,
    instance_count: int = 1,
) -> str:
    preferred = ", ".join(str(value) for value in preferred_batch_sizes(max_batch))
    return (
        f'name: "{model_name}"\n'
        'platform: "tensorrt_plan"\n'
        f"max_batch_size: {max_batch}\n"
        "input [\n"
        "  {\n"
        '    name: "x"\n'
        "    data_type: TYPE_FP32\n"
        f"    dims: [ {num_samples} ]\n"
        "  },\n"
        "  {\n"
        '    name: "lengths"\n'
        "    data_type: TYPE_INT64\n"
        "    dims: [ ]\n"
        "  }\n"
        "]\n"
        "output [\n"
        "  {\n"
        '    name: "emb"\n'
        "    data_type: TYPE_FP32\n"
        "    dims: [ 256 ]\n"
        "  }\n"
        "]\n"
        "dynamic_batching {\n"
        f"  preferred_batch_size: [ {preferred} ]\n"
        f"  max_queue_delay_microseconds: {queue_delay_us}\n"
        "}\n"
        "instance_group [\n"
        "  {\n"
        "    kind: KIND_GPU\n"
        f"    count: {instance_count}\n"
        "  }\n"
        "]\n"
    )


def render_python_router_config(
    model_name: str,
    repo_root: Path,
    allowed_roots: Iterable[Path],
    bucket_seconds: list[int],
    bucket_model_prefix: str,
    max_batch_size: int = 64,
    queue_delay_us: int = 8000,
    normalize_embeddings: bool = True,
    instance_count: int = 4,
    embedding_cache_items: int = 4096,
    audio_backend: str = "soundfile",
) -> str:
    preferred = ", ".join(str(value) for value in preferred_batch_sizes(max_batch_size))
    parameters = {
        "repo_root": str(repo_root),
        "allowed_roots_json": json.dumps([str(Path(root).resolve()) for root in allowed_roots], ensure_ascii=False),
        "bucket_seconds_json": json.dumps(bucket_seconds, ensure_ascii=False),
        "bucket_model_prefix": bucket_model_prefix,
        "normalize_embeddings": "true" if normalize_embeddings else "false",
        "embedding_cache_items": str(int(embedding_cache_items)),
        "embedding_cache_namespace": f"{bucket_model_prefix}:normalize={normalize_embeddings}",
        "audio_backend": audio_backend,
    }
    rendered_parameters = "".join(
        (
            "parameters {\n"
            f'  key: "{key}"\n'
            "  value {\n"
            f'    string_value: "{_escape_pbtxt_string(value)}"\n'
            "  }\n"
            "}\n"
        )
        for key, value in parameters.items()
    )
    return (
        f'name: "{model_name}"\n'
        'backend: "python"\n'
        f"max_batch_size: {max_batch_size}\n"
        "input [\n"
        "  {\n"
        '    name: "AUDIO_PATH"\n'
        "    data_type: TYPE_STRING\n"
        "    dims: [ 1 ]\n"
        "  }\n"
        "]\n"
        "output [\n"
        "  {\n"
        '    name: "EMBEDDING"\n'
        "    data_type: TYPE_FP32\n"
        "    dims: [ 256 ]\n"
        "  }\n"
        "]\n"
        "dynamic_batching {\n"
        f"  preferred_batch_size: [ {preferred} ]\n"
        f"  max_queue_delay_microseconds: {queue_delay_us}\n"
        "}\n"
        "instance_group [\n"
        "  {\n"
        "    kind: KIND_CPU\n"
        f"    count: {instance_count}\n"
        "  }\n"
        "]\n"
        f"{rendered_parameters}"
    )


def _materialize_plan_artifact(source: Path, destination: Path, copy_plan: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if copy_plan:
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source.resolve())


def materialize_triton_model_repository(
    manifest_path: Path,
    outdir: Path,
    allowed_roots: Iterable[Path] = (),
    router_model_name: str = "path_embedding_router",
    bucket_model_prefix: str = "wavlm_ecapa",
    copy_plan: bool = False,
    queue_delay_us: int = 8000,
    router_instance_count: int = 4,
    embedding_cache_items: int = 4096,
    audio_backend: str = "soundfile",
    bucket_instance_counts: dict[int, int] | None = None,
) -> Path:
    manifest_path = Path(manifest_path).resolve()
    outdir = Path(outdir).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outdir.mkdir(parents=True, exist_ok=True)
    bucket_instance_counts = bucket_instance_counts or {}

    bucket_seconds: list[int] = []
    for entry in manifest["buckets"]:
        seconds = int(entry["seconds"])
        bucket_seconds.append(seconds)
        model_name = f"{bucket_model_prefix}_{entry['seconds']}s"
        model_dir = outdir / model_name
        version_dir = model_dir / "1"
        version_dir.mkdir(parents=True, exist_ok=True)
        engine_path = Path(entry["engine_path"]).resolve()
        if not engine_path.exists():
            raise FileNotFoundError(f"Missing TensorRT plan for bucket {entry['seconds']}s: {engine_path}")
        _materialize_plan_artifact(engine_path, version_dir / "model.plan", copy_plan=copy_plan)
        config_text = render_tensorrt_config(
            model_name=model_name,
            num_samples=int(entry["num_samples"]),
            max_batch=int(entry["max_batch"]),
            queue_delay_us=queue_delay_us,
            instance_count=bucket_instance_counts.get(seconds, 1),
        )
        (model_dir / "config.pbtxt").write_text(config_text, encoding="utf-8")

    router_dir = outdir / router_model_name
    router_version_dir = router_dir / "1"
    router_version_dir.mkdir(parents=True, exist_ok=True)
    template_model = TRITON_TEMPLATE_DIR / "path_embedding_router" / "1" / "model.py"
    shutil.copy2(template_model, router_version_dir / "model.py")
    router_config = render_python_router_config(
        model_name=router_model_name,
        repo_root=ROOT,
        allowed_roots=list(allowed_roots),
        bucket_seconds=bucket_seconds,
        bucket_model_prefix=bucket_model_prefix,
        queue_delay_us=queue_delay_us,
        instance_count=router_instance_count,
        embedding_cache_items=embedding_cache_items,
        audio_backend=audio_backend,
    )
    (router_dir / "config.pbtxt").write_text(router_config, encoding="utf-8")
    return outdir


def parse_bucket_instance_counts(values: list[str]) -> dict[int, int]:
    parsed: dict[int, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected --bucket-instance-count as <seconds>=<count>, got {value}")
        seconds_text, count_text = value.split("=", 1)
        seconds = int(seconds_text)
        count = int(count_text)
        if seconds <= 0 or count <= 0:
            raise ValueError(f"Bucket seconds and count must be positive, got {value}")
        parsed[seconds] = count
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Materialize a Triton model repository from a TensorRT bucket manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=ROOT / "artifacts" / "triton_model_repository")
    parser.add_argument("--allowed-root", action="append", default=[])
    parser.add_argument("--router-model-name", default="path_embedding_router")
    parser.add_argument("--bucket-model-prefix", default="wavlm_ecapa")
    parser.add_argument("--copy-plan", action="store_true")
    parser.add_argument("--queue-delay-us", type=int, default=8000)
    parser.add_argument("--router-instance-count", type=int, default=4)
    parser.add_argument("--embedding-cache-items", type=int, default=4096)
    parser.add_argument("--audio-backend", choices=("soundfile", "librosa"), default="soundfile")
    parser.add_argument(
        "--bucket-instance-count",
        action="append",
        default=["20=2", "24=2", "30=2"],
        help="Per-bucket TensorRT instance count, formatted as <seconds>=<count>. May be repeated.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo = materialize_triton_model_repository(
        manifest_path=args.manifest,
        outdir=args.outdir,
        allowed_roots=[Path(root) for root in args.allowed_root],
        router_model_name=args.router_model_name,
        bucket_model_prefix=args.bucket_model_prefix,
        copy_plan=args.copy_plan,
        queue_delay_us=args.queue_delay_us,
        router_instance_count=args.router_instance_count,
        embedding_cache_items=args.embedding_cache_items,
        audio_backend=args.audio_backend,
        bucket_instance_counts=parse_bucket_instance_counts(args.bucket_instance_count),
    )
    print(repo)


if __name__ == "__main__":
    main()
