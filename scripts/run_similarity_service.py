#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from similarity_service import (
    AudioLoader,
    BucketPolicy,
    DEFAULT_BUCKET_SPECS,
    DEFAULT_CHECKPOINT,
    PyTorchEmbeddingBackend,
    ReferenceEmbeddingCache,
    SimilarityService,
    TensorRTEmbeddingBackend,
    create_app,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the WavLM+ECAPA similarity reward service.")
    parser.add_argument("--backend", choices=["pytorch", "tensorrt"], default="pytorch")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, help="TensorRT manifest JSON. Required when --backend=tensorrt.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allowed-root", action="append", default=[], help="Absolute root directory allowed for path requests.")
    parser.add_argument("--ref-cache-items", type=int, default=10000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main():
    args = parse_args()
    bucket_policy = BucketPolicy(DEFAULT_BUCKET_SPECS)
    audio_loader = AudioLoader(
        allowed_roots=[Path(root) for root in args.allowed_root],
        max_samples=bucket_policy.max_samples,
    )
    if args.backend == "pytorch":
        backend = PyTorchEmbeddingBackend(checkpoint_path=args.checkpoint, device=args.device)
    else:
        if args.manifest is None:
            raise ValueError("--manifest is required when --backend=tensorrt")
        backend = TensorRTEmbeddingBackend(manifest_path=args.manifest, device=args.device)

    service = SimilarityService(
        audio_loader=audio_loader,
        bucket_policy=bucket_policy,
        backend=backend,
        ref_cache=ReferenceEmbeddingCache(max_items=args.ref_cache_items),
    )
    app = create_app(service)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
