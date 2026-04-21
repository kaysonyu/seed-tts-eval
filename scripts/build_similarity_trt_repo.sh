#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/build_similarity_trt_repo.sh \
    --variant <name> \
    [--trtexec <path>] \
    [--device <gpu-index>] \
    [--artifact-root <abs-path>] \
    [--allowed-root <abs-path>]... \
    [--no-tf32] \
    [--hardware-compatibility-level <none|ampere+|sameComputeCapability>] \
    [--version-compatible] \
    [--router-instance-count <n>] \
    [--embedding-cache-items <n>] \
    [--audio-backend <soundfile|librosa>] \
    [--bucket-instance-count <seconds=count>]...

Behavior:
  - Builds in the current environment only; there is no Docker-in-Docker path.
  - Use the same TensorRT runtime environment that will run Triton.
  - Default trtexec is /usr/src/tensorrt/bin/trtexec.
  - Default artifact root is /inspire/hdd/project/embodied-multimodality/public/kyu/workspace/tts-eval/artifacts.

Example:
  scripts/build_similarity_trt_repo.sh \
    --variant h200 \
    --trtexec /usr/src/tensorrt/bin/trtexec \
    --device 0 \
    --allowed-root /data/audio \
    --no-tf32 \
    --router-instance-count 4 \
    --embedding-cache-items 4096 \
    --audio-backend soundfile \
    --bucket-instance-count 20=2 \
    --bucket-instance-count 24=2 \
    --bucket-instance-count 30=2

Outputs:
  <artifact-root>/similarity_trt_buckets_<variant>/
  <artifact-root>/triton_model_repository_<variant>/
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_ARTIFACT_ROOT="/inspire/hdd/project/embodied-multimodality/public/kyu/workspace/tts-eval/artifacts"
VARIANT=""
TRTEXEC="/usr/src/tensorrt/bin/trtexec"
DEVICE="0"
ARTIFACT_ROOT="$DEFAULT_ARTIFACT_ROOT"
NO_TF32=0
HW_COMPAT="none"
VERSION_COMPATIBLE=0
ROUTER_INSTANCE_COUNT=4
EMBEDDING_CACHE_ITEMS=4096
AUDIO_BACKEND="soundfile"
ALLOWED_ROOTS=()
BUCKET_INSTANCE_COUNTS=("20=2" "24=2" "30=2")

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "python3/python is required for this script" >&2
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant)
      VARIANT="${2:-}"
      shift 2
      ;;
    --trtexec)
      TRTEXEC="${2:-}"
      shift 2
      ;;
    --device)
      DEVICE="${2:-}"
      shift 2
      ;;
    --artifact-root)
      ARTIFACT_ROOT="${2:-}"
      shift 2
      ;;
    --allowed-root)
      ALLOWED_ROOTS+=("${2:-}")
      shift 2
      ;;
    --no-tf32)
      NO_TF32=1
      shift
      ;;
    --hardware-compatibility-level)
      HW_COMPAT="${2:-}"
      shift 2
      ;;
    --version-compatible)
      VERSION_COMPATIBLE=1
      shift
      ;;
    --router-instance-count)
      ROUTER_INSTANCE_COUNT="${2:-}"
      shift 2
      ;;
    --embedding-cache-items)
      EMBEDDING_CACHE_ITEMS="${2:-}"
      shift 2
      ;;
    --audio-backend)
      AUDIO_BACKEND="${2:-}"
      shift 2
      ;;
    --bucket-instance-count)
      BUCKET_INSTANCE_COUNTS+=("${2:-}")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$VARIANT" ]]; then
  usage >&2
  exit 1
fi

if [[ "$ARTIFACT_ROOT" != /* ]]; then
  echo "--artifact-root must be an absolute path, got: $ARTIFACT_ROOT" >&2
  exit 1
fi

if [[ ! -x "$TRTEXEC" ]]; then
  echo "local trtexec not found or not executable: $TRTEXEC" >&2
  echo "Run this script inside the target Triton/TensorRT environment, or pass --trtexec <path>." >&2
  exit 1
fi

mkdir -p "$ARTIFACT_ROOT"

BUCKET_OUTDIR="$ARTIFACT_ROOT/similarity_trt_buckets_${VARIANT}"
REPO_OUTDIR="$ARTIFACT_ROOT/triton_model_repository_${VARIANT}"
MANIFEST_PATH="$BUCKET_OUTDIR/manifest.json"

EXPORT_ARGS=(
  "$PYTHON_BIN" scripts/build_similarity_trt_buckets.py
  --outdir "$BUCKET_OUTDIR"
  --skip-trt-build
)
if [[ "$NO_TF32" -eq 1 ]]; then
  EXPORT_ARGS+=(--no-tf32)
fi

REBUILD_ARGS=(
  "$PYTHON_BIN" scripts/rebuild_trt_engines_from_manifest.py
  --manifest "$MANIFEST_PATH"
  --trtexec "$TRTEXEC"
  --device "$DEVICE"
  --hardware-compatibility-level "$HW_COMPAT"
)
if [[ "$NO_TF32" -eq 1 ]]; then
  REBUILD_ARGS+=(--no-tf32)
fi
if [[ "$VERSION_COMPATIBLE" -eq 1 ]]; then
  REBUILD_ARGS+=(--version-compatible)
fi

REPO_ARGS=(
  "$PYTHON_BIN" scripts/build_triton_model_repository.py
  --manifest "$MANIFEST_PATH"
  --outdir "$REPO_OUTDIR"
  --copy-plan
  --router-instance-count "$ROUTER_INSTANCE_COUNT"
  --embedding-cache-items "$EMBEDDING_CACHE_ITEMS"
  --audio-backend "$AUDIO_BACKEND"
)
for root in "${ALLOWED_ROOTS[@]}"; do
  REPO_ARGS+=(--allowed-root "$root")
done
for value in "${BUCKET_INSTANCE_COUNTS[@]}"; do
  REPO_ARGS+=(--bucket-instance-count "$value")
done

echo "[1/3] Exporting ONNX buckets to $BUCKET_OUTDIR"
(
  cd "$ROOT"
  "${EXPORT_ARGS[@]}"
)

echo "[2/3] Rebuilding TensorRT engines with local TRT: $TRTEXEC"
(
  cd "$ROOT"
  "${REBUILD_ARGS[@]}"
)

echo "[3/3] Materializing Triton model repository to $REPO_OUTDIR"
(
  cd "$ROOT"
  "${REPO_ARGS[@]}"
)

echo
echo "DONE"
echo "Artifact root: $ARTIFACT_ROOT"
echo "Manifest: $MANIFEST_PATH"
echo "Repository: $REPO_OUTDIR"
