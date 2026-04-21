# Triton Similarity Service Runbook

## 1. 目标

这个服务把音频路径转成 `WavLM Large + ECAPA_TDNN` speaker embedding：

1. 客户端向 Triton 发送一个音频绝对路径。
2. `path_embedding_router` 用 Python backend 校验路径、解码音频、重采样到 `16 kHz`、按时长落桶并 pad。
3. router 通过 Triton BLS 调用对应 bucket 的 TensorRT model。
4. Triton 返回 `EMBEDDING`，shape 为 `[1, 256]`。

当前只支持“在当前 Triton/TensorRT 环境里本机构建 TRT plan”。不要在脚本里 Docker-in-Docker，也不要用别的 TensorRT 版本构建后再拿到当前 Triton runtime 里加载。

## 2. 必要文件关系

### 模型源

- `wavlm_large_ecapa_tdnn.py`：单文件 WavLM Large + ECAPA_TDNN 实现，是 PyTorch reference、ONNX 导出和 TensorRT 构建的源头。

### Triton 主链路

- `similarity_service/audio.py`：路径校验、allowed roots、音频解码、取第一声道、重采样、文件签名。
- `similarity_service/buckets.py`：定义 `4/8/12/16/20/24/30/45/60/90s` buckets、`max_batch` 和 pad 逻辑。
- `similarity_service/triton_router_core.py`：纯 Python router 逻辑，负责 cache、按 bucket regroup、chunk、调用 bucket embedding 函数、按原顺序回填。
- `triton_templates/path_embedding_router/1/model.py`：Triton Python backend 模板，负责 `pb_utils` 输入输出和 BLS 调用。
- `scripts/build_triton_model_repository.py`：生成 Triton `config.pbtxt`，复制模板和 `model.plan`。

### 构建入口

- `scripts/build_similarity_trt_repo.sh`：推荐的一键入口，串起 ONNX 导出、TensorRT plan 构建、Triton model repository 生成。
- `scripts/build_similarity_trt_buckets.py`：内部 helper，导出每个 bucket 的 ONNX 和 `manifest.json`。
- `scripts/rebuild_trt_engines_from_manifest.py`：内部 helper，用当前环境的 `trtexec` 从 manifest 重建 plan。
- `scripts/build_triton_model_repository.py`：内部 helper，从 manifest 生成 Triton model repository。

### 备用 FastAPI 链路

- `similarity_service/backends.py`、`similarity_service/service.py`、`similarity_service/app.py`、`scripts/run_similarity_service.py`：本地 PyTorch/TensorRT FastAPI 备用链路，不属于 Triton 主链路。

## 3. 构建 TRT 和 Triton Repository

先进入仓库：

```bash
cd /inspire/ssd/project/embodied-multimodality/yukang-CZXS25240036/workspace/tts-eval
```

确认当前环境有 `trtexec`：

```bash
/usr/src/tensorrt/bin/trtexec --version
```

一键构建：

```bash
scripts/build_similarity_trt_repo.sh \
  --variant h200 \
  --trtexec /usr/src/tensorrt/bin/trtexec \
  --device 0 \
  --no-tf32 \
  --router-instance-count 4 \
  --embedding-cache-items 4096 \
  --audio-backend soundfile \
  --bucket-instance-count 20=2 \
  --bucket-instance-count 24=2 \
  --bucket-instance-count 30=2
```

默认输出到 HDD：

```text
/inspire/hdd/project/embodied-multimodality/public/kyu/workspace/tts-eval/artifacts/similarity_trt_buckets_h200
/inspire/hdd/project/embodied-multimodality/public/kyu/workspace/tts-eval/artifacts/triton_model_repository_h200
```

参数含义：

| 参数 | 含义 |
|---|---|
| `--variant h200` | 输出目录后缀 |
| `--trtexec /usr/src/tensorrt/bin/trtexec` | 使用当前环境的 TensorRT builder |
| `--device 0` | 构建/验证时使用 GPU 0 |
| `--no-tf32` | 禁用 TF32，保持更接近 PyTorch FP32 reference |
| `--router-instance-count 4` | Python router CPU instance 数 |
| `--embedding-cache-items 4096` | 每个 router instance 的 path -> embedding LRU cache 容量 |
| `--audio-backend soundfile` | 优先用 soundfile 解码，失败再 fallback |
| `--bucket-instance-count 24=2` | 指定某个 bucket 的 TensorRT instance count |

如果要换输出目录：

```bash
scripts/build_similarity_trt_repo.sh \
  --variant h200_test \
  --artifact-root /some/large/disk/artifacts \
  --trtexec /usr/src/tensorrt/bin/trtexec \
  --device 0 \
  --no-tf32
```

## 4. 启动 Triton

```bash
tritonserver \
  --model-repository /inspire/hdd/project/embodied-multimodality/public/kyu/workspace/tts-eval/artifacts/triton_model_repository_h200
```

健康检查：

```bash
curl http://localhost:8000/v2/health/live
curl http://localhost:8000/v2/health/ready
curl http://localhost:8000/v2/models/path_embedding_router/config
```

## 5. 请求方式

入口模型名：

```text
path_embedding_router
```

输入：

| 名称 | 类型 | shape |
|---|---|---|
| `AUDIO_PATH` | `TYPE_STRING` | `[1]` |

输出：

| 名称 | 类型 | shape |
|---|---|---|
| `EMBEDDING` | `TYPE_FP32` | `[256]` |

Python HTTP 示例：

```python
import numpy as np
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient("127.0.0.1:8000")

inp = httpclient.InferInput("AUDIO_PATH", [1], "BYTES")
inp.set_data_from_numpy(np.array(["/data/audio/example.wav"], dtype=object))
out = httpclient.InferRequestedOutput("EMBEDDING")

resp = client.infer("path_embedding_router", inputs=[inp], outputs=[out])
emb = resp.as_numpy("EMBEDDING")
print(emb.shape)  # (1, 256)
```

单个 request 只传一条路径。多请求共享 batching 由 Triton `dynamic_batching` 完成，不要在一个 request 里传多条路径。

## 6. 数值正确性

当前服务端验证过的误差量级：

| 场景 | 最大 max_abs_diff | 最大 mean_abs_diff | 最小 cosine |
|---|---:|---:|---:|
| 单请求，覆盖 `4/8/16/24s` buckets | `2.30e-6` | `5.29e-7` | `0.99999988` |
| 并发 unique path 请求，覆盖 `4/8/16/24s` buckets | `2.62e-6` | `6.25e-7` | `1.0` |

验证报告保存在 HDD：

```text
/inspire/hdd/project/embodied-multimodality/public/kyu/workspace/tts-eval/numerical_validation_audio/triton_numerical_validation_report.json
/inspire/hdd/project/embodied-multimodality/public/kyu/workspace/tts-eval/numerical_validation_audio/triton_numerical_validation_concurrent_report.json
```

## 7. 常见错误

### TensorRT serialization version mismatch

典型日志：

```text
Serialization assertion stdVersionRead == kSERIALIZATION_VERSION failed
Current Version: 239, Serialized Engine Version: 240
```

原因是构建 `model.plan` 的 TensorRT 版本和 Triton runtime 的 TensorRT 版本不一致。解决方式是在当前 Triton/TensorRT runtime 环境内重新运行：

```bash
scripts/build_similarity_trt_repo.sh \
  --variant h200 \
  --trtexec /usr/src/tensorrt/bin/trtexec \
  --device 0 \
  --no-tf32
```

### `trtexec: command not found`

Triton 镜像中常见路径是：

```bash
/usr/src/tensorrt/bin/trtexec
```

不要只写 `trtexec`，除非它已经在 `PATH` 中。

### `Module onnx is not installed`

ONNX 导出阶段需要 `onnx` 包。它属于构建依赖，不是 TensorRT runtime 自己提供的能力。安装后重新运行构建脚本。

### router READY 但 bucket UNAVAILABLE

通常是某个 `model.plan` 加载失败。优先检查：

- `model.plan` 是否由当前 Triton runtime 对应的 TensorRT 构建。
- `model.plan` 是否存在且完整。
- `config.pbtxt` 中的输入 shape 是否和 plan 一致。

## 8. 维护原则

- 改 router 业务逻辑：改 `similarity_service/triton_router_core.py`。
- 改 Triton Python backend I/O 或 BLS：改 `triton_templates/path_embedding_router/1/model.py`。
- 改 Triton config 生成：改 `scripts/build_triton_model_repository.py`。
- 不要直接改生成出来的 `artifacts/triton_model_repository_*/path_embedding_router/1/model.py`，重新生成 repo 即可。
- 换 Triton/TensorRT 版本、GPU 架构、checkpoint、bucket 列表、模型代码或精度策略时，都要重新构建。
