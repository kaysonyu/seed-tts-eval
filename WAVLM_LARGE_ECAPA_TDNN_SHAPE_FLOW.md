# WavLM Large + ECAPA_TDNN 形状流转说明

本文档说明当前单文件 `wavlm_large_ecapa_tdnn.py` 的推理接口、形状变化、pad 处理方式，以及它为什么适合后续 trace / TensorRT。

当前核心文件：

- `wavlm_large_ecapa_tdnn.py`
- 一致性与导出测试：`test_wavlm_large_ecapa_tdnn_consistency.py`

## 1. 当前接口

当前核心推理入口固定为：

```python
emb = model(x, lengths)
```

其中：

- `x.shape == [B, T_bucket]`
- `lengths.shape == [B]`
- `x` 由模型外部分桶并 pad
- `lengths` 是每条语音的真实采样点长度

也就是说：

- 模型内部不再接收 `List[Tensor]`
- 模型内部不再做 `pad_sequence`
- 核心导出路径里不依赖 `lengths=None`

这也是当前实现可 trace / export 的前提。

## 2. 例子 A：等长 batch `[2, 32000]`

输入：

```python
x.shape = [2, 32000]
lengths.shape = [2]
lengths = [32000, 32000]
```

形状总览：

| 阶段 | 输出形状 |
|---|---|
| 输入波形 | `[2, 32000]` |
| WavLM conv frontend | `[2, 512, 99]` |
| 转置 + layer norm | `[2, 99, 512]` |
| post extract proj | `[2, 99, 1024]` |
| hidden states 堆叠 | `[25, 2, 99, 1024]` |
| hidden-state 加权融合 | `[2, 1024, 99]` |
| ECAPA layer1 | `[2, 512, 99]` |
| ECAPA layer2 | `[2, 512, 99]` |
| ECAPA layer3 | `[2, 512, 99]` |
| ECAPA layer4 | `[2, 512, 99]` |
| 拼接 `out2/out3/out4` | `[2, 1536, 99]` |
| attentive stats pooling | `[2, 3072]` |
| 最终 embedding | `[2, 256]` |

关键点：

- `25` 路 hidden states = `encoder 输入 1 路 + 24 层输出`
- `feature_weight.shape == [25]`
- 等长 batch 下，新实现与老的耦合模型仍然保持数值级一致
- 当前实测在老模型对齐上，`get_feat` 误差约 `4e-5 ~ 7e-5`，最终 `embedding` 误差约 `2e-6 ~ 6e-6`

## 3. 例子 B：单条输入 `[1, 16000]`

输入：

```python
x.shape = [1, 16000]
lengths.shape = [1]
lengths = [16000]
```

形状总览：

| 阶段 | 输出形状 |
|---|---|
| 输入波形 | `[1, 16000]` |
| WavLM conv frontend | `[1, 512, 49]` |
| post extract proj | `[1, 49, 1024]` |
| hidden states 堆叠 | `[25, 1, 49, 1024]` |
| hidden-state 加权融合 | `[1, 1024, 49]` |
| 最终 embedding | `[1, 256]` |

这里的 `49` 是由 WavLM 前端卷积链按真实长度 `16000` 推出来的。

## 4. 例子 C：不等长 batch pad 到固定桶

假设三条语音长度分别为：

```python
lengths = [16000, 24000, 32000]
```

服务入口先 pad 到同一桶长：

```python
x.shape = [3, 32000]
lengths.shape = [3]
```

这时 WavLM 实际会在 bucket 长度上跑出：

```python
conv_raw.shape = [3, 512, 99]
hidden_states.shape = [25, 3, 99, 1024]
```

但每条样本的“真实有效 feature 长度”并不都是 `99`。当前实现会按卷积公式单独计算：

- `16000 -> 49`
- `24000 -> 74`
- `32000 -> 99`

于是特征级 pad mask 变成：

```python
feat_padding_mask.shape = [3, 99]
valid_lengths = [49, 74, 99]
```

然后：

- WavLM hidden states 在无效尾帧上会被清零
- `MaskedInstanceNorm1d` 只统计有效帧
- `SE_Connect` 只在有效帧上求均值
- `AttentiveStatsPool` 只在有效帧上做注意力池化
- ECAPA 每个时域块输出后都会把无效尾帧重新清零

因此当前“pad 后一起前向”的目标语义是：

- 对于 `[16000, 24000, 32000] -> pad 到 [3, 32000]`
- 每条样本的结果尽量贴近“各自单独前向，再把结果拼起来”

当前测试实测：

- `get_feat` 与单条前向的最大误差约 `7e-5`
- 最终 embedding 与单条前向的最大误差约 `5e-6`

更具体地说，在当前测试用的三条语音：

- 长度分别为 `16000 / 24000 / 32000`
- pad 到同一 bucket：`[3, 32000]`
- 对照基线是“逐条单独前向”

实测最大绝对误差如下：

| 样本 | 长度 | `get_feat` 最大绝对误差 | `embedding` 最大绝对误差 |
|---|---:|---:|---:|
| sample 0 | `16000` | `5.930662155151367e-05` | `4.500150680541992e-06` |
| sample 1 | `24000` | `5.608797073364258e-05` | `3.7997961044311523e-06` |
| sample 2 | `32000` | `6.744265556335449e-05` | `4.49642539024353e-06` |

如果把“单条基线”也先 pad 到同一个 bucket 再前向，短语音的 `get_feat` 误差还会略微下降：

| 样本 | 长度 | `get_feat` 最大绝对误差 | `embedding` 最大绝对误差 |
|---|---:|---:|---:|
| sample 0 | `16000` | `4.4912099838256836e-05` | `4.26173210144043e-06` |
| sample 1 | `24000` | `4.682503640651703e-05` | `4.246830940246582e-06` |
| sample 2 | `32000` | `6.744265556335449e-05` | `4.49642539024353e-06` |

这说明当前误差主要是数值级浮点误差，而不是 pad mask 语义错误：

- WavLM valid prefix hidden states 与单条前向相比，最大误差约 `4.8e-6`
- 在 hidden-state 融合与 mask-aware norm 之后，`get_feat` 误差会被放大到 `1e-5 ~ 1e-4`
- 到最终 `256` 维 embedding 时，误差又回到 `1e-6 ~ 1e-5`

## 5. 为什么现在比旧写法更适合 trace / TensorRT

当前核心路径已经去掉了最影响 trace 的输入处理方式：

- 不再对输入 tensor 做 Python `for sample in x`
- 不再基于输入 batch 组装 `List[Tensor]`
- 不再在模型内部做 `pad_sequence`
- 核心入口固定为 `(x, lengths)`

这意味着部署时可以直接采用：

1. 服务入口按长度分桶
2. pad 到固定 `T_bucket`
3. 调用 `model(x, lengths)`
4. 后续再按桶做 trace / export / TensorRT engine

当前测试已经覆盖：

- 等长输入与老实现的一致性
- 不等长 pad batch 与单条前向的一致性
- `torch.jit.trace(model, (x, lengths))`
- `torch.export.export(model, (x, lengths))`
- ONNXRuntime 与 eager 的 hidden states / embedding 数值对齐
- ONNX 导出后的静态 TensorRT engine 构建

## 6. ONNX attention 偏差定位与修复

这次修复的核心不是 ECAPA，也不是 `MaskedInstanceNorm1d`，而是 WavLM attention 在 relative position bias 场景下存在“快慢路径语义分叉”。

### 6.1 修复前的现象

修复前，用代表性输入：

```python
x.shape = [2, 16000]
lengths = [16000, 12000]
```

曾测得：

- `feature_extract()['default']` 与 ONNX 基本一致，最大误差约 `3e-6`
- `feature_extract()['hidden_states']` 与 ONNX 偏差明显，最大误差约 `0.08`
- 最终 `embedding` 与 ONNX 偏差约 `0.08 ~ 0.12`
- `fast eager vs slow/export path` 的 hidden-state 最大误差约 `0.05 ~ 0.065`

这说明问题不是最终层输出，而是中间 hidden states 的 relative-attention 路径。

### 6.2 根因

原始实现里，relative-attention 实际存在两条路径：

- eager 快路径：调用 `torch.nn.functional.multi_head_attention_forward`
- 导出/分解路径：手写 `qk^T + mask + bias -> softmax -> bmm`

但两条路径在 `gru_rel_pos` 上并不完全等价：

- 快路径里的 gate 使用未投影的 `query`
- 慢路径里的 gate 使用投影后的 `q`

因此：

- 老耦合模型 eager 和单文件 eager 语义接近快路径
- ONNX 导出复现的是慢路径
- 两者在第 1 层 relative bias 进入后就开始分叉

另外，原始 bucket 公式里会对后续会被 `torch.where` 丢弃的位置先做 `log(0)`，这会在 ONNX 参考执行里触发 `log/cast` warning。这个 warning 不是主误差来源，但会污染导出链路。

### 6.3 修复方式

当前单文件实现已经改成：

- 只要 `position_bias` 参与 attention，就统一走一条显式 reference attention 路径
- 这条路径在 eager / export / ONNX 下使用同一套张量计算
- `gru_rel_pos` gate 固定使用未投影的 `query`，对齐老耦合模型 eager 语义
- `key_padding_mask` 统一转换成 additive float mask，再与 relative bias 相加
- relative-position bucket 改成 ONNX-safe 写法：
  - large branch 先 `clamp(min=max_exact)`，再 `log`
  - bucket 输出与原公式保持完全一致

### 6.4 修复后的实测数字

老耦合模型对齐测试：

| 样例 | `get_feat` 最大绝对误差 | `embedding` 最大绝对误差 |
|---|---:|---:|
| `seeded_random_batch_2x32000` | `5.608797073364258e-05` | `2.3171305656433105e-06` |
| `deterministic_sine_batch_2x32000` | `6.178021430969238e-05` | `4.231929779052734e-06` |
| `single_sine_1x16000` | `4.242360591888428e-05` | `5.0067901611328125e-06` |

内部 attention 分叉回归：

- 代表性输入 `x=[2,16000]`, `lengths=[16000,12000]`
- `fast_vs_export_hidden_max_abs_diff = 0.0`

ONNXRuntime 对 eager 的对齐：

- `onnx_hidden_max_abs_diff = 1.284480094909668e-05`
- `onnx_emb_max_abs_diff = 6.326939910650253e-06`

TensorRT 构建验证：

- 使用 `opset 18` 导出的静态 ONNX
- `trtexec` 静态 engine 构建成功

## 7. 从 PyTorch 到 TensorRT 的完整命令链

下面整理的是当前单文件模型从 `PyTorch eager -> ONNX -> ONNXRuntime -> TensorRT` 的一条已验证链路。这里优先记录“静态 batch / 静态长度桶”的部署方式，因为它是当前已经测通、也最容易做精度回归的路径。

### 7.0 推荐直接执行脚本

当前仓库已经提供了一个可直接执行的脚本：

```bash
./scripts/export_validate_trt.py --outdir artifacts/tensorrt_static_script_cpu_baseline
```

如果想同时关闭 TensorRT 的 TF32 路径：

```bash
./scripts/export_validate_trt.py \
  --outdir artifacts/tensorrt_static_script_cpu_baseline_no_tf32 \
  --no-tf32
```

脚本位置：

- `scripts/export_validate_trt.py`

脚本会自动完成：

1. 加载单文件模型和 checkpoint
2. 导出静态 ONNX
3. 用 ONNXRuntime 对比 PyTorch eager
4. 调用 `trtexec` 构建静态 TensorRT engine
5. 用 TensorRT Python runtime 真正执行推理
6. 输出并保存 `report.json`

脚本会在 `--outdir` 目录里生成：

- `wavlm_large_ecapa_tdnn_static.onnx`
- `wavlm_large_ecapa_tdnn_static.plan`
- `trtexec.log`
- `report.json`

当前脚本里的数值基线特意使用 CPU eager，而不是 GPU eager。这样可以避免把 PyTorch 自己的 GPU TF32 / kernel 数值路径混进 ONNXRuntime 与 TensorRT 的对比里。

在当前机器上，执行一次默认脚本的实测时长大致是：

- 加载模型：约 `5.94s`
- 导出 ONNX：约 `21.51s`
- ONNXRuntime 校验：约 `5.32s`
- TensorRT 构建：约 `233.41s`
- TensorRT 输出校验：约 `2.44s`
- 全流程总耗时：约 `268.61s`

也就是：

- 总体约 `4.5` 分钟
- 其中真正最耗时的是 TensorRT build，约 `3.9` 分钟

### 7.1 环境与路径约定

先进入工程目录并设置变量：

```bash
cd /inspire/ssd/project/embodied-multimodality/yukang-CZXS25240036/workspace/tts-eval

export WORKDIR=/inspire/ssd/project/embodied-multimodality/yukang-CZXS25240036/workspace/tts-eval
export CKPT=/inspire/hdd/project/embodied-multimodality/public/btjiang/tts/checkpoint/Seed-Similarity/wavlm_large_finetune.pth
export OUTDIR=$WORKDIR/artifacts/tensorrt_static
mkdir -p "$OUTDIR"
```

当前链路依赖：

- `wavlm_large_ecapa_tdnn.py`
- checkpoint: `$CKPT`
- `onnxruntime`
- `tensorrt`
- `trtexec`，当前环境路径是 `/opt/tensorrt/bin/trtexec`

### 7.2 第一步：在 PyTorch 中加载 checkpoint 并做 eager 推理

这一步的目的有两个：

- 确认 checkpoint 可以被单文件模型正确加载
- 记录一个后续要对齐的 eager 基线输出

如果只是为了跑完整链路，优先直接使用 `./scripts/export_validate_trt.py`。下面保留逐步命令，方便单独排查某一步。

命令：

```bash
python - <<'PY'
from pathlib import Path
import importlib.util
import torch

workdir = Path("/inspire/ssd/project/embodied-multimodality/yukang-CZXS25240036/workspace/tts-eval")
ckpt = Path("/inspire/hdd/project/embodied-multimodality/public/btjiang/tts/checkpoint/Seed-Similarity/wavlm_large_finetune.pth")

spec = importlib.util.spec_from_file_location("wavlm_single_file", workdir / "wavlm_large_ecapa_tdnn.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

state = torch.load(ckpt, map_location="cpu")["model"]
model = module.WavLMLargeECAPATDNN().eval()
model.load_state_dict(state, strict=True)

x = torch.randn(2, 16000)
lengths = torch.tensor([16000, 12000], dtype=torch.long)

with torch.no_grad():
    emb = model(x, lengths)

print("emb.shape =", tuple(emb.shape))
print("emb.dtype =", emb.dtype)
print("emb[0, :8] =", emb[0, :8])
PY
```

这里的输入约束是：

- `x.shape == [B, T_bucket]`
- `lengths.shape == [B]`
- `lengths` 当前固定传 `int64`

### 7.3 第二步：导出 ONNX

当前推荐直接从 checkpoint-loaded 模型导出 `opset 18` 的静态 ONNX。

命令：

```bash
python - <<'PY'
from pathlib import Path
import importlib.util
import torch

workdir = Path("/inspire/ssd/project/embodied-multimodality/yukang-CZXS25240036/workspace/tts-eval")
ckpt = Path("/inspire/hdd/project/embodied-multimodality/public/btjiang/tts/checkpoint/Seed-Similarity/wavlm_large_finetune.pth")
onnx_path = workdir / "artifacts/tensorrt_static/wavlm_large_ecapa_tdnn_static.onnx"

spec = importlib.util.spec_from_file_location("wavlm_single_file", workdir / "wavlm_large_ecapa_tdnn.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

state = torch.load(ckpt, map_location="cpu")["model"]
model = module.WavLMLargeECAPATDNN().eval()
model.load_state_dict(state, strict=True)

x = torch.randn(2, 16000)
lengths = torch.tensor([16000, 12000], dtype=torch.long)

torch.onnx.export(
    model,
    (x, lengths),
    str(onnx_path),
    input_names=["x", "lengths"],
    output_names=["emb"],
    opset_version=18,
    dynamo=True,
)

print(onnx_path)
PY
```

导出完成后，目录里通常会出现：

- `wavlm_large_ecapa_tdnn_static.onnx`
- 如果权重较大，还可能伴随 `.onnx.data`

### 7.4 第三步：先用 ONNXRuntime 校验 ONNX 是否和 PyTorch 对齐

这一步非常重要。当前流程里先看 ORT，是为了把“导出图问题”和“TensorRT backend 数值问题”分开。

命令：

```bash
python - <<'PY'
from pathlib import Path
import importlib.util
import onnxruntime as ort
import torch

workdir = Path("/inspire/ssd/project/embodied-multimodality/yukang-CZXS25240036/workspace/tts-eval")
ckpt = Path("/inspire/hdd/project/embodied-multimodality/public/btjiang/tts/checkpoint/Seed-Similarity/wavlm_large_finetune.pth")
onnx_path = workdir / "artifacts/tensorrt_static/wavlm_large_ecapa_tdnn_static.onnx"

spec = importlib.util.spec_from_file_location("wavlm_single_file", workdir / "wavlm_large_ecapa_tdnn.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

state = torch.load(ckpt, map_location="cpu")["model"]
model = module.WavLMLargeECAPATDNN().eval()
model.load_state_dict(state, strict=True)

x = torch.randn(2, 16000)
lengths = torch.tensor([16000, 12000], dtype=torch.long)

with torch.no_grad():
    eager = model(x, lengths)

sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
ort_out = torch.from_numpy(
    sess.run(None, {"x": x.numpy(), "lengths": lengths.numpy()})[0]
)

diff = (eager - ort_out).abs()
print("ort max_abs_diff =", diff.max().item())
print("ort mean_abs_diff =", diff.mean().item())
print("ort cosine_mean =", torch.nn.functional.cosine_similarity(eager, ort_out, dim=-1).mean().item())
PY
```

当前已经测到，修复后的链路里：

- `ONNXRuntime vs eager` 的 `hidden_states` 误差约 `1e-5`
- `ONNXRuntime vs eager` 的 `embedding` 误差约 `1e-6`

因此如果 ORT 这里已经明显偏大，就应先回头检查导出图，不要直接怀疑 TensorRT。

### 7.5 第四步：用 trtexec 构建静态 TensorRT engine

当前已经验证通过的构建命令是：

```bash
/opt/tensorrt/bin/trtexec \
  --onnx=$OUTDIR/wavlm_large_ecapa_tdnn_static.onnx \
  --saveEngine=$OUTDIR/wavlm_large_ecapa_tdnn_static.plan \
  --skipInference
```

说明：

- 当前这版 `trtexec` 不认 `--buildOnly`，要用 `--skipInference`
- 当前 ONNX parser 会提示 `lengths` 是 `Int64 binding`，这是预期行为
- 这一步只负责构建与反序列化 engine，不负责数值精度对比

如果想避免 TF32 带来的额外数值漂移，可以显式关闭：

```bash
/opt/tensorrt/bin/trtexec \
  --onnx=$OUTDIR/wavlm_large_ecapa_tdnn_static.onnx \
  --saveEngine=$OUTDIR/wavlm_large_ecapa_tdnn_static_no_tf32.plan \
  --skipInference \
  --noTF32
```

这里的精度含义是：

- 默认不加 `--fp16 / --bf16 / --int8` 时，整体仍然是 FP32 图
- 如果不加 `--noTF32`，TensorRT 允许部分 FP32 matmul 走 TF32 Tensor Core 路径
- 加 `--noTF32` 以后，仍是 FP32，只是禁止 TF32 内部实现

### 7.6 第五步：用 TensorRT runtime 真正跑一遍，并和 PyTorch eager 对比

构建成功不代表数值已经满足要求，必须实际执行 engine。

下面这条命令会：

- 重新加载 checkpoint-loaded PyTorch 模型
- 读取上一步构建出的 `.plan`
- 在 GPU 上跑 TensorRT inference
- 输出 `max_abs_diff / mean_abs_diff / cosine_mean`

命令：

```bash
python - <<'PY'
from pathlib import Path
import importlib.util
import tensorrt as trt
import torch

workdir = Path("/inspire/ssd/project/embodied-multimodality/yukang-CZXS25240036/workspace/tts-eval")
ckpt = Path("/inspire/hdd/project/embodied-multimodality/public/btjiang/tts/checkpoint/Seed-Similarity/wavlm_large_finetune.pth")
engine_path = workdir / "artifacts/tensorrt_static/wavlm_large_ecapa_tdnn_static.plan"

spec = importlib.util.spec_from_file_location("wavlm_single_file", workdir / "wavlm_large_ecapa_tdnn.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

state = torch.load(ckpt, map_location="cpu")["model"]
model = module.WavLMLargeECAPATDNN().eval().cuda()
model.load_state_dict(state, strict=True)

logger = trt.Logger(trt.Logger.ERROR)
runtime = trt.Runtime(logger)
with open(engine_path, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

def trt_dtype_to_torch(dtype):
    return {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int64: torch.int64,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }[dtype]

x = torch.randn(2, 16000, device="cuda", dtype=torch.float32)
lengths = torch.tensor([16000, 12000], device="cuda", dtype=torch.int64)

inputs = {"x": x.contiguous(), "lengths": lengths.contiguous()}
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    if engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
        continue
    shape = tuple(context.get_tensor_shape(name))
    if -1 in shape:
        context.set_input_shape(name, tuple(inputs[name].shape))

outputs = {}
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    if engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
        continue
    outputs[name] = torch.empty(
        tuple(context.get_tensor_shape(name)),
        dtype=trt_dtype_to_torch(engine.get_tensor_dtype(name)),
        device="cuda",
    )

for name, tensor in {**inputs, **outputs}.items():
    context.set_tensor_address(name, tensor.data_ptr())

stream = torch.cuda.current_stream().cuda_stream
ok = context.execute_async_v3(stream)
assert ok
torch.cuda.synchronize()

with torch.no_grad():
    eager = model(x, lengths).cpu()
trt_out = outputs["emb"].cpu()

diff = (eager - trt_out).abs()
print("trt max_abs_diff =", diff.max().item())
print("trt mean_abs_diff =", diff.mean().item())
print("trt cosine_mean =", torch.nn.functional.cosine_similarity(eager, trt_out, dim=-1).mean().item())
PY
```

### 7.7 当前这条链路的实测精度

用上面这套命令链，当前已经测到：

默认 TensorRT 配置，也就是允许 TF32 时：

| 输入样例 | `max_abs_diff` | `mean_abs_diff` | `cosine_mean` |
|---|---:|---:|---:|
| `random_pad`, `lengths=[16000,12000]` | `0.0015369206666946411` | `0.0003386555763427168` | `0.999993622303009` |
| `deterministic_full`, `lengths=[16000,16000]` | `0.001552291214466095` | `0.0004661862039938569` | `0.9999948740005493` |

关闭 TF32，也就是 `--noTF32` 后：

| 输入样例 | `max_abs_diff` | `mean_abs_diff` | `cosine_mean` |
|---|---:|---:|---:|
| `random_pad`, `lengths=[16000,12000]` | `0.0005144625902175903` | `0.00013451171980705112` | `0.9999988675117493` |

这说明：

- 当前 `PyTorch -> ONNX -> ONNXRuntime` 的语义已经基本对齐
- TensorRT engine 也已经能正确构建并执行
- 剩余的部署误差主要来自 TensorRT backend 的数值实现差异，而不是 ONNX 图语义错误

### 7.8 当前脚本一次默认执行的实测时长

下面这组时长来自直接执行：

```bash
./scripts/export_validate_trt.py --outdir artifacts/tensorrt_static_script_cpu_baseline
```

当前机器环境：

- GPU: `NVIDIA H200`
- 日期：`2026-04-20`

实测结果：

| 阶段 | 耗时 |
|---|---:|
| `load_model` | `5.940555337816477s` |
| `export_onnx` | `21.50637928303331s` |
| `validate_onnxruntime` | `5.31574000697583s` |
| `build_tensorrt` | `233.40632039122283s` |
| `validate_tensorrt` | `2.44150104559958s` |
| `total` | `268.61049606464803s` |

可近似理解为：

- 导出 ONNX 大约 `20 ~ 25s`
- ORT 校验大约 `5 ~ 6s`
- TensorRT build 大约 `230 ~ 240s`
- TensorRT 输出校验大约 `2 ~ 3s`
- 全流程总计大约 `4.5min`

### 7.9 当前链路的建议顺序

建议每次都按下面顺序做，而不是直接上 TensorRT：

1. 先在 PyTorch eager 下确认 checkpoint-loaded 模型输出正常
2. 导出 ONNX
3. 先用 ONNXRuntime 看和 eager 的误差
4. 再构建 TensorRT engine
5. 最后用 TensorRT runtime 做真实精度比对

如果 ORT 已经偏得很大，应先修导出图。

如果 ORT 很准，但 TensorRT 偏大，应重点排查：

- `TF32`
- TensorRT tactic / fusion
- attention 和归约算子的 backend 数值路径

## 8. 备注

- 当前文档只描述核心单文件模型；老的 `thirdparty/UniSpeech/downstreams/speaker_verification/verification.py` 仍然是两条语音分别前向的老调用方式，尚未切到新的 batch API。
