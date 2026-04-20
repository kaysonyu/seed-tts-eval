import math
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch.nn.utils.rnn import pad_sequence


ROOT = Path(__file__).resolve().parent
CHECKPOINT = Path(
    "/inspire/hdd/project/embodied-multimodality/public/btjiang/tts/checkpoint/Seed-Similarity/wavlm_large_finetune.pth"
)
OLD_MODEL_FILE = ROOT / "thirdparty/UniSpeech/downstreams/speaker_verification/models/ecapa_tdnn.py"
NEW_MODEL_FILE = ROOT / "wavlm_large_ecapa_tdnn.py"
GET_FEAT_ATOL = 1e-4
EMBED_ATOL = 1e-5
ONNX_HIDDEN_ATOL = 2e-5
ONNX_EMBED_ATOL = 1e-4


def _load_module_from_path(path: Path, module_name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def modules():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "s3prl"))

    old_module = _load_module_from_path(OLD_MODEL_FILE, "old_ecapa_tdnn")
    new_module = _load_module_from_path(NEW_MODEL_FILE, "new_wavlm_large_ecapa_tdnn")
    from s3prl.upstream.wavlm.expert import UpstreamExpert

    return old_module, new_module, UpstreamExpert


@pytest.fixture(scope="module")
def checkpoint_state():
    assert CHECKPOINT.exists(), f"Missing checkpoint: {CHECKPOINT}"
    return torch.load(CHECKPOINT, map_location="cpu")["model"]


def _build_old_and_new_models(modules, checkpoint_state):
    old_module, new_module, upstream_expert_cls = modules

    upstream_state = {
        key[len("feature_extract.model.") :]: value
        for key, value in checkpoint_state.items()
        if key.startswith("feature_extract.model.")
    }

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
        upstream_ckpt = tmp_file.name

    torch.save(
        {
            "cfg": new_module.build_wavlm_large_config(),
            "model": upstream_state,
        },
        upstream_ckpt,
    )

    def fake_hub_load(repo_or_dir, model, *args, **kwargs):
        assert model == "wavlm_large", model
        return upstream_expert_cls(upstream_ckpt)

    try:
        with patch.object(torch.hub, "load", side_effect=fake_hub_load):
            old_model = old_module.ECAPA_TDNN_SMALL(
                feat_dim=1024,
                feat_type="wavlm_large",
                config_path=None,
            )
        new_model = new_module.WavLMLargeECAPATDNN()

        old_model.load_state_dict(checkpoint_state, strict=False)
        new_model.load_state_dict(checkpoint_state, strict=True)
        old_model.eval()
        new_model.eval()
        return old_model, new_model
    finally:
        os.unlink(upstream_ckpt)


def _full_lengths(x: torch.Tensor) -> torch.Tensor:
    return torch.full((x.size(0),), x.size(1), dtype=torch.long, device=x.device)


def _run_onnxruntime(model: torch.nn.Module, x: torch.Tensor, lengths: torch.Tensor, output_names):
    ort = pytest.importorskip("onnxruntime")
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "model.onnx"
        torch.onnx.export(
            model,
            (x, lengths),
            str(onnx_path),
            input_names=["x", "lengths"],
            output_names=output_names,
            opset_version=18,
            dynamo=True,
        )
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        outputs = session.run(
            None,
            {
                "x": x.cpu().numpy(),
                "lengths": lengths.cpu().numpy(),
            },
        )
    return [torch.from_numpy(output) for output in outputs]


def _original_relative_positions_bucket(relative_positions, num_buckets: int, max_distance: int, bidirectional: bool):
    relative_buckets = 0
    if bidirectional:
        num_buckets = num_buckets // 2
        relative_buckets += (relative_positions > 0).to(torch.long) * num_buckets
        relative_positions = torch.abs(relative_positions)
    else:
        relative_positions = -torch.min(relative_positions, torch.zeros_like(relative_positions))

    max_exact = num_buckets // 2
    is_small = relative_positions < max_exact
    relative_postion_if_large = max_exact + (
        torch.log(relative_positions.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_postion_if_large = torch.min(
        relative_postion_if_large,
        torch.full_like(relative_postion_if_large, num_buckets - 1),
    )
    return relative_buckets + torch.where(is_small, relative_positions, relative_postion_if_large)


def _example_inputs():
    torch.manual_seed(0)
    seeded_random = torch.randn(2, 32000)

    sr = 16000
    t = torch.arange(0, sr * 2, dtype=torch.float32) / sr
    sine_a = 0.2 * torch.sin(2 * math.pi * 220 * t)
    sine_b = 0.15 * torch.sin(2 * math.pi * 440 * t + 0.3)
    sine_b = sine_b + 0.05 * torch.cos(2 * math.pi * 90 * t)
    deterministic_sine = torch.stack([sine_a, sine_b], dim=0)
    single_sine = sine_a[:sr].unsqueeze(0)

    return {
        "seeded_random_batch_2x32000": {
            "input": seeded_random,
            "feat_shape": (2, 1024, 99),
            "emb_shape": (2, 256),
        },
        "deterministic_sine_batch_2x32000": {
            "input": deterministic_sine,
            "feat_shape": (2, 1024, 99),
            "emb_shape": (2, 256),
        },
        "single_sine_1x16000": {
            "input": single_sine,
            "feat_shape": (1, 1024, 49),
            "emb_shape": (1, 256),
        },
    }


@pytest.mark.parametrize("example_name", list(_example_inputs().keys()))
def test_get_feat_matches_old_implementation(modules, checkpoint_state, example_name):
    old_model, new_model = _build_old_and_new_models(modules, checkpoint_state)
    example = _example_inputs()[example_name]
    x = example["input"]
    lengths = _full_lengths(x)

    with torch.no_grad():
        old_feat = old_model.get_feat(x)
        new_feat = new_model.get_feat(x, lengths)

    assert old_feat.shape == new_feat.shape == example["feat_shape"]
    torch.testing.assert_close(old_feat, new_feat, rtol=0.0, atol=GET_FEAT_ATOL)


@pytest.mark.parametrize("example_name", list(_example_inputs().keys()))
def test_forward_matches_old_implementation(modules, checkpoint_state, example_name):
    old_model, new_model = _build_old_and_new_models(modules, checkpoint_state)
    example = _example_inputs()[example_name]
    x = example["input"]
    lengths = _full_lengths(x)

    with torch.no_grad():
        old_emb = old_model(x)
        new_emb = new_model(x, lengths)

    assert old_emb.shape == new_emb.shape == example["emb_shape"]
    torch.testing.assert_close(old_emb, new_emb, rtol=0.0, atol=EMBED_ATOL)


def test_padded_batch_matches_single_utterance_forward(modules, checkpoint_state):
    _, new_module, _ = modules
    model = new_module.WavLMLargeECAPATDNN()
    model.load_state_dict(checkpoint_state, strict=True)
    model.eval()

    sr = 16000
    t_long = torch.arange(0, sr * 2, dtype=torch.float32) / sr
    t_mid = torch.arange(0, int(sr * 1.5), dtype=torch.float32) / sr
    t_short = torch.arange(0, sr, dtype=torch.float32) / sr
    wav_a = 0.20 * torch.sin(2 * math.pi * 220 * t_short)
    wav_b = 0.18 * torch.sin(2 * math.pi * 330 * t_mid + 0.2)
    wav_c = 0.15 * torch.sin(2 * math.pi * 440 * t_long + 0.5)
    wav_c = wav_c + 0.04 * torch.cos(2 * math.pi * 120 * t_long)
    wavs = [wav_a, wav_b, wav_c]

    padded = pad_sequence(wavs, batch_first=True)
    lengths = torch.tensor([wav.numel() for wav in wavs], dtype=torch.long)

    with torch.no_grad():
        batch_feat, batch_mask = model.get_feat(padded, lengths, return_padding_mask=True)
        batch_emb = model(padded, lengths)
        single_feats = []
        single_embs = []
        for wav in wavs:
            x = wav.unsqueeze(0)
            x_lengths = torch.tensor([wav.numel()], dtype=torch.long)
            single_feats.append(model.get_feat(x, x_lengths)[0])
            single_embs.append(model(x, x_lengths)[0])

    single_emb = torch.stack(single_embs, dim=0)
    for idx, single_feat in enumerate(single_feats):
        valid_frames = single_feat.size(-1)
        torch.testing.assert_close(batch_feat[idx, :, :valid_frames], single_feat, rtol=0.0, atol=1e-4)
        assert torch.count_nonzero(batch_feat[idx, :, valid_frames:]) == 0
        assert bool(batch_mask[idx, valid_frames:].all().item())
    torch.testing.assert_close(batch_emb, single_emb, rtol=0.0, atol=1e-5)


def test_trace_and_export_accept_tensor_batch_with_lengths(modules):
    _, new_module, _ = modules
    model = new_module.WavLMLargeECAPATDNN().eval()

    x = torch.randn(2, 16000)
    lengths = torch.tensor([16000, 12000], dtype=torch.long)
    traced = torch.jit.trace(model, (x, lengths), strict=True)

    other_x = torch.randn(3, 16000)
    other_lengths = torch.tensor([16000, 9000, 15000], dtype=torch.long)
    with torch.no_grad():
        eager_out = model(other_x, other_lengths)
        traced_out = traced(other_x, other_lengths)

    torch.testing.assert_close(traced_out, eager_out, rtol=0.0, atol=1e-5)
    exported = torch.export.export(model, (x, lengths))
    assert exported is not None


def test_relative_bucket_matches_original_formula(modules):
    _, new_module, _ = modules
    attn = new_module.MultiheadAttention(
        embed_dim=1024,
        num_heads=16,
        self_attention=True,
        has_relative_attention_bias=True,
        num_buckets=320,
        max_distance=800,
        gru_rel_pos=True,
    )
    relative_positions = torch.arange(-1000, 1001)

    safe_bucket = attn._relative_positions_bucket(relative_positions, bidirectional=True)
    original_bucket = _original_relative_positions_bucket(
        relative_positions,
        num_buckets=attn.num_buckets,
        max_distance=attn.max_distance,
        bidirectional=True,
    )

    assert torch.equal(safe_bucket, original_bucket)


def test_relative_attention_fast_and_export_paths_match(modules, checkpoint_state):
    _, new_module, _ = modules
    model = new_module.WavLMLargeECAPATDNN().eval()
    model.load_state_dict(checkpoint_state, strict=True)

    x = torch.randn(2, 16000)
    lengths = torch.tensor([16000, 12000], dtype=torch.long)

    with torch.no_grad():
        eager_hidden = model.feature_extract(x, lengths)["hidden_states"]

    with patch.object(torch.compiler, "is_compiling", return_value=True):
        with torch.no_grad():
            export_hidden = model.feature_extract(x, lengths)["hidden_states"]

    torch.testing.assert_close(eager_hidden, export_hidden, rtol=0.0, atol=ONNX_HIDDEN_ATOL)


def test_onnxruntime_matches_eager_hidden_states_and_embeddings(modules, checkpoint_state):
    _, new_module, _ = modules
    model = new_module.WavLMLargeECAPATDNN().eval()
    model.load_state_dict(checkpoint_state, strict=True)

    class HiddenStatesWrapper(torch.nn.Module):
        def __init__(self, wrapped_model):
            super().__init__()
            self.wrapped_model = wrapped_model

        def forward(self, x, lengths):
            return self.wrapped_model.feature_extract(x, lengths)["hidden_states"]

    x = torch.randn(2, 16000)
    lengths = torch.tensor([16000, 12000], dtype=torch.long)

    with torch.no_grad():
        eager_hidden = model.feature_extract(x, lengths)["hidden_states"]
        eager_emb = model(x, lengths)

    onnx_hidden = _run_onnxruntime(
        HiddenStatesWrapper(model).eval(),
        x,
        lengths,
        output_names=["hidden_states"],
    )[0]
    onnx_emb = _run_onnxruntime(model, x, lengths, output_names=["emb"])[0]

    torch.testing.assert_close(eager_hidden, onnx_hidden, rtol=0.0, atol=ONNX_HIDDEN_ATOL)
    torch.testing.assert_close(eager_emb, onnx_emb, rtol=0.0, atol=ONNX_EMBED_ATOL)


def test_lengths_validation_rejects_invalid_lengths(modules):
    _, new_module, _ = modules
    model = new_module.WavLMLargeECAPATDNN().eval()
    x = torch.randn(2, 16000)

    with pytest.raises(ValueError, match="positive"):
        model(x, torch.tensor([16000, 0], dtype=torch.long))

    with pytest.raises(ValueError, match="less than or equal"):
        model(x, torch.tensor([16000, 16001], dtype=torch.long))
