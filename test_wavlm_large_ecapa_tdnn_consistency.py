import math
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch


ROOT = Path(__file__).resolve().parent
CHECKPOINT = Path(
    "/inspire/hdd/project/embodied-multimodality/public/btjiang/tts/checkpoint/Seed-Similarity/wavlm_large_finetune.pth"
)
OLD_MODEL_FILE = ROOT / "thirdparty/UniSpeech/downstreams/speaker_verification/models/ecapa_tdnn.py"
NEW_MODEL_FILE = ROOT / "wavlm_large_ecapa_tdnn.py"


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

    with torch.no_grad():
        old_feat = old_model.get_feat(x)
        new_feat = new_model.get_feat(x)

    assert old_feat.shape == new_feat.shape == example["feat_shape"]
    assert torch.equal(old_feat, new_feat), (
        f"get_feat mismatch for {example_name}: "
        f"max_abs_diff={(old_feat - new_feat).abs().max().item()}"
    )


@pytest.mark.parametrize("example_name", list(_example_inputs().keys()))
def test_forward_matches_old_implementation(modules, checkpoint_state, example_name):
    old_model, new_model = _build_old_and_new_models(modules, checkpoint_state)
    example = _example_inputs()[example_name]
    x = example["input"]

    with torch.no_grad():
        old_emb = old_model(x)
        new_emb = new_model(x)

    assert old_emb.shape == new_emb.shape == example["emb_shape"]
    assert torch.equal(old_emb, new_emb), (
        f"forward mismatch for {example_name}: "
        f"max_abs_diff={(old_emb - new_emb).abs().max().item()}"
    )
