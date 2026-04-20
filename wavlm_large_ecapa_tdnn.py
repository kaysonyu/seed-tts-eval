from __future__ import annotations

import logging
import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LayerNorm, Parameter
from torch.nn.utils.rnn import pad_sequence


logger = logging.getLogger(__name__)


def build_wavlm_large_config() -> Dict[str, object]:
    return {
        "extractor_mode": "layer_norm",
        "encoder_layers": 24,
        "encoder_embed_dim": 1024,
        "encoder_ffn_embed_dim": 4096,
        "encoder_attention_heads": 16,
        "activation_fn": "gelu",
        "layer_norm_first": False,
        "conv_feature_layers": "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        "conv_bias": False,
        "feature_grad_mult": 1.0,
        "normalize": False,
        "dropout": 0.1,
        "attention_dropout": 0.1,
        "activation_dropout": 0.0,
        "encoder_layerdrop": 0.0,
        "dropout_input": 0.0,
        "dropout_features": 0.0,
        "mask_length": 10,
        "mask_prob": 0.65,
        "mask_selection": "static",
        "mask_other": 0.0,
        "no_mask_overlap": False,
        "mask_min_space": 1,
        "mask_channel_length": 10,
        "mask_channel_prob": 0.0,
        "mask_channel_selection": "static",
        "mask_channel_other": 0.0,
        "no_mask_channel_overlap": False,
        "mask_channel_min_space": 1,
        "conv_pos": 128,
        "conv_pos_groups": 16,
        "relative_position_embedding": True,
        "num_buckets": 320,
        "max_distance": 800,
        "gru_rel_pos": True,
    }


# Source adapted from s3prl/s3prl/upstream/wavlm/modules.py
class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2, -1)


# Source adapted from s3prl/s3prl/upstream/wavlm/modules.py
class Fp32LayerNorm(nn.LayerNorm):
    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


# Source adapted from s3prl/s3prl/upstream/wavlm/modules.py
class Fp32GroupNorm(nn.GroupNorm):
    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


# Source adapted from s3prl/s3prl/upstream/wavlm/modules.py
class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.new(x)

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


# Source adapted from s3prl/s3prl/upstream/wavlm/modules.py
class SamePad(nn.Module):
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


# Source adapted from s3prl/s3prl/upstream/wavlm/modules.py
class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        return x * self.act(x)


# Source adapted from s3prl/s3prl/upstream/wavlm/modules.py
class GLU_Linear(nn.Module):
    def __init__(self, input_dim, output_dim, glu_type="sigmoid", bias_in_glu=True):
        super().__init__()

        self.glu_type = glu_type
        self.output_dim = output_dim

        if glu_type == "sigmoid":
            self.glu_act = torch.nn.Sigmoid()
        elif glu_type == "swish":
            self.glu_act = Swish()
        elif glu_type == "relu":
            self.glu_act = torch.nn.ReLU()
        elif glu_type == "gelu":
            self.glu_act = torch.nn.GELU()

        if bias_in_glu:
            self.linear = nn.Linear(input_dim, output_dim * 2, True)
        else:
            self.linear = nn.Linear(input_dim, output_dim * 2, False)

    def forward(self, x):
        x = self.linear(x)

        if self.glu_type == "bilinear":
            x = (
                x[:, :, 0 : self.output_dim]
                * x[:, :, self.output_dim : self.output_dim * 2]
            )
        else:
            x = x[:, :, 0 : self.output_dim] * self.glu_act(
                x[:, :, self.output_dim : self.output_dim * 2]
            )
        return x


# Source adapted from s3prl/s3prl/upstream/wavlm/modules.py
def gelu_accurate(x):
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return (
        0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))
    )


# Source adapted from s3prl/s3prl/upstream/wavlm/modules.py
def gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x.float()).type_as(x)


# Source adapted from s3prl/s3prl/upstream/wavlm/modules.py
def get_activation_fn(activation: str):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return gelu
    if activation == "gelu_fast":
        warnings.warn("--activation-fn=gelu_fast has been renamed to gelu_accurate")
        return gelu_accurate
    if activation == "gelu_accurate":
        return gelu_accurate
    if activation == "tanh":
        return torch.tanh
    if activation == "linear":
        return lambda x: x
    if activation == "glu":
        return lambda x: x
    raise RuntimeError(f"--activation-fn {activation} not supported")


# Source adapted from s3prl/s3prl/upstream/wavlm/modules.py
def init_bert_params(module):
    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


# Source adapted from s3prl/s3prl/upstream/wavlm/modules.py
def quant_noise(module, p, block_size):
    if p <= 0:
        return module

    assert isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d))
    is_conv = module.weight.ndim == 4

    if not is_conv:
        assert module.weight.size(1) % block_size == 0
    else:
        if module.kernel_size == (1, 1):
            assert module.in_channels % block_size == 0
        else:
            k = module.kernel_size[0] * module.kernel_size[1]
            assert k % block_size == 0

    def _forward_pre_hook(mod, input):
        if not mod.training:
            return

        weight = mod.weight
        if not is_conv:
            in_features = weight.size(1)
            out_features = weight.size(0)
            mask = torch.zeros(
                in_features // block_size * out_features,
                device=weight.device,
            )
            mask.bernoulli_(p)
            mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)
        else:
            if mod.kernel_size == (1, 1):
                mask = torch.zeros(
                    int(mod.in_channels // block_size * mod.out_channels),
                    device=weight.device,
                )
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, mod.in_channels)
            else:
                mask = torch.zeros(weight.size(0), weight.size(1), device=weight.device)
                mask.bernoulli_(p)
                mask = (
                    mask.unsqueeze(2)
                    .unsqueeze(3)
                    .repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])
                )

        mask = mask.to(torch.bool)
        mod.weight.data = (1 / (1 - p)) * weight.masked_fill(mask, 0)

    module.register_forward_pre_hook(_forward_pre_hook)
    return module


# Source adapted from s3prl/s3prl/upstream/wavlm/modules.py
class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
        has_relative_attention_bias=False,
        num_buckets=32,
        max_distance=128,
        gru_rel_pos=False,
        rescale_init=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout)

        self.has_relative_attention_bias = has_relative_attention_bias
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

        self.head_dim = embed_dim // num_heads
        self.q_head_dim = self.head_dim
        self.k_head_dim = self.head_dim
        assert self.head_dim * num_heads == self.embed_dim
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert not self.self_attention or self.qkv_same_dim

        k_bias = not rescale_init
        self.k_proj = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=k_bias),
            q_noise,
            qn_block_size,
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias),
            q_noise,
            qn_block_size,
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias),
            q_noise,
            qn_block_size,
        )
        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias),
            q_noise,
            qn_block_size,
        )

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self.gru_rel_pos = gru_rel_pos
        if self.gru_rel_pos:
            self.grep_linear = nn.Linear(self.q_head_dim, 8)
            self.grep_a = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)
        if self.has_relative_attention_bias:
            nn.init.xavier_normal_(self.relative_attention_bias.weight)

    def _relative_positions_bucket(self, relative_positions, bidirectional=True):
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        relative_buckets = 0

        if bidirectional:
            num_buckets = num_buckets // 2
            relative_buckets += (relative_positions > 0).to(torch.long) * num_buckets
            relative_positions = torch.abs(relative_positions)
        else:
            relative_positions = -torch.min(
                relative_positions, torch.zeros_like(relative_positions)
            )

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
        relative_buckets += torch.where(
            is_small, relative_positions, relative_postion_if_large
        )
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_positions_bucket(
            relative_position,
            bidirectional=True,
        )
        relative_position_bucket = relative_position_bucket.to(
            self.relative_attention_bias.weight.device
        )
        values = self.relative_attention_bias(relative_position_bucket)
        return values.permute([2, 0, 1])

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
        position_bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        if need_head_weights:
            need_weights = True

        is_tpu = query.device.type == "xla"
        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim

        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        if self.has_relative_attention_bias and position_bias is None:
            position_bias = self.compute_bias(tgt_len, src_len)
            position_bias = (
                position_bias.unsqueeze(0)
                .repeat(bsz, 1, 1, 1)
                .view(bsz * self.num_heads, tgt_len, src_len)
            )

        if (
            not is_tpu
            and incremental_state is None
            and not static_kv
            and not torch.jit.is_scripting()
            and self.q_head_dim == self.head_dim
        ):
            assert key is not None and value is not None
            assert attn_mask is None

            attn_mask_rel_pos = None
            if position_bias is not None:
                attn_mask_rel_pos = position_bias
                if self.gru_rel_pos:
                    query_layer = query.transpose(0, 1)
                    new_x_shape = query_layer.size()[:-1] + (self.num_heads, -1)
                    query_layer = query_layer.view(*new_x_shape).permute(0, 2, 1, 3)
                    _B, _H, _L, __ = query_layer.size()
                    gate_a, gate_b = torch.sigmoid(
                        self.grep_linear(query_layer)
                        .view(_B, _H, _L, 2, 4)
                        .sum(-1, keepdim=False)
                    ).chunk(2, dim=-1)
                    gate_a_1 = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
                    attn_mask_rel_pos = (
                        gate_a_1.view(bsz * self.num_heads, -1, 1) * position_bias
                    )
                attn_mask_rel_pos = attn_mask_rel_pos.view((-1, tgt_len, tgt_len))

            x, attn = F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                torch.empty([0]),
                torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout_module.p,
                self.out_proj.weight,
                self.out_proj.bias,
                self.training,
                key_padding_mask,
                need_weights,
                attn_mask_rel_pos,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
            )
            return x, attn, position_bias

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state and static_kv:
                assert self.encoder_decoder_attention and not self.self_attention
                key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)],
                    dim=1,
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.q_head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.k_head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            if "prev_key" in saved_state:
                prev_key = saved_state["prev_key"]
                assert prev_key is not None
                prev_key = prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                k = prev_key if static_kv else torch.cat([prev_key, k], dim=1)
                src_len = k.size(1)
            if "prev_value" in saved_state:
                prev_value = saved_state["prev_value"]
                assert prev_value is not None
                prev_value = prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                v = prev_value if static_kv else torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask = saved_state.get("prev_key_padding_mask")
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )
            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            assert incremental_state is not None
            self._set_input_buffer(incremental_state, saved_state)

        assert k is not None
        assert k.size(1) == src_len

        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)],
                    dim=1,
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask),
                    ],
                    dim=1,
                )

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_weights += attn_mask.unsqueeze(0)

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not is_tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v, position_bias

        if position_bias is not None:
            if self.gru_rel_pos == 1:
                query_layer = q.view(bsz, self.num_heads, tgt_len, self.q_head_dim)
                _B, _H, _L, __ = query_layer.size()
                gate_a, gate_b = torch.sigmoid(
                    self.grep_linear(query_layer)
                    .view(_B, _H, _L, 2, 4)
                    .sum(-1, keepdim=False)
                ).chunk(2, dim=-1)
                gate_a_1 = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
                position_bias = gate_a_1.view(bsz * self.num_heads, -1, 1) * position_bias

            position_bias = position_bias.view(attn_weights.size())
            attn_weights = attn_weights + position_bias

        attn_weights_float = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout_module(attn_weights_float.type_as(attn_weights))
        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        attn_weights_out: Optional[Tensor] = None
        if need_weights:
            attn_weights_out = attn_weights_float.view(
                bsz,
                self.num_heads,
                tgt_len,
                src_len,
            ).transpose(1, 0)
            if not need_head_weights:
                attn_weights_out = attn_weights_out.mean(dim=0)

        return attn, attn_weights_out, position_bias

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        if prev_key_padding_mask is not None and static_kv:
            return prev_key_padding_mask
        if prev_key_padding_mask is not None and key_padding_mask is not None:
            return torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()],
                dim=1,
            )
        if prev_key_padding_mask is not None:
            if src_len > prev_key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - prev_key_padding_mask.size(1)),
                    device=prev_key_padding_mask.device,
                )
                return torch.cat([prev_key_padding_mask.float(), filler.float()], dim=1)
            return prev_key_padding_mask.float()
        if key_padding_mask is not None:
            if src_len > key_padding_mask.size(1):
                filler = torch.zeros(
                    (batch_size, src_len - key_padding_mask.size(1)),
                    device=key_padding_mask.device,
                )
                return torch.cat([filler.float(), key_padding_mask.float()], dim=1)
            return key_padding_mask.float()
        return prev_key_padding_mask

    def get_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        key: str,
    ):
        if incremental_state is None:
            return None
        return incremental_state.get(key)

    def set_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        key: str,
        value: Dict[str, Optional[Tensor]],
    ):
        incremental_state[key] = value
        return incremental_state

    def _get_input_buffer(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        return {}

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights


# Source adapted from s3prl/s3prl/upstream/wavlm/WavLM.py
def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
) -> np.ndarray:
    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)
    all_num_mask = int(mask_prob * all_sz / float(mask_length) + np.random.rand())
    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(mask_prob * sz / float(mask_length) + np.random.rand())
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = np.random.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = np.random.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = np.random.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception(f"unknown mask selection {mask_type}")

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = np.random.randint(s, e - length)
                mask_idc.extend(span_start + j for j in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - keep_length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in parts),
                    np.int64,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = np.random.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1

            mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)
            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    min_len = min(len(m) for m in mask_idcs)
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True

    return mask


# Source adapted from s3prl/s3prl/upstream/wavlm/WavLM.py
class WavLMConfig:
    def __init__(self, cfg=None):
        self.extractor_mode: str = "default"
        self.encoder_layers: int = 12
        self.encoder_embed_dim: int = 768
        self.encoder_ffn_embed_dim: int = 3072
        self.encoder_attention_heads: int = 12
        self.activation_fn: str = "gelu"
        self.layer_norm_first: bool = False
        self.conv_feature_layers: str = "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2"
        self.conv_bias: bool = False
        self.feature_grad_mult: float = 1.0
        self.normalize: bool = False
        self.dropout: float = 0.1
        self.attention_dropout: float = 0.1
        self.activation_dropout: float = 0.0
        self.encoder_layerdrop: float = 0.0
        self.dropout_input: float = 0.0
        self.dropout_features: float = 0.0
        self.mask_length: int = 10
        self.mask_prob: float = 0.65
        self.mask_selection: str = "static"
        self.mask_other: float = 0.0
        self.no_mask_overlap: bool = False
        self.mask_min_space: int = 1
        self.mask_channel_length: int = 10
        self.mask_channel_prob: float = 0.0
        self.mask_channel_selection: str = "static"
        self.mask_channel_other: float = 0.0
        self.no_mask_channel_overlap: bool = False
        self.mask_channel_min_space: int = 1
        self.conv_pos: int = 128
        self.conv_pos_groups: int = 16
        self.relative_position_embedding: bool = False
        self.num_buckets: int = 320
        self.max_distance: int = 1280
        self.gru_rel_pos: bool = False
        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict):
        self.__dict__.update(cfg)


# Source adapted from s3prl/s3prl/upstream/wavlm/WavLM.py
class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
        conv_type: str = "default",
    ):
        super().__init__()
        assert mode in {"default", "layer_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert not (is_layer_norm and is_group_norm)

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(n_out, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            if is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(n_out, n_out, affine=True),
                    nn.GELU(),
                )
            return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        self.conv_type = conv_type
        self.conv_layers = nn.ModuleList()
        if self.conv_type == "default":
            in_d = 1
            for i, cl in enumerate(conv_layers):
                dim, k, stride = cl
                self.conv_layers.append(
                    block(
                        in_d,
                        dim,
                        k,
                        stride,
                        is_layer_norm=mode == "layer_norm",
                        is_group_norm=mode == "default" and i == 0,
                        conv_bias=conv_bias,
                    )
                )
                in_d = dim
        else:
            raise ValueError(f"Unsupported conv_type: {conv_type}")

    def forward(self, x, mask=None):
        x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)
        return x


# Source adapted from s3prl/s3prl/upstream/wavlm/WavLM.py
class TransformerSentenceEncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
        has_relative_attention_bias: bool = False,
        num_buckets: int = 0,
        max_distance: int = 0,
        rescale_init: bool = False,
        gru_rel_pos: bool = False,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_name = activation_fn
        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            has_relative_attention_bias=has_relative_attention_bias,
            num_buckets=num_buckets,
            max_distance=max_distance,
            rescale_init=rescale_init,
            gru_rel_pos=gru_rel_pos,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.layer_norm_first = layer_norm_first
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)

        if self.activation_name == "glu":
            self.fc1 = GLU_Linear(self.embedding_dim, ffn_embedding_dim, "swish")
        else:
            self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)
        self.final_layer_norm = LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        pos_bias=None,
    ):
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn, pos_bias = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
                position_bias=pos_bias,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.fc1(x) if self.activation_name == "glu" else self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn, pos_bias = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
                attn_mask=self_attn_mask,
                position_bias=pos_bias,
            )
            x = self.dropout1(x)
            x = residual + x
            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.fc1(x) if self.activation_name == "glu" else self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, attn, pos_bias


# Source adapted from s3prl/s3prl/upstream/wavlm/WavLM.py
class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim

        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=args.conv_pos,
            padding=args.conv_pos // 2,
            groups=args.conv_pos_groups,
        )
        std = math.sqrt((4 * (1.0 - 0.0)) / (args.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)
        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())

        if hasattr(args, "relative_position_embedding"):
            self.relative_position_embedding = args.relative_position_embedding
            self.num_buckets = args.num_buckets
            self.max_distance = args.max_distance
        else:
            self.relative_position_embedding = False
            self.num_buckets = 0
            self.max_distance = 0

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                    num_attention_heads=args.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    activation_fn=args.activation_fn,
                    layer_norm_first=args.layer_norm_first,
                    has_relative_attention_bias=(self.relative_position_embedding and i == 0),
                    num_buckets=self.num_buckets,
                    max_distance=self.max_distance,
                    gru_rel_pos=args.gru_rel_pos,
                )
                for i in range(args.encoder_layers)
            ]
        )
        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop
        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None, streaming_mask=None, layer=None):
        x, layer_results = self.extract_features(x, padding_mask, streaming_mask, layer)
        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)
        return x, layer_results

    def extract_features(self, x, padding_mask=None, streaming_mask=None, tgt_layer=None):
        if padding_mask is not None:
            x[padding_mask] = 0

        x_conv = self.pos_conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0, 1)

        layer_results = []
        z = None
        if tgt_layer is not None:
            layer_results.append((x, z))
        r = None
        pos_bias = None
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z, pos_bias = layer(
                    x,
                    self_attn_padding_mask=padding_mask,
                    need_weights=False,
                    self_attn_mask=streaming_mask,
                    pos_bias=pos_bias,
                )
            if tgt_layer is not None:
                layer_results.append((x, z))
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        x = x.transpose(0, 1)
        return x, layer_results


# Source adapted from s3prl/s3prl/upstream/wavlm/WavLM.py
class WavLM(nn.Module):
    def __init__(self, cfg: WavLMConfig) -> None:
        super().__init__()
        logger.info("WavLM Config: %s", cfg.__dict__)
        self.cfg = cfg
        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )
        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space
        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space
        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)
        self.feature_grad_mult = cfg.feature_grad_mult
        self.mask_emb = nn.Parameter(torch.FloatTensor(cfg.encoder_embed_dim).uniform_())
        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)

    def apply_mask(self, x, padding_mask):
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def forward_padding_mask(self, features: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        return padding_mask.all(-1)

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
        ret_layer_results: bool = False,
    ):
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)

        features = self.layer_norm(features.transpose(1, 2))
        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        x, mask_indices = self.apply_mask(features, padding_mask) if mask else (features, None)
        del mask_indices

        x, layer_results = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1,
        )
        res = {
            "x": x,
            "padding_mask": padding_mask,
            "features": features,
            "layer_results": layer_results,
        }
        feature = res["features"] if ret_conv else res["x"]
        if ret_layer_results:
            feature = (feature, res["layer_results"])
        return feature, res["padding_mask"]


class SingleFileWavLMUpstream(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = WavLM(WavLMConfig(build_wavlm_large_config()))
        self.model.feature_grad_mult = 0.0
        self.model.encoder.layerdrop = 0.0

    def forward(self, wavs: List[Tensor]):
        if self.model.cfg.normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)
        (features, layer_results), feat_padding_mask = self.model.extract_features(
            padded_wav,
            padding_mask=wav_padding_mask,
            mask=False,
            output_layer=self.model.cfg.encoder_layers,
            ret_layer_results=True,
        )
        hidden_states = [x.transpose(0, 1) for x, _ in layer_results]
        return {
            "default": features,
            "hidden_states": hidden_states,
            "last_hidden_state": hidden_states[-1],
            "padding_mask": feat_padding_mask,
        }


# Source adapted from thirdparty/UniSpeech/downstreams/speaker_verification/models/ecapa_tdnn.py
class Res2Conv1dReluBn(nn.Module):
    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, scale=4):
        super().__init__()
        assert channels % scale == 0, f"{channels} % {scale} != 0"
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias)
                for _ in range(self.nums)
            ]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(self.nums)])

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            sp = spx[i] if i == 0 else sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.bns[i](F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        return torch.cat(out, dim=1)


# Source adapted from thirdparty/UniSpeech/downstreams/speaker_verification/models/ecapa_tdnn.py
class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


# Source adapted from thirdparty/UniSpeech/downstreams/speaker_verification/models/ecapa_tdnn.py
class SE_Connect(nn.Module):
    def __init__(self, channels, se_bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        return x * out.unsqueeze(2)


# Source adapted from thirdparty/UniSpeech/downstreams/speaker_verification/models/ecapa_tdnn.py
class SE_Res2Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, scale, se_bottleneck_dim):
        super().__init__()
        self.Conv1dReluBn1 = Conv1dReluBn(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.Res2Conv1dReluBn = Res2Conv1dReluBn(out_channels, kernel_size, stride, padding, dilation, scale=scale)
        self.Conv1dReluBn2 = Conv1dReluBn(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.SE_Connect = SE_Connect(out_channels, se_bottleneck_dim)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.shortcut(x) if self.shortcut else x
        x = self.Conv1dReluBn1(x)
        x = self.Res2Conv1dReluBn(x)
        x = self.Conv1dReluBn2(x)
        x = self.SE_Connect(x)
        return x + residual


# Source adapted from thirdparty/UniSpeech/downstreams/speaker_verification/models/ecapa_tdnn.py
class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, attention_channels=128, global_context_att=False):
        super().__init__()
        self.global_context_att = global_context_att
        if global_context_att:
            self.linear1 = nn.Conv1d(in_dim * 3, attention_channels, kernel_size=1)
        else:
            self.linear1 = nn.Conv1d(in_dim, attention_channels, kernel_size=1)
        self.linear2 = nn.Conv1d(attention_channels, in_dim, kernel_size=1)

    def forward(self, x):
        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-10).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x

        alpha = torch.tanh(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * (x ** 2), dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


class _CheckpointCompatibilityLossCalculator(nn.Module):
    def __init__(self, input_dim=256, output_dim=5994):
        super().__init__()
        # This head is not used during SIM inference. It only exists so the
        # released finetuned checkpoint loads strictly without dropping keys.
        self.projection = nn.Linear(input_dim, output_dim, bias=False)


class WavLMLargeECAPATDNN(nn.Module):
    def __init__(
        self,
        feat_dim=1024,
        channels=512,
        emb_dim=256,
        global_context_att=False,
        sr=16000,
        feature_selection="hidden_states",
        update_extract=False,
    ):
        super().__init__()
        self.feat_type = "wavlm_large"
        self.feature_selection = feature_selection
        self.update_extract = update_extract
        self.sr = sr

        self.feature_extract = SingleFileWavLMUpstream()
        if len(self.feature_extract.model.encoder.layers) == 24 and hasattr(
            self.feature_extract.model.encoder.layers[23].self_attn,
            "fp32_attention",
        ):
            self.feature_extract.model.encoder.layers[23].self_attn.fp32_attention = False
        if len(self.feature_extract.model.encoder.layers) == 24 and hasattr(
            self.feature_extract.model.encoder.layers[11].self_attn,
            "fp32_attention",
        ):
            self.feature_extract.model.encoder.layers[11].self_attn.fp32_attention = False

        self.feat_num = self.get_feat_num()
        self.feature_weight = nn.Parameter(torch.zeros(self.feat_num))

        freeze_list = ["final_proj", "label_embs_concat", "mask_emb", "project_q", "quantizer"]
        for name, param in self.feature_extract.named_parameters():
            for freeze_val in freeze_list:
                if freeze_val in name:
                    param.requires_grad = False
                    break

        if not self.update_extract:
            for param in self.feature_extract.parameters():
                param.requires_grad = False

        self.loss_calculator = _CheckpointCompatibilityLossCalculator(
            input_dim=emb_dim,
            output_dim=5994,
        )
        self.instance_norm = nn.InstanceNorm1d(feat_dim)
        self.channels = [channels] * 4 + [1536]
        self.layer1 = Conv1dReluBn(feat_dim, self.channels[0], kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(
            self.channels[0],
            self.channels[1],
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2,
            scale=8,
            se_bottleneck_dim=128,
        )
        self.layer3 = SE_Res2Block(
            self.channels[1],
            self.channels[2],
            kernel_size=3,
            stride=1,
            padding=3,
            dilation=3,
            scale=8,
            se_bottleneck_dim=128,
        )
        self.layer4 = SE_Res2Block(
            self.channels[2],
            self.channels[3],
            kernel_size=3,
            stride=1,
            padding=4,
            dilation=4,
            scale=8,
            se_bottleneck_dim=128,
        )
        self.conv = nn.Conv1d(channels * 3, self.channels[-1], kernel_size=1)
        self.pooling = AttentiveStatsPool(
            self.channels[-1],
            attention_channels=128,
            global_context_att=global_context_att,
        )
        self.bn = nn.BatchNorm1d(self.channels[-1] * 2)
        self.linear = nn.Linear(self.channels[-1] * 2, emb_dim)

    def get_feat_num(self):
        self.feature_extract.eval()
        wav = [torch.randn(self.sr).to(next(self.feature_extract.parameters()).device)]
        with torch.no_grad():
            features = self.feature_extract(wav)
        select_feature = features[self.feature_selection]
        if isinstance(select_feature, (list, tuple)):
            return len(select_feature)
        return 1

    def get_feat(self, x):
        if self.update_extract:
            features = self.feature_extract([sample for sample in x])
        else:
            with torch.no_grad():
                features = self.feature_extract([sample for sample in x])

        x = features[self.feature_selection]
        if isinstance(x, (list, tuple)):
            x = torch.stack(x, dim=0)
        else:
            x = x.unsqueeze(0)

        norm_weights = F.softmax(self.feature_weight, dim=-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = (norm_weights * x).sum(dim=0)
        x = torch.transpose(x, 1, 2) + 1e-6
        return self.instance_norm(x)

    def forward(self, x):
        x = self.get_feat(x)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        out = self.bn(self.pooling(out))
        return self.linear(out)


def load_wavlm_large_ecapa_tdnn(
    checkpoint_path: Optional[str] = None,
    map_location="cpu",
) -> WavLMLargeECAPATDNN:
    model = WavLMLargeECAPATDNN()
    if checkpoint_path is None:
        return model

    state = torch.load(checkpoint_path, map_location=map_location)
    if "model" not in state:
        raise KeyError("Expected checkpoint to contain a top-level 'model' key")
    model.load_state_dict(state["model"], strict=True)
    return model


__all__ = ["WavLMLargeECAPATDNN", "load_wavlm_large_ecapa_tdnn"]
