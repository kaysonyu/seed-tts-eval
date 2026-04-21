import json
import sys
from pathlib import Path

import numpy as np
import torch
import triton_python_backend_utils as pb_utils


def _extract_single_path(input_array) -> str:
    values = np.asarray(input_array).reshape(-1)
    if values.size != 1:
        raise ValueError(f"Expected exactly one audio path per request, got {values.size}")
    value = values[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    raise TypeError(f"Unsupported AUDIO_PATH element type: {type(value).__name__}")


def _get_string_parameter(model_config: dict, key: str, default: str = "") -> str:
    parameters = model_config.get("parameters", {})
    if key not in parameters:
        return default
    return parameters[key]["string_value"]


def _get_int_parameter(model_config: dict, key: str, default: int) -> int:
    value = _get_string_parameter(model_config, key, str(default))
    return int(value)


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        repo_root = _get_string_parameter(self.model_config, "repo_root")
        if repo_root and repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        from similarity_service.audio import AudioLoader
        from similarity_service.buckets import BucketPolicy, DEFAULT_BUCKET_SPECS
        from similarity_service.triton_router_core import PathEmbeddingRouterCore

        bucket_seconds = json.loads(_get_string_parameter(self.model_config, "bucket_seconds_json", "[]"))
        bucket_policy = BucketPolicy(DEFAULT_BUCKET_SPECS)
        selected_specs = bucket_policy.bucket_specs_from_seconds(bucket_seconds)
        self.bucket_policy = BucketPolicy(selected_specs)
        allowed_roots = json.loads(_get_string_parameter(self.model_config, "allowed_roots_json", "[]"))
        self.audio_loader = AudioLoader(
            allowed_roots=[Path(root) for root in allowed_roots],
            max_samples=self.bucket_policy.max_samples,
            audio_backend=_get_string_parameter(self.model_config, "audio_backend", "soundfile"),
        )
        normalize_embeddings = _get_string_parameter(
            self.model_config,
            "normalize_embeddings",
            "true",
        ).lower() == "true"
        self.bucket_model_prefix = _get_string_parameter(
            self.model_config,
            "bucket_model_prefix",
            "wavlm_ecapa",
        )
        self.router_core = PathEmbeddingRouterCore(
            audio_loader=self.audio_loader,
            bucket_policy=self.bucket_policy,
            normalize_embeddings=normalize_embeddings,
            embedding_cache_items=_get_int_parameter(self.model_config, "embedding_cache_items", 4096),
            cache_namespace=_get_string_parameter(
                self.model_config,
                "embedding_cache_namespace",
                f"{self.bucket_model_prefix}:normalize={normalize_embeddings}",
            ),
        )

    def _bucket_model_name(self, seconds: int) -> str:
        return f"{self.bucket_model_prefix}_{seconds}s"

    def _invoke_bucket_model(self, bucket, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        lengths_for_triton = lengths.reshape(-1, 1)
        request = pb_utils.InferenceRequest(
            model_name=self._bucket_model_name(bucket.seconds),
            requested_output_names=["emb"],
            inputs=[
                pb_utils.Tensor("x", x.numpy().astype(np.float32, copy=False)),
                pb_utils.Tensor("lengths", lengths_for_triton.numpy().astype(np.int64, copy=False)),
            ],
            preferred_memory=pb_utils.PreferredMemory(
                pb_utils.TRITONSERVER_MEMORY_CPU,
                0,
            ),
        )
        response = request.exec()
        if response.has_error():
            raise RuntimeError(response.error().message())
        output_tensor = pb_utils.get_output_tensor_by_name(response, "emb")
        return torch.from_numpy(output_tensor.as_numpy())

    def execute(self, requests):
        responses = [None] * len(requests)
        valid_request_indices = []
        audio_paths = []

        for idx, request in enumerate(requests):
            try:
                input_tensor = pb_utils.get_input_tensor_by_name(request, "AUDIO_PATH")
                if input_tensor is None:
                    raise ValueError("Missing required input AUDIO_PATH")
                audio_paths.append(_extract_single_path(input_tensor.as_numpy()))
                valid_request_indices.append(idx)
            except Exception as exc:
                responses[idx] = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(str(exc))
                )

        if audio_paths:
            result = self.router_core.embed_paths(audio_paths, self._invoke_bucket_model)
            for local_idx, request_idx in enumerate(valid_request_indices):
                error = result.errors[local_idx]
                if error is not None:
                    responses[request_idx] = pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(str(error))
                    )
                    continue
                embedding = result.embeddings[local_idx]
                responses[request_idx] = pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor(
                            "EMBEDDING",
                            embedding.numpy().astype(np.float32, copy=False),
                        )
                    ]
                )

        return responses
