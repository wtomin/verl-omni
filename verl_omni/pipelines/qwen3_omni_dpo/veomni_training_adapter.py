# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3-Omni DPO training-side adapter for the VeOmni engine."""

from typing import Any

import torch

from verl_omni.pipelines.omni_training_adapter import OmniTrainingAdapterBase

__all__ = ["Qwen3OmniDPO"]


@OmniTrainingAdapterBase.register("Qwen3OmniMoeForConditionalGeneration", algorithm="dpo")
class Qwen3OmniDPO(OmniTrainingAdapterBase):
    """Prepare Qwen3-Omni processor outputs for VeOmni offline DPO training."""

    @staticmethod
    def _drop_zero_rows(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.numel() == 0:
            return tensor
        keep = tensor.reshape(tensor.shape[0], -1).abs().sum(dim=-1) != 0
        return tensor[keep]

    @classmethod
    def prepare_model_inputs(cls, model_inputs: dict[str, Any], dtype: torch.dtype) -> dict[str, Any]:
        """Normalize padded Qwen3-Omni multimodal inputs before model forward."""
        model_inputs = dict(model_inputs)
        position_ids = model_inputs.get("position_ids")
        if isinstance(position_ids, torch.Tensor) and position_ids.ndim == 4 and position_ids.shape[2] == 1:
            model_inputs["position_ids"] = position_ids.squeeze(2).contiguous()

        for pixel_key, grid_key in (("pixel_values", "image_grid_thw"), ("pixel_values_videos", "video_grid_thw")):
            pixel_values = model_inputs.get(pixel_key)
            grid = model_inputs.get(grid_key)
            if isinstance(grid, torch.Tensor) and grid.ndim == 3:
                model_inputs[grid_key] = cls._drop_zero_rows(grid.reshape(-1, grid.shape[-1]))
            if isinstance(pixel_values, torch.Tensor) and pixel_values.ndim == 3:
                model_inputs[pixel_key] = cls._drop_zero_rows(pixel_values.reshape(-1, pixel_values.shape[-1]))

        input_features = model_inputs.get("input_features")
        if isinstance(input_features, torch.Tensor) and input_features.ndim == 3:
            model_inputs["input_features"] = cls._drop_zero_rows(input_features.reshape(-1, input_features.shape[-1]))

        audio_feature_lengths = model_inputs.get("audio_feature_lengths")
        if isinstance(audio_feature_lengths, torch.Tensor) and audio_feature_lengths.ndim > 1:
            audio_feature_lengths = audio_feature_lengths.reshape(-1)
            model_inputs["audio_feature_lengths"] = audio_feature_lengths[audio_feature_lengths != 0]

        for key in ("pixel_values", "pixel_values_videos", "input_features"):
            value = model_inputs.get(key)
            if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                model_inputs[key] = value.to(dtype=dtype)
        return model_inputs
