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
"""GPU integration test for the vLLM-Omni SD3 DPO rollout pipeline.

This test intentionally uses full SD3 weights instead of a tiny/random model.
Set ``SD3_DPO_MODEL`` to a local checkpoint path to avoid downloading from the
Hub during local runs.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import ExitStack
from uuid import uuid4

import pytest
import torch

async_omni_mod = pytest.importorskip("vllm_omni.entrypoints.async_omni")
inputs_data_mod = pytest.importorskip("vllm_omni.inputs.data")
tokenizer_mod = pytest.importorskip("verl.utils.tokenizer")

AsyncOmni = async_omni_mod.AsyncOmni
OmniDiffusionSamplingParams = inputs_data_mod.OmniDiffusionSamplingParams
normalize_token_ids = tokenizer_mod.normalize_token_ids

MODEL = os.getenv("SD3_DPO_MODEL", "stabilityai/stable-diffusion-3.5-medium")
CUSTOM_PIPELINE_CLASS = (
    "verl_omni.pipelines.sd3_dpo.vllm_omni_rollout_adapter.StableDiffusion3DPOPipeline"
)


def _server_custom_prompt(prompt: str, negative_prompt: str = "") -> dict:
    """Build the OmniCustomPrompt shape emitted by vLLMOmniHttpServer.generate()."""
    custom_prompt = {"prompt_ids": normalize_token_ids([0]), "prompt": prompt}
    if negative_prompt is not None:
        custom_prompt["negative_prompt"] = negative_prompt
    return custom_prompt


def _server_sampling_params(sampling_params: dict) -> OmniDiffusionSamplingParams:
    """Mirror vLLMOmniHttpServer.generate()'s dict-to-sampling-params split."""
    sampling_kwargs = {}
    extra_args = {}
    for key, value in sampling_params.items():
        if hasattr(OmniDiffusionSamplingParams, key):
            sampling_kwargs[key] = value
        else:
            extra_args[key] = value
    sampling_kwargs["extra_args"] = extra_args
    return OmniDiffusionSamplingParams(**sampling_kwargs)


def _assert_cpu_tensor(tensor: torch.Tensor | None) -> None:
    assert tensor is not None
    assert tensor.device.type == "cpu"


async def _run_sd3_dpo_pipeline_full_weights_accepts_async_server_request():
    prompt = (
        "a cinematic photo of a red panda astronaut standing on the moon, "
        "soft rim lighting, detailed space suit, sharp focus"
    )
    negative_prompt = "blurry, low quality, distorted anatomy, watermark"
    sampling_params = _server_sampling_params(
        {
            "height": 256,
            "width": 256,
            "num_inference_steps": 2,
            "guidance_scale": 4.0,
            "max_sequence_length": 256,
            "seed": 1234,
            "logprobs": False,
        }
    )

    with ExitStack() as after:
        engine = AsyncOmni(
            model=MODEL,
            custom_pipeline_args={"pipeline_class": CUSTOM_PIPELINE_CLASS},
            enforce_eager=True,
            dtype="bfloat16",
        )
        after.callback(engine.shutdown)

        final_output = None
        async for output in engine.generate(
            prompt=_server_custom_prompt(prompt, negative_prompt=negative_prompt),
            request_id=f"sd3_dpo_{uuid4().hex[:8]}",
            sampling_params_list=[sampling_params],
            output_modalities=["image"],
        ):
            final_output = output

    assert final_output is not None
    assert final_output.images, "Expected SD3 DPO rollout to return an image"

    custom_output = final_output.custom_output or {}
    assert custom_output.get("image_latents") is not None
    assert custom_output.get("prompt_embeds") is not None
    assert custom_output.get("pooled_prompt_embeds") is not None
    _assert_cpu_tensor(custom_output.get("image_latents"))
    _assert_cpu_tensor(custom_output.get("prompt_embeds"))
    _assert_cpu_tensor(custom_output.get("pooled_prompt_embeds"))
    _assert_cpu_tensor(custom_output.get("negative_prompt_embeds"))
    _assert_cpu_tensor(custom_output.get("negative_pooled_prompt_embeds"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SD3 DPO full-weight rollout requires CUDA")
def test_sd3_dpo_pipeline_full_weights_accepts_async_server_request():
    asyncio.run(_run_sd3_dpo_pipeline_full_weights_accepts_async_server_request())
