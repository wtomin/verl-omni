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
"""CPU tests for the SD3 DPO vLLM-Omni rollout adapter."""

from types import MethodType, SimpleNamespace

import pytest

pytest.importorskip(
    "vllm_omni.diffusion.models.sd3.pipeline_sd3",
    reason="SD3 DPO rollout adapter tests require real vllm_omni",
)

import torch
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.sd3_dpo.vllm_omni_rollout_adapter import StableDiffusion3DPOPipeline


def _make_pipeline(pipeline_cls):
    pipe = pipeline_cls.__new__(pipeline_cls)
    pipe.device = torch.device("cpu")
    pipe.default_sample_size = 8
    pipe.vae_scale_factor = 8
    pipe.output_type = "latent"
    pipe.transformer = SimpleNamespace(in_channels=16)
    pipe.calls = {}

    def check_inputs(self, *args, **kwargs):
        self.calls["check_inputs"] = (args, kwargs)

    def encode_prompt(
        self,
        prompt,
        prompt_2="",
        prompt_3="",
        prompt_embeds=None,
        max_sequence_length=256,
        num_images_per_prompt=1,
    ):
        self.calls.setdefault("encode_prompt", []).append(
            {
                "prompt": prompt,
                "prompt_2": prompt_2,
                "prompt_3": prompt_3,
                "max_sequence_length": max_sequence_length,
                "num_images_per_prompt": num_images_per_prompt,
            }
        )
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        effective_batch = batch_size * num_images_per_prompt
        embeds = torch.arange(effective_batch * 6, dtype=torch.float32).reshape(effective_batch, 2, 3)
        pooled = torch.arange(effective_batch * 4, dtype=torch.float32).reshape(effective_batch, 4)
        return embeds, pooled

    def prepare_latents(self, batch_size, num_channels_latents, height, width, generator, latents=None):
        self.calls["prepare_latents"] = {
            "batch_size": batch_size,
            "num_channels_latents": num_channels_latents,
            "height": height,
            "width": width,
            "generator": generator,
            "latents": latents,
        }
        if latents is not None:
            return latents
        return torch.zeros(
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

    def prepare_timesteps(self, num_inference_steps, sigmas, image_seq_len):
        self.calls["prepare_timesteps"] = {
            "num_inference_steps": num_inference_steps,
            "sigmas": sigmas,
            "image_seq_len": image_seq_len,
        }
        return torch.tensor([1.0, 0.0]), 2

    def diffuse(
        self,
        latents,
        timesteps,
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
        do_true_cfg,
        guidance_scale,
        cfg_normalize=False,
    ):
        self.calls["diffuse"] = {
            "timesteps": timesteps,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
            "do_true_cfg": do_true_cfg,
            "guidance_scale": guidance_scale,
            "cfg_normalize": cfg_normalize,
        }
        return latents + 1

    pipe.check_inputs = MethodType(check_inputs, pipe)
    pipe.encode_prompt = MethodType(encode_prompt, pipe)
    pipe.prepare_latents = MethodType(prepare_latents, pipe)
    pipe.prepare_timesteps = MethodType(prepare_timesteps, pipe)
    pipe.diffuse = MethodType(diffuse, pipe)
    return pipe


def test_sd3_dpo_pipeline_registered_with_real_vllm_omni():
    pipeline_cls = VllmOmniPipelineBase.get_class("StableDiffusion3Pipeline", "dpo")

    assert pipeline_cls is StableDiffusion3DPOPipeline


def test_sd3_dpo_pipeline_forward_runs_with_sd3_prepare_latents_signature():
    pipe = _make_pipeline(StableDiffusion3DPOPipeline)
    req = OmniDiffusionRequest(
        prompts=[
            {"caption": "a cat", "negative_prompt": "blurry"},
            {"extra_args": {"text": "a dog"}, "raw_negative_prompt": "low quality"},
        ],
        request_ids=["req-0", "req-1"],
        sampling_params=OmniDiffusionSamplingParams(
            height=64,
            width=64,
            sigmas=None,
            max_sequence_length=128,
            num_inference_steps=3,
            generator=None,
            seed=123,
            num_outputs_per_prompt=2,
            guidance_scale=1.0,
        ),
    )

    output = pipe.forward(req)

    assert isinstance(output, DiffusionOutput)
    assert output.output.shape == (4, 16, 8, 8)
    assert torch.equal(output.output, output.custom_output["image_latents"])

    assert pipe.calls["check_inputs"][0][:6] == (
        ["a cat", "a dog"],
        "",
        "",
        64,
        64,
        ["blurry", "low quality"],
    )
    assert pipe.calls["prepare_latents"]["batch_size"] == 4
    assert pipe.calls["prepare_latents"]["num_channels_latents"] == 16
    assert pipe.calls["prepare_latents"]["height"] == 64
    assert pipe.calls["prepare_latents"]["width"] == 64
    assert isinstance(pipe.calls["prepare_latents"]["generator"], torch.Generator)
    assert pipe.calls["prepare_latents"]["latents"] is None

    assert pipe.calls["prepare_timesteps"] == {
        "num_inference_steps": 3,
        "sigmas": None,
        "image_seq_len": 16,
    }
    assert pipe.calls["diffuse"]["do_true_cfg"] is False
    assert pipe.calls["diffuse"]["guidance_scale"] == 1.0
    assert pipe.calls["diffuse"]["cfg_normalize"] is False

    assert output.custom_output["prompt_embeds"].shape == (4, 2, 3)
    assert output.custom_output["pooled_prompt_embeds"].shape == (4, 4)
    assert output.custom_output["negative_prompt_embeds"] is None
    assert output.custom_output["negative_pooled_prompt_embeds"] is None
