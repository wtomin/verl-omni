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

"""Stable Diffusion 3 rollout-side adapter for diffusion DPO."""

import logging
from typing import Any

import torch
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.sd3.pipeline_sd3 import StableDiffusion3Pipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest

from verl_omni.pipelines.model_base import VllmOmniPipelineBase

__all__ = ["StableDiffusion3DPOPipeline"]

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def _prompt_value(prompt: Any, *keys: str, default: str = "") -> str:
    """Extract a raw text prompt from vLLM-Omni custom prompt dictionaries."""
    if isinstance(prompt, str):
        return prompt
    if not isinstance(prompt, dict):
        return default

    for key in keys:
        value = prompt.get(key)
        if isinstance(value, str):
            return value

    extra_args = prompt.get("extra_args")
    if isinstance(extra_args, dict):
        for key in keys:
            value = extra_args.get(key)
            if isinstance(value, str):
                return value

    return default


def _extract_prompt_list(req: OmniDiffusionRequest, *keys: str, default: str = "") -> list[str]:
    return [_prompt_value(prompt, *keys, default=default) for prompt in req.prompts]


def _extract_optional_prompt_list(req: OmniDiffusionRequest, *keys: str) -> list[str] | str:
    values = _extract_prompt_list(req, *keys, default="")
    return values if any(values) else ""


@VllmOmniPipelineBase.register("StableDiffusion3Pipeline", algorithm="dpo")
class StableDiffusion3DPOPipeline(StableDiffusion3Pipeline):
    """Rollout pipeline for SD3 DPO.

    DPO only needs the final denoised image latent plus text-condition embeddings
    for the training-side flow-matching objective; it does not collect per-step
    log-probabilities or intermediate latents.
    """

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | list[str] = "",
        prompt_2: str | list[str] = "",
        prompt_3: str | list[str] = "",
        negative_prompt: str | list[str] = "",
        negative_prompt_2: str | list[str] = "",
        negative_prompt_3: str | list[str] = "",
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 28,
        sigmas: list[float] | None = None,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        pooled_prompt_embeds: torch.Tensor | None = None,
        negative_pooled_prompt_embeds: torch.Tensor | None = None,
        max_sequence_length: int = 256,
        output_type: str | None = None,
    ) -> DiffusionOutput:
        if req.prompts:
            prompt = _extract_prompt_list(
                req, "prompt", "raw_prompt", "caption", "text", default=""
            )  # extract from request, required
            prompt_2 = _extract_optional_prompt_list(req, "prompt_2", "raw_prompt_2")  # extract from request, optional
            prompt_3 = _extract_optional_prompt_list(req, "prompt_3", "raw_prompt_3")  # extract from request, optional
            negative_prompt = _extract_prompt_list(
                req, "negative_prompt", "raw_negative_prompt", default=""
            )  # extract from request, required
            negative_prompt_2 = _extract_optional_prompt_list(
                req, "negative_prompt_2", "raw_negative_prompt_2"
            )  # extract from request, optional
            negative_prompt_3 = _extract_optional_prompt_list(
                req, "negative_prompt_3", "raw_negative_prompt_3"
            )  # extract from request, optional

        if prompt == "":
            logger.warning("Prompt is empty. Please check the input prompts.")
        sampling_params = req.sampling_params
        height = sampling_params.height or self.default_sample_size * self.vae_scale_factor
        width = sampling_params.width or self.default_sample_size * self.vae_scale_factor
        sigmas = sampling_params.sigmas or sigmas
        max_sequence_length = sampling_params.max_sequence_length or max_sequence_length
        num_inference_steps = sampling_params.num_inference_steps or num_inference_steps
        generator = sampling_params.generator or generator
        if generator is None and sampling_params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(sampling_params.seed)
        num_images_per_prompt = (
            sampling_params.num_outputs_per_prompt
            if sampling_params.num_outputs_per_prompt > 0
            else num_images_per_prompt
        )

        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = sampling_params.guidance_scale if sampling_params.guidance_scale is not None else 1.0
        self._current_timestep = None
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            prompt_embeds=prompt_embeds,
            max_sequence_length=max_sequence_length,
            num_images_per_prompt=num_images_per_prompt,
        )

        do_cfg = self.guidance_scale > 1
        if do_cfg:
            negative_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_3=negative_prompt_3,
                prompt_embeds=negative_prompt_embeds,
                max_sequence_length=max_sequence_length,
                num_images_per_prompt=num_images_per_prompt,
            )

        num_channels_latents = self.transformer.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,  # might be removed after vllm-omni 0.18.0
            self.device,  # might be removed after vllm-omni 0.18.0
            generator,
            latents,
        )

        timesteps, num_inference_steps = self.prepare_timesteps(num_inference_steps, sigmas, latents.shape[1])
        self._num_timesteps = len(timesteps)

        latents = self.diffuse(
            latents=latents,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds if do_cfg else None,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds if do_cfg else None,
            do_true_cfg=do_cfg,
            guidance_scale=self.guidance_scale,
            cfg_normalize=False,
        )

        self._current_timestep = None
        image_latents = latents
        output_type = output_type or self.output_type
        if output_type == "latent":
            image = image_latents
        else:
            decode_latents = image_latents.to(self.vae.dtype)
            decode_latents = (decode_latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(decode_latents, return_dict=False)[0]

        return DiffusionOutput(
            output=image,
            custom_output={
                "image_latents": image_latents,
                "prompt_embeds": prompt_embeds,
                "prompt_embeds_mask": None,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds if do_cfg else None,
                "negative_prompt_embeds_mask": None,
                "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds if do_cfg else None,
            },
        )
