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

"""Stable Diffusion 3.5 rollout adapter for diffusion DPO.

Runs a deterministic Euler denoise loop (no log-probabilities) and returns
offline-DPO tensors in ``custom_output``:

* ``latents`` — denoised latents after the full loop (before VAE decode).
* ``prompt_embeds``, ``prompt_embeds_mask``, ``pooled_prompt_embeds``
* ``negative_prompt_embeds``, ``negative_prompt_embeds_mask``,
  ``negative_pooled_prompt_embeds`` (when CFG is enabled).
"""

from typing import Any

import torch
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.sd3 import StableDiffusion3Pipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest

from verl_omni.pipelines.model_base import VllmOmniPipelineBase

from .common import maybe_to_cpu, prompt_embeds_mask_from_embeds

__all__ = ["StableDiffusion3DPOPipeline"]


@VllmOmniPipelineBase.register("StableDiffusion3Pipeline", algorithm="dpo")
class StableDiffusion3DPOPipeline(StableDiffusion3Pipeline):
    """Rollout pipeline for SD3/SD3.5 Diffusion-DPO (Euler ODE, no log-probs)."""

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
        **kwargs: Any,
    ) -> DiffusionOutput:
        del kwargs

        prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in req.prompts] or prompt
        negative_prompt = [
            "" if isinstance(p, str) else (p.get("negative_prompt") or "") for p in req.prompts
        ] or negative_prompt

        height = req.sampling_params.height or self.default_sample_size * self.vae_scale_factor
        width = req.sampling_params.width or self.default_sample_size * self.vae_scale_factor
        sigmas = req.sampling_params.sigmas or sigmas
        max_sequence_length = req.sampling_params.max_sequence_length or max_sequence_length
        num_inference_steps = req.sampling_params.num_inference_steps or num_inference_steps
        generator = req.sampling_params.generator or generator
        num_images_per_prompt = (
            req.sampling_params.num_outputs_per_prompt
            if req.sampling_params.num_outputs_per_prompt > 0
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

        self._guidance_scale = req.sampling_params.guidance_scale
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
        prompt_embeds_mask = prompt_embeds_mask_from_embeds(prompt_embeds)

        do_cfg = self.guidance_scale > 1
        negative_prompt_embeds_out = None
        negative_prompt_embeds_mask = None
        negative_pooled_prompt_embeds_out = None
        if do_cfg:
            negative_prompt_embeds_out, negative_pooled_prompt_embeds_out = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_3=negative_prompt_3,
                prompt_embeds=negative_prompt_embeds,
                max_sequence_length=max_sequence_length,
                num_images_per_prompt=num_images_per_prompt,
            )
            negative_prompt_embeds_mask = prompt_embeds_mask_from_embeds(negative_prompt_embeds_out)

        num_channels_latents = self.transformer.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
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
            negative_prompt_embeds=negative_prompt_embeds_out if do_cfg else None,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds_out if do_cfg else None,
            do_true_cfg=do_cfg,
            guidance_scale=self.guidance_scale,
            cfg_normalize=False,
        )
        rollout_latents = latents.detach().clone()

        self._current_timestep = None
        if self.output_type == "latent":
            image = latents
        else:
            latents = latents.to(self.vae.dtype)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]

        return DiffusionOutput(
            output=maybe_to_cpu(image),
            custom_output={
                "latents": maybe_to_cpu(rollout_latents),
                "prompt_embeds": maybe_to_cpu(prompt_embeds),
                "prompt_embeds_mask": maybe_to_cpu(prompt_embeds_mask),
                "pooled_prompt_embeds": maybe_to_cpu(pooled_prompt_embeds),
                "negative_prompt_embeds": maybe_to_cpu(negative_prompt_embeds_out),
                "negative_prompt_embeds_mask": maybe_to_cpu(negative_prompt_embeds_mask),
                "negative_pooled_prompt_embeds": maybe_to_cpu(negative_pooled_prompt_embeds_out),
            },
        )
