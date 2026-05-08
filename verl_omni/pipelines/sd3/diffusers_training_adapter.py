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
"""
SD3 (Stable Diffusion 3.x) training-side adapter for diffusers-based diffusion RL.
"""

from typing import Optional

import torch
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from tensordict import TensorDict
from verl.utils.device import get_device_name

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.workers.config import DiffusionModelConfig

__all__ = ["SD3Adapter"]


def _build_sd3_scheduler(model_path: str) -> FlowMatchSDEDiscreteScheduler:
    euler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path=model_path,
        subfolder="scheduler",
    )
    return FlowMatchSDEDiscreteScheduler.from_config(euler.config)


def _configure_sd3_scheduler(
    scheduler: FlowMatchSDEDiscreteScheduler,
    *,
    num_inference_steps: int,
    device: str,
) -> None:
    scheduler.set_timesteps(num_inference_steps, device=device)


@DiffusionModelBase.register("StableDiffusion3Pipeline")
class SD3Adapter(DiffusionModelBase):
    """Training adapter for Stable Diffusion 3.x (e.g. SD3.0, SD3.5) diffusion models.

    Implements the :class:`~verl_omni.pipelines.model_base.DiffusionModelBase`
    interface for the ``StableDiffusion3Pipeline`` architecture, providing scheduler
    configuration, model-input construction, and the forward/sampling step
    used during RL training (e.g. DPO, FlowGRPO).

    Registered under ``"StableDiffusion3Pipeline"`` so it is automatically selected
    when ``DiffusionModelConfig.architecture`` matches that name.
    """

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig):
        """Build and configure the flow-matching scheduler for SD3.

        Args:
            model_config (DiffusionModelConfig): Configuration for the diffusion model,
                used to determine the model path and timestep settings.

        Returns:
            FlowMatchSDEDiscreteScheduler: Scheduler with timesteps already set
                for the current device.
        """
        scheduler = _build_sd3_scheduler(model_config.local_path)
        cls.set_timesteps(scheduler, model_config, get_device_name())
        return scheduler

    @classmethod
    def set_timesteps(cls, scheduler: FlowMatchSDEDiscreteScheduler, model_config: DiffusionModelConfig, device: str):
        """Configure timesteps on the scheduler for SD3.

        Args:
            scheduler (FlowMatchSDEDiscreteScheduler): The scheduler whose timesteps
                will be set.
            model_config (DiffusionModelConfig): Configuration providing
                number of inference steps.
            device (str): The device (e.g. ``"cuda"``) to move the timesteps to.
        """
        _configure_sd3_scheduler(
            scheduler,
            num_inference_steps=model_config.pipeline.num_inference_steps,
            device=device,
        )

    @classmethod
    def prepare_model_inputs(
        cls,
        module: SD3Transformer2DModel,
        model_config: DiffusionModelConfig,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_prompt_embeds_mask: torch.Tensor,
        micro_batch: TensorDict,
        step: int,
    ) -> tuple[dict, dict]:
        """Build SD3-specific inputs for the transformer forward pass.

        Args:
            module (SD3Transformer2DModel): The SD3 transformer module.
            model_config (DiffusionModelConfig): Configuration providing guidance
                scale and other model settings.
            latents (torch.Tensor): Full latent tensor of shape ``(B, T, ...)``.
            timesteps (torch.Tensor): Full timestep tensor of shape ``(B, T)``.
            prompt_embeds (torch.Tensor): Positive prompt embeddings of shape ``(B, L, D)``.
            prompt_embeds_mask (torch.Tensor): Attention mask for *prompt_embeds* of shape ``(B, L)``.
            negative_prompt_embeds (torch.Tensor): Negative prompt embeddings of shape ``(B, L, D)``.
            negative_prompt_embeds_mask (torch.Tensor): Attention mask for *negative_prompt_embeds*.
            micro_batch (TensorDict): Micro-batch containing metadata such as
                ``height``, ``width``, and ``vae_scale_factor``.
            step (int): Current denoising step index used to slice *latents* and *timesteps*.

        Returns:
            tuple[dict, dict]: A pair of ``(model_inputs, negative_model_inputs)`` dicts
                ready to be unpacked into the transformer forward call.
        """
        hidden_states = latents[:, step]
        timestep = timesteps[:, step]

        # SD3 uses pooled projections for the second text encoder; pooled_prompt_embeds
        # are passed separately in StableDiffusion3Pipeline.
        pooled_prompt_embeds = micro_batch.get("pooled_prompt_embeds", None)
        pooled_negative_prompt_embeds = micro_batch.get("pooled_negative_prompt_embeds", None)

        model_inputs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "pooled_projections": pooled_prompt_embeds,
            "return_dict": False,
        }

        negative_model_inputs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": negative_prompt_embeds,
            "pooled_projections": pooled_negative_prompt_embeds,
            "return_dict": False,
        }

        return model_inputs, negative_model_inputs

    @classmethod
    def forward_and_sample_previous_step(
        cls,
        module: SD3Transformer2DModel,
        scheduler: FlowMatchSDEDiscreteScheduler,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]],
        scheduler_inputs: Optional[TensorDict | dict[str, torch.Tensor]],
        step: int,
    ):
        """Run the SD3 transformer and sample the previous denoising step.

        Used by RL algorithms (DPO, FlowGRPO) that require log-probabilities for
        reversed sampling.

        Args:
            module (SD3Transformer2DModel): The SD3 transformer module.
            scheduler (FlowMatchSDEDiscreteScheduler): Scheduler used to sample
                the previous step and compute log-probabilities.
            model_config (DiffusionModelConfig): Configuration providing
                ``guidance_scale``, ``algo.noise_level``, and ``algo.sde_type``.
            model_inputs (dict[str, torch.Tensor]): Positive-prompt inputs for
                the transformer forward pass.
            negative_model_inputs (Optional[dict[str, torch.Tensor]]): Negative-prompt
                inputs used for guidance; may be ``None`` when guidance is disabled.
            scheduler_inputs (Optional[TensorDict | dict[str, torch.Tensor]]): Must
                contain ``"all_latents"`` and ``"all_timesteps"`` tensors.
            step (int): Current denoising step index.

        Returns:
            tuple: A 3-tuple of ``(log_prob, prev_sample_mean, std_dev_t)``.
        """
        assert scheduler_inputs is not None
        latents = scheduler_inputs["all_latents"]
        timesteps = scheduler_inputs["all_timesteps"]

        noise_pred = module(**model_inputs)[0]

        guidance_scale = model_config.pipeline.guidance_scale
        if guidance_scale > 1.0 and negative_model_inputs is not None:
            neg_noise_pred = module(**negative_model_inputs)[0]
            noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

        algo = model_config.algo
        noise_level = algo.noise_level if algo is not None else 0.7
        sde_type = algo.sde_type if algo is not None else "sde"

        _, log_prob, prev_sample_mean, std_dev_t = scheduler.sample_previous_step(
            sample=latents[:, step].float(),
            model_output=noise_pred.float(),
            timestep=timesteps[:, step],
            noise_level=noise_level,
            prev_sample=latents[:, step + 1].float(),
            sde_type=sde_type,
            return_logprobs=True,
        )
        return log_prob, prev_sample_mean, std_dev_t
