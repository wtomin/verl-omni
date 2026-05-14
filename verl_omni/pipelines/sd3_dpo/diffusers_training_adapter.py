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

"""Stable Diffusion 3 training-side adapter for diffusion DPO."""

from typing import Any, Optional

import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler, ModelMixin, SchedulerMixin
from tensordict import TensorDict
from verl.utils.device import get_device_name

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.workers.config import DiffusionModelConfig

__all__ = ["StableDiffusion3DPO"]


def _build_sd3_scheduler(model_path: str) -> FlowMatchEulerDiscreteScheduler:
    return FlowMatchEulerDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path=model_path,
        subfolder="scheduler",
    )


@DiffusionModelBase.register("StableDiffusion3Pipeline", algorithm="dpo")
class StableDiffusion3DPO(DiffusionModelBase):
    """Training adapter for SD3 Diffusion-DPO.

    This adapter is intentionally limited to SD3-specific tensor preparation and
    transformer forwarding. The pairwise DPO objective itself belongs in
    ``verl_omni.workers.utils.losses``.
    """

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig):
        """Build and configure the SD3 flow-matching scheduler."""
        scheduler = _build_sd3_scheduler(model_config.local_path)
        cls.set_timesteps(scheduler, model_config, get_device_name())
        return scheduler

    @classmethod
    def set_timesteps(cls, scheduler: SchedulerMixin, model_config: DiffusionModelConfig, device: str):
        """Configure scheduler timesteps for SD3."""
        scheduler.set_timesteps(model_config.pipeline.num_inference_steps, device=device)

    @classmethod
    def prepare_model_inputs(
        cls,
        module: ModelMixin,
        model_config: DiffusionModelConfig,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_prompt_embeds_mask: torch.Tensor,
        micro_batch: TensorDict,
        step: int,
    ) -> tuple[dict[str, Any], None]:
        """Build SD3 transformer inputs.

        For DPO-specific training, callers should normally pass already-noised
        latents and the sampled training timesteps.
        latents: (B, C, H, W)  # already-noised latents
        timesteps: (B,)
        """
        del prompt_embeds_mask, negative_prompt_embeds_mask, step

        pooled_prompt_embeds = micro_batch.get("pooled_prompt_embeds", None)
        negative_pooled_prompt_embeds = micro_batch.get("negative_pooled_prompt_embeds", None)
        do_true_cfg = (
            model_config.pipeline.true_cfg_scale > 1.0
            and negative_prompt_embeds is not None
            and negative_pooled_prompt_embeds is not None
        )

        model_inputs = cls.build_transformer_inputs(
            latents=latents,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds if do_true_cfg else None,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds if do_true_cfg else None,
            do_true_cfg=do_true_cfg,
            guidance_scale=model_config.pipeline.guidance_scale,
        )

        # Keep a lightweight sanity check near the adapter boundary; SD3 uses
        # pooled prompt projections in addition to sequence prompt embeddings.
        if model_inputs["pooled_prompt_embeds"] is None:
            raise KeyError("SD3 DPO training requires `pooled_prompt_embeds` in the micro batch.")

        del module
        return model_inputs, None

    @staticmethod
    def build_transformer_inputs(
        *,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        do_true_cfg: bool = False,
        guidance_scale: Optional[float] = None,
    ) -> dict[str, Any]:
        """Create the SD3 transformer keyword arguments."""
        return {
            "latents": latents,
            "timesteps": timesteps,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
            "do_true_cfg": do_true_cfg,
            "guidance_scale": guidance_scale,
        }

    @staticmethod
    def forward_mse(
        module: ModelMixin,
        model_inputs: dict[str, Any],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run SD3 transformer and return per-sample flow-matching MSE."""
        model_pred = module(**model_inputs)[0]
        mse = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        reduce_dims = tuple(range(1, mse.ndim))
        return mse.mean(dim=reduce_dims), model_pred

    @classmethod
    def forward_dpo_step(
        cls,
        module: ModelMixin,
        scheduler: SchedulerMixin,
        model_config: DiffusionModelConfig,
        micro_batch: TensorDict,
    ) -> dict[str, torch.Tensor]:
        """Run one SD3 flow-matching training step for DPO."""
        latents = micro_batch.get("image_latents", None)

        if latents is None:
            raise KeyError("SD3 DPO training requires `image_latents` in the micro batch.")

        noise = torch.randn_like(latents)
        timestep_indices = torch.randint(
            0,
            len(scheduler.timesteps),
            (latents.shape[0],),
            device=latents.device,
        )
        timesteps = scheduler.timesteps.to(device=latents.device)[timestep_indices]
        sigmas = scheduler.sigmas.to(device=latents.device, dtype=latents.dtype)[timestep_indices]
        sigmas = sigmas.view(-1, *([1] * (latents.ndim - 1)))

        noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
        target = noise - latents

        model_inputs, _ = cls.prepare_model_inputs(
            module=module,
            model_config=model_config,
            latents=noisy_latents,
            timesteps=timesteps,
            prompt_embeds=micro_batch["prompt_embeds"],
            prompt_embeds_mask=micro_batch.get("prompt_embeds_mask", None),
            negative_prompt_embeds=micro_batch.get("negative_prompt_embeds", None),
            negative_prompt_embeds_mask=micro_batch.get("negative_prompt_embeds_mask", None),
            micro_batch=micro_batch,
            step=0,
        )
        mse, _ = cls.forward_mse(module=module, model_inputs=model_inputs, target=target)
        return {"mse": mse, "timesteps": timesteps}
