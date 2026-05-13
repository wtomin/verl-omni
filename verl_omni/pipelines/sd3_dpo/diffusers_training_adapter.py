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

import os
from typing import Optional

import torch
from diffusers.models import AutoencoderKL
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from tensordict import TensorDict
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5Tokenizer
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_name

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.workers.config import DiffusionModelConfig

__all__ = ["SD3Adapter"]


def _build_sd3_scheduler(model_path: str) -> FlowMatchEulerDiscreteScheduler:
    euler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path=model_path,
        subfolder="scheduler",
    )
    return euler


def _configure_sd3_scheduler(
    scheduler: FlowMatchEulerDiscreteScheduler,
    *,
    num_inference_steps: int,
    device: str,
) -> None:
    scheduler.set_timesteps(num_inference_steps, device=device)


@DiffusionModelBase.register("StableDiffusion3Pipeline", algorithm="dpo")
class SD3Adapter(DiffusionModelBase):
    """Training adapter for Stable Diffusion 3.x (e.g. SD3.0, SD3.5) diffusion models.

    Implements the :class:`~verl_omni.pipelines.model_base.DiffusionModelBase`
    interface for the ``StableDiffusion3Pipeline`` architecture, providing scheduler
    configuration, model-input construction, and the forward/sampling step
    used during RL training (e.g. DPO, FlowGRPO).

    Registered under ``("StableDiffusion3Pipeline", "dpo")`` so it is automatically
    selected when ``DiffusionModelConfig.architecture`` and ``algorithm`` match.
    """

    @staticmethod
    def _normalize_device(device: torch.device | int | str) -> torch.device:
        if isinstance(device, int):
            return torch.device("cuda", device)
        return torch.device(device)

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig):
        """Build and configure the flow-matching scheduler for SD3.

        Args:
            model_config (DiffusionModelConfig): Configuration for the diffusion model,
                used to determine the model path and timestep settings.

        Returns:
            FlowMatchEulerDiscreteScheduler: Scheduler with timesteps already set
                for the current device.
        """
        scheduler = _build_sd3_scheduler(model_config.local_path)
        cls.set_timesteps(scheduler, model_config, get_device_name())
        return scheduler

    @classmethod
    def set_timesteps(cls, scheduler: FlowMatchEulerDiscreteScheduler, model_config: DiffusionModelConfig, device: str):
        """Configure timesteps on the scheduler for SD3.

        Args:
            scheduler (FlowMatchEulerDiscreteScheduler): The scheduler whose timesteps
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
    def build_final_image_dpo_components(
        cls,
        model_config: DiffusionModelConfig,
        *,
        device: torch.device | int | str,
        dtype: torch.dtype,
    ) -> dict:
        device = cls._normalize_device(device)
        model_path = model_config.local_path
        local_files_only = os.path.exists(model_path)
        components = {
            "tokenizer": CLIPTokenizer.from_pretrained(
                model_path, subfolder="tokenizer", local_files_only=local_files_only
            ),
            "tokenizer_2": CLIPTokenizer.from_pretrained(
                model_path, subfolder="tokenizer_2", local_files_only=local_files_only
            ),
            "tokenizer_3": T5Tokenizer.from_pretrained(
                model_path, subfolder="tokenizer_3", local_files_only=local_files_only
            ),
            "text_encoder": CLIPTextModelWithProjection.from_pretrained(
                model_path, subfolder="text_encoder", torch_dtype=dtype, local_files_only=local_files_only
            ),
            "text_encoder_2": CLIPTextModelWithProjection.from_pretrained(
                model_path, subfolder="text_encoder_2", torch_dtype=dtype, local_files_only=local_files_only
            ),
            "text_encoder_3": T5EncoderModel.from_pretrained(
                model_path, subfolder="text_encoder_3", torch_dtype=dtype, local_files_only=local_files_only
            ),
            "vae": AutoencoderKL.from_pretrained(
                model_path, subfolder="vae", torch_dtype=dtype, local_files_only=local_files_only
            ),
        }
        for component in components.values():
            if hasattr(component, "to"):
                component.to(device)
            if hasattr(component, "eval"):
                component.eval()
            if hasattr(component, "requires_grad_"):
                component.requires_grad_(False)
        return components

    @staticmethod
    def _stringify_prompt(prompt) -> str:
        if prompt is None:
            return ""
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, dict):
            return SD3Adapter._stringify_prompt(prompt.get("content", prompt.get("prompt", "")))
        if isinstance(prompt, (list, tuple)):
            return " ".join(part for part in (SD3Adapter._stringify_prompt(item) for item in prompt) if part)
        return str(prompt)

    @classmethod
    def _get_prompt_texts(cls, micro_batch: TensorDict, key: str, batch_size: int, default: str = "") -> list[str]:
        values = tu.get_non_tensor_data(data=micro_batch, key=key, default=None)
        if values is None:
            return [default] * batch_size
        if not isinstance(values, (list, tuple)):
            values = list(values) if hasattr(values, "__len__") and not isinstance(values, str) else [values]
        texts = [cls._stringify_prompt(value) for value in values]
        if len(texts) == 1 and batch_size > 1:
            texts = texts * batch_size
        return texts[:batch_size]

    @classmethod
    def _get_clip_prompt_embeds(
        cls,
        components: dict,
        prompt: list[str],
        *,
        clip_model_index: int,
        device: torch.device | int | str,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokenizer = components["tokenizer"] if clip_model_index == 0 else components["tokenizer_2"]
        text_encoder = components["text_encoder"] if clip_model_index == 0 else components["text_encoder_2"]
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_embeds = text_encoder(text_input_ids, output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0].to(dtype=dtype, device=device)
        prompt_embeds = prompt_embeds.hidden_states[-2].to(dtype=dtype, device=device)
        return prompt_embeds, pooled_prompt_embeds

    @classmethod
    def _get_t5_prompt_embeds(
        cls,
        components: dict,
        prompt: list[str],
        *,
        max_sequence_length: int,
        joint_attention_dim: int,
        device: torch.device | int | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        tokenizer = components["tokenizer_3"]
        text_encoder = components["text_encoder_3"]
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_embeds = text_encoder(text_inputs.input_ids.to(device))[0]
        return prompt_embeds.to(dtype=dtype, device=device)

    @classmethod
    def _encode_prompt_texts(
        cls,
        module: SD3Transformer2DModel,
        components: dict,
        prompt: list[str],
        *,
        max_sequence_length: int,
        device: torch.device | int | str,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_embed, pooled_prompt_embed = cls._get_clip_prompt_embeds(
            components, prompt, clip_model_index=0, device=device, dtype=dtype
        )
        prompt_2_embed, pooled_prompt_2_embed = cls._get_clip_prompt_embeds(
            components, prompt, clip_model_index=1, device=device, dtype=dtype
        )
        clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)
        t5_prompt_embed = cls._get_t5_prompt_embeds(
            components,
            prompt,
            max_sequence_length=max_sequence_length,
            joint_attention_dim=module.config.joint_attention_dim,
            device=device,
            dtype=dtype,
        )
        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
        )
        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
        pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)
        return prompt_embeds, pooled_prompt_embeds

    @classmethod
    def _encode_response_images(
        cls,
        components: dict,
        images: torch.Tensor,
        *,
        device: torch.device | int | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        vae = components["vae"]
        images = images.to(device=device, dtype=dtype)
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.max() > 2:
            images = images / 255.0
        images = images * 2.0 - 1.0
        posterior = vae.encode(images).latent_dist
        latents = posterior.sample()
        scaling_factor = getattr(vae.config, "scaling_factor", 1.0)
        shift_factor = getattr(vae.config, "shift_factor", 0.0)
        return (latents - shift_factor) * scaling_factor

    @classmethod
    def prepare_final_image_dpo_inputs(
        cls,
        module: SD3Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        model_config: DiffusionModelConfig,
        components: dict,
        micro_batch: TensorDict,
        *,
        device: torch.device | int | str,
        dtype: torch.dtype,
    ) -> tuple[dict, dict | None, TensorDict]:
        device = cls._normalize_device(device)
        responses = micro_batch["responses"]
        with torch.no_grad():
            latents = cls._encode_response_images(components, responses, device=device, dtype=dtype)
            batch_size = latents.shape[0]
            prompt = cls._get_prompt_texts(micro_batch, "raw_prompt", batch_size)
            negative_prompt = cls._get_prompt_texts(micro_batch, "raw_negative_prompt", batch_size)
            prompt_embeds, pooled_prompt_embeds = cls._encode_prompt_texts(
                module,
                components,
                prompt,
                max_sequence_length=model_config.pipeline.max_sequence_length,
                device=device,
                dtype=dtype,
            )
            negative_prompt_embeds, pooled_negative_prompt_embeds = cls._encode_prompt_texts(
                module,
                components,
                negative_prompt,
                max_sequence_length=model_config.pipeline.max_sequence_length,
                device=device,
                dtype=dtype,
            )

            timestep_indices = torch.randint(0, len(scheduler.timesteps), (batch_size,), device=device)
            timesteps = scheduler.timesteps.to(device=device)[timestep_indices]
            sigmas = scheduler.sigmas.to(device=device, dtype=dtype)[timestep_indices]
            sigmas = sigmas.view(batch_size, *([1] * (latents.ndim - 1)))
            noise = torch.randn_like(latents)
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
            velocity_target = noise - latents

        model_inputs = {
            "hidden_states": noisy_latents,
            "timestep": timesteps,
            "encoder_hidden_states": prompt_embeds,
            "pooled_projections": pooled_prompt_embeds,
            "return_dict": False,
        }
        negative_model_inputs = {
            "hidden_states": noisy_latents,
            "timestep": timesteps,
            "encoder_hidden_states": negative_prompt_embeds,
            "pooled_projections": pooled_negative_prompt_embeds,
            "return_dict": False,
        }
        loss_data = TensorDict({"fm_velocity_target": velocity_target}, batch_size=batch_size, device=device)
        return model_inputs, negative_model_inputs, loss_data

    @classmethod
    def forward_final_image_dpo_step(
        cls,
        module: SD3Transformer2DModel,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        noise_pred = module(**model_inputs)[0]
        guidance_scale = model_config.pipeline.guidance_scale
        if guidance_scale is None:
            guidance_scale = model_config.pipeline.true_cfg_scale
        if guidance_scale > 1.0 and negative_model_inputs is not None:
            neg_noise_pred = module(**negative_model_inputs)[0]
            noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)
        return noise_pred
