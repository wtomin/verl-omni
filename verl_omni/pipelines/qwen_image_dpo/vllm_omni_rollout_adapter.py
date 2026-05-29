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

"""Qwen-Image rollout adapter for diffusion DPO.

Unlike FlowGRPO, DPO rollout runs a deterministic Euler denoise loop (no SDE
window, no per-step log-probabilities).  The returned ``custom_output`` carries
only the tensors needed to build offline-style training batches:

* ``latents`` — packed latents after the full Euler denoise loop (before VAE decode).
* ``prompt_embeds``, ``prompt_embeds_mask``, ``negative_prompt_embeds``,
  ``negative_prompt_embeds_mask`` — text-encoder outputs for the training pass.
"""

import os
from typing import Any

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.qwen_image import QwenImagePipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest

from verl_omni.pipelines.model_base import VllmOmniPipelineBase

from .common import apply_true_cfg, build_img_shapes, coalesce_not_none, maybe_to_cpu

__all__ = ["QwenImageDPOPipeline"]


def _debug_generation_enabled() -> bool:
    return os.getenv("VERL_OMNI_DEBUG_GENERATION", "").lower() in {"1", "true", "yes"}


def _debug_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "shape": tuple(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, list):
        return f"list(len={len(value)})"
    return value


def _debug_generation(message: str, **fields: Any) -> None:
    if not _debug_generation_enabled():
        return
    formatted = " ".join(f"{key}={_debug_value(value)}" for key, value in fields.items())
    print(f"[QwenImageDPOPipeline] {message} {formatted}", flush=True)


def _debug_cuda_sync(stage: str) -> None:
    if not _debug_generation_enabled() or not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception as exc:
        raise RuntimeError(f"CUDA failure after QwenImageDPOPipeline stage: {stage}") from exc


@VllmOmniPipelineBase.register("QwenImagePipeline", algorithm="dpo")
class QwenImageDPOPipeline(QwenImagePipeline):
    """Rollout pipeline for Qwen-Image Diffusion-DPO (Euler ODE, no log-probs)."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.device = get_local_device()
        model = od_config.model
        local_files_only = os.path.exists(model)

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )

    def _get_qwen_prompt_embeds(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
    ):
        dtype = dtype or self.text_encoder.dtype

        if attention_mask is None:
            attention_mask = torch.ones_like(prompt_ids, dtype=torch.long)

        prompt_ids = prompt_ids.unsqueeze(0) if prompt_ids.ndim == 1 else prompt_ids
        attention_mask = attention_mask.unsqueeze(0) if attention_mask.ndim == 1 else attention_mask
        drop_idx = self.prompt_template_encode_start_idx
        encoder_hidden_states = self.text_encoder(
            input_ids=prompt_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            output_hidden_states=True,
        )
        hidden_states = encoder_hidden_states.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        kept_lengths = [e.size(0) for e in split_hidden_states]
        _debug_generation(
            "prompt embedding prefix drop",
            prompt_ids=prompt_ids,
            attention_mask=attention_mask,
            drop_idx=drop_idx,
            kept_lengths=kept_lengths,
        )
        if any(length <= 0 for length in kept_lengths):
            raise ValueError(
                "Qwen-Image prompt is empty after dropping the chat-template prefix: "
                f"drop_idx={drop_idx}, kept_lengths={kept_lengths}."
            )
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype)
        _debug_cuda_sync("encode_prompt")

        return prompt_embeds, encoder_attention_mask

    def encode_prompt(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        max_sequence_length: int = 1024,
    ):
        prompt_ids = prompt_ids.unsqueeze(0) if prompt_ids.ndim == 1 else prompt_ids
        attention_mask = (
            attention_mask.unsqueeze(0) if attention_mask is not None and attention_mask.ndim == 1 else attention_mask
        )

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompt_ids, attention_mask=attention_mask)

        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

        if num_images_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            prompt_embeds_mask = prompt_embeds_mask.repeat_interleave(num_images_per_prompt, dim=0)

        return prompt_embeds, prompt_embeds_mask

    def diffuse(
        self,
        prompt_embeds,
        prompt_embeds_mask,
        negative_prompt_embeds,
        negative_prompt_embeds_mask,
        latents,
        img_shapes,
        txt_seq_lens,
        negative_txt_seq_lens,
        timesteps,
        do_true_cfg,
        guidance,
        true_cfg_scale,
        generator,
    ):
        """Run the full Euler denoise loop and return final latents."""
        self.scheduler.set_begin_index(0)
        for step_idx, timestep_value in enumerate(timesteps):
            if self.interrupt:
                continue

            self._current_timestep = timestep_value
            timestep = timestep_value.expand(latents.shape[0]).to(device=latents.device, dtype=latents.dtype)
            x = latents.to(self.transformer.img_in.weight.dtype)
            _debug_generation(
                "denoise step start",
                step_idx=step_idx,
                timestep=timestep_value,
                latents=latents,
                x=x,
                prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_embeds_mask,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                do_true_cfg=do_true_cfg,
            )

            self.transformer.do_true_cfg = do_true_cfg
            noise_pred = self.transformer(
                hidden_states=x,
                timestep=timestep / 1000,
                guidance=guidance,
                encoder_hidden_states_mask=prompt_embeds_mask,
                encoder_hidden_states=prompt_embeds,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                attention_kwargs=self.attention_kwargs,
                return_dict=False,
            )[0]
            _debug_cuda_sync(f"transformer_positive_step_{step_idx}")
            if do_true_cfg:
                neg_noise_pred = self.transformer(
                    hidden_states=x,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=negative_prompt_embeds_mask,
                    encoder_hidden_states=negative_prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=negative_txt_seq_lens,
                    attention_kwargs=self.attention_kwargs,
                    return_dict=False,
                )[0]
                _debug_cuda_sync(f"transformer_negative_step_{step_idx}")
                noise_pred = apply_true_cfg(noise_pred, neg_noise_pred, true_cfg_scale)
                _debug_cuda_sync(f"true_cfg_step_{step_idx}")

            step_output = self.scheduler.step(
                noise_pred.float(),
                timestep_value,
                latents,
                generator=generator,
                return_dict=False,
            )
            latents = step_output[0] if isinstance(step_output, tuple) else step_output
            _debug_cuda_sync(f"scheduler_step_{step_idx}")

        return latents

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt_ids: torch.Tensor | list[int] | None = None,
        prompt_mask: torch.Tensor | None = None,
        negative_prompt_ids: torch.Tensor | list[int] | None = None,
        negative_prompt_mask: torch.Tensor | None = None,
        true_cfg_scale: float = 4.0,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float = 1.0,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
        output_type: str | None = "pil",
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end_tensor_inputs: tuple[str, ...] = ("latents",),
        max_sequence_length: int = 512,
        **kwargs: Any,
    ) -> DiffusionOutput:
        del kwargs, callback_on_step_end_tensor_inputs

        custom_prompt = req.prompts[0] if req.prompts else {}
        if isinstance(custom_prompt, dict):
            prompt_ids = custom_prompt.get("prompt_ids", prompt_ids)
            prompt_mask = custom_prompt.get("prompt_mask", prompt_mask)
            negative_prompt_ids = custom_prompt.get("negative_prompt_ids", negative_prompt_ids)
            negative_prompt_mask = custom_prompt.get("negative_prompt_mask", negative_prompt_mask)

        sampling_params = req.sampling_params
        height = sampling_params.height or self.default_sample_size * self.vae_scale_factor
        width = sampling_params.width or self.default_sample_size * self.vae_scale_factor
        num_inference_steps = sampling_params.num_inference_steps or num_inference_steps
        max_sequence_length = sampling_params.max_sequence_length or max_sequence_length

        generator = sampling_params.generator or generator
        if generator is None and sampling_params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(sampling_params.seed)
        true_cfg_scale = coalesce_not_none(sampling_params.true_cfg_scale, true_cfg_scale)
        req_num_outputs = getattr(sampling_params, "num_outputs_per_prompt", None)
        if req_num_outputs and req_num_outputs > 0:
            num_images_per_prompt = req_num_outputs

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        if prompt_ids is not None:
            if isinstance(prompt_ids, list):
                prompt_ids = torch.tensor(prompt_ids, device=self.device)
            batch_size = prompt_ids.shape[0] if prompt_ids.ndim == 2 else 1
        elif prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        else:
            return DiffusionOutput(output=None, custom_output={})

        if isinstance(negative_prompt_ids, list):
            negative_prompt_ids = torch.tensor(negative_prompt_ids, device=self.device)

        has_neg_prompt = negative_prompt_ids is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        _debug_generation(
            "forward inputs",
            prompt_ids=prompt_ids,
            negative_prompt_ids=negative_prompt_ids,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            true_cfg_scale=true_cfg_scale,
            guidance_scale=guidance_scale,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            do_true_cfg=do_true_cfg,
        )

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt_ids=prompt_ids,
            attention_mask=prompt_mask,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt_ids=negative_prompt_ids,
                attention_mask=negative_prompt_mask,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )
        else:
            negative_prompt_embeds = None
            negative_prompt_embeds_mask = None
        _debug_generation(
            "encoded prompts",
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
        )

        num_channels_latents = self.transformer.in_channels // 4
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            generator,
            latents,
        )
        _debug_cuda_sync("prepare_latents")
        img_shapes = build_img_shapes(height, width, batch_size, self.vae_scale_factor)
        timesteps, num_inference_steps = self.prepare_timesteps(num_inference_steps, sigmas, latents.shape[1])
        self._num_timesteps = len(timesteps)
        _debug_generation(
            "prepared diffusion inputs",
            latents=latents,
            img_shapes=img_shapes,
            timesteps=timesteps,
            num_inference_steps=num_inference_steps,
        )

        if self.transformer.guidance_embeds:
            guidance = torch.full([1], guidance_scale, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
        )
        _debug_generation("sequence lengths", txt_seq_lens=txt_seq_lens, negative_txt_seq_lens=negative_txt_seq_lens)

        latents = self.diffuse(
            prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
            latents,
            img_shapes,
            txt_seq_lens,
            negative_txt_seq_lens,
            timesteps,
            do_true_cfg,
            guidance,
            true_cfg_scale,
            generator,
        )
        rollout_latents = latents.detach().clone()

        self._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = latents.to(self.vae.dtype)
            _debug_generation("vae decode inputs", latents=latents)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            _debug_cuda_sync("vae_decode")

        return DiffusionOutput(
            output=maybe_to_cpu(image),
            custom_output={
                "latents": maybe_to_cpu(rollout_latents),
                "prompt_embeds": maybe_to_cpu(prompt_embeds),
                "prompt_embeds_mask": maybe_to_cpu(prompt_embeds_mask),
                "negative_prompt_embeds": maybe_to_cpu(negative_prompt_embeds),
                "negative_prompt_embeds_mask": maybe_to_cpu(negative_prompt_embeds_mask),
            },
        )
