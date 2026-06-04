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

"""Qwen-Image offline DPO tensor utilities."""

import argparse
import inspect
import io

import torch
from PIL import Image

from .sd3 import DEFAULT_MODEL_PATH as SD3_DEFAULT_MODEL_PATH

PIPELINE_KEY = "qwen_image"
DEFAULT_MODEL_PATH = "Qwen/Qwen-Image"


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save(tensor.detach().cpu(), buffer)
    return buffer.getvalue()


def apply_arg_defaults(args: argparse.Namespace) -> None:
    if args.model_path == SD3_DEFAULT_MODEL_PATH:
        args.model_path = DEFAULT_MODEL_PATH
    if args.guidance_scale == 4.0:
        args.guidance_scale = 1.0


def load_pipeline(args: argparse.Namespace, dtype: torch.dtype):
    from diffusers import QwenImagePipeline

    return QwenImagePipeline.from_pretrained(args.model_path, torch_dtype=dtype)


def build_generate_kwargs(
    args: argparse.Namespace, prompt: str, generator: torch.Generator | list[torch.Generator]
) -> dict:
    return {
        "prompt": prompt,
        "negative_prompt": args.negative_prompt,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "true_cfg_scale": args.true_cfg_scale,
        "max_sequence_length": args.max_sequence_length,
        "num_images_per_prompt": args.num_images_per_prompt,
        "generator": generator,
    }


def _normalize_latents(pipe, latents: torch.Tensor) -> torch.Tensor:
    latents_mean = getattr(pipe.vae.config, "latents_mean", None)
    latents_std = getattr(pipe.vae.config, "latents_std", None)
    if latents_mean is None or latents_std is None:
        scaling_factor = getattr(pipe.vae.config, "scaling_factor", 1.0)
        shift_factor = getattr(pipe.vae.config, "shift_factor", 0.0)
        return (latents - shift_factor) * scaling_factor

    view_shape = (1, -1, *([1] * (latents.ndim - 2)))
    mean = torch.tensor(latents_mean, device=latents.device, dtype=latents.dtype).view(view_shape)
    std = torch.tensor(latents_std, device=latents.device, dtype=latents.dtype).view(view_shape)
    return (latents - mean) / std


def _retrieve_vae_latents(encoder_output) -> torch.Tensor:
    latents = getattr(encoder_output, "latents", None)
    if latents is not None:
        return latents

    latent_dist = getattr(encoder_output, "latent_dist", None)
    if latent_dist is None:
        raise AttributeError("Qwen-Image VAE encode output does not contain `latents` or `latent_dist`.")

    mode = getattr(latent_dist, "mode", None)
    if callable(mode):
        return mode()
    return latent_dist.sample()


def _pack_latents(pipe, latents: torch.Tensor) -> torch.Tensor:
    pack_latents = getattr(pipe, "_pack_latents", None)
    if pack_latents is None:
        raise AttributeError("QwenImagePipeline does not expose `_pack_latents`; cannot prepare offline DPO latents.")

    batch_size = latents.shape[0]
    num_channels = latents.shape[2] if latents.ndim == 5 else latents.shape[1]
    latent_height = latents.shape[-2]
    latent_width = latents.shape[-1]

    try:
        parameter_count = len(inspect.signature(pack_latents).parameters)
    except (TypeError, ValueError):
        parameter_count = 5

    if parameter_count == 5:
        return pack_latents(latents, batch_size, num_channels, latent_height, latent_width)
    if parameter_count == 4:
        return pack_latents(latents, batch_size, latent_height, latent_width)
    return pack_latents(latents)


def encode_image_latent(pipe, image: Image.Image, args: argparse.Namespace) -> torch.Tensor:
    pixel_values = pipe.image_processor.preprocess([image], height=args.height, width=args.width)
    pixel_values = pixel_values.to(device=args.device, dtype=pipe.vae.dtype)
    if pixel_values.ndim == 4:
        pixel_values = pixel_values.unsqueeze(2)
    with torch.no_grad():
        latents = _retrieve_vae_latents(pipe.vae.encode(pixel_values))
    latents = _normalize_latents(pipe, latents)
    if latents.ndim == 5:
        if latents.shape[2] != 1:
            raise ValueError(f"Expected single-frame Qwen-Image latents, got shape {tuple(latents.shape)}.")
        latents = latents.squeeze(2)
    latents = _pack_latents(pipe, latents)
    return latents[0].detach().cpu()


def _call_encode_prompt(pipe, prompt: str, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    encode_kwargs = {
        "prompt": [prompt],
        "device": args.device,
        "num_images_per_prompt": 1,
        "max_sequence_length": args.max_sequence_length,
    }
    try:
        encoded = pipe.encode_prompt(**encode_kwargs)
    except TypeError:
        encode_kwargs["prompt"] = prompt
        encoded = pipe.encode_prompt(**encode_kwargs)

    if isinstance(encoded, tuple):
        prompt_embeds = encoded[0]
        prompt_embeds_mask = encoded[1] if len(encoded) > 1 else None
    else:
        prompt_embeds = encoded
        prompt_embeds_mask = None
    if not isinstance(prompt_embeds, torch.Tensor):
        raise ValueError("Unexpected Qwen-Image encode_prompt output; expected prompt embeds tensor.")
    return prompt_embeds, prompt_embeds_mask


def encode_prompt_tensors(pipe, prompt: str, negative_prompt: str, args: argparse.Namespace) -> dict[str, bytes | None]:
    with torch.no_grad():
        prompt_embeds, prompt_embeds_mask = _call_encode_prompt(pipe, prompt, args)

    result = {
        "prompt_embeds": tensor_to_bytes(prompt_embeds[0]),
        "prompt_embeds_mask": tensor_to_bytes(prompt_embeds_mask[0].to(dtype=torch.int32))
        if prompt_embeds_mask is not None
        else None,
        "negative_prompt_embeds": None,
        "negative_prompt_embeds_mask": None,
    }
    if args.true_cfg_scale is not None and args.true_cfg_scale > 1.0:
        with torch.no_grad():
            negative_prompt_embeds, negative_prompt_embeds_mask = _call_encode_prompt(pipe, negative_prompt, args)
        result["negative_prompt_embeds"] = tensor_to_bytes(negative_prompt_embeds[0])
        result["negative_prompt_embeds_mask"] = (
            tensor_to_bytes(negative_prompt_embeds_mask[0].to(dtype=torch.int32))
            if negative_prompt_embeds_mask is not None
            else None
        )
    return result
