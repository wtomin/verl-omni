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

"""Stable Diffusion 3 offline DPO tensor utilities."""

import argparse
import io

import torch
from PIL import Image

PIPELINE_KEY = "sd3"
DEFAULT_MODEL_PATH = "stabilityai/stable-diffusion-3.5-medium"


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save(tensor.detach().cpu(), buffer)
    return buffer.getvalue()


def apply_arg_defaults(args: argparse.Namespace) -> None:
    del args


def load_pipeline(args: argparse.Namespace, dtype: torch.dtype):
    from diffusers import StableDiffusion3Pipeline

    return StableDiffusion3Pipeline.from_pretrained(args.model_path, torch_dtype=dtype)


def build_generate_kwargs(args: argparse.Namespace, prompt: str, generator: torch.Generator) -> dict:
    return {
        "prompt": prompt,
        "negative_prompt": args.negative_prompt,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "max_sequence_length": args.max_sequence_length,
        "generator": generator,
    }


def encode_image_latent(pipe, image: Image.Image, args: argparse.Namespace) -> torch.Tensor:
    pixel_values = pipe.image_processor.preprocess([image], height=args.height, width=args.width)
    pixel_values = pixel_values.to(device=args.device, dtype=pipe.vae.dtype)
    with torch.no_grad():
        latents = pipe.vae.encode(pixel_values).latent_dist.sample()
    scaling_factor = getattr(pipe.vae.config, "scaling_factor", 1.0)
    shift_factor = getattr(pipe.vae.config, "shift_factor", 0.0)
    latents = (latents - shift_factor) * scaling_factor
    return latents[0].detach().cpu()


def encode_prompt_tensors(pipe, prompt: str, negative_prompt: str, args: argparse.Namespace) -> dict[str, bytes | None]:
    do_cfg = args.guidance_scale is not None and args.guidance_scale > 1.0
    with torch.no_grad():
        encoded = pipe.encode_prompt(
            prompt=[prompt],
            prompt_2=None,
            prompt_3=None,
            device=args.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=[negative_prompt] if do_cfg else None,
            negative_prompt_2=None,
            negative_prompt_3=None,
            max_sequence_length=args.max_sequence_length,
        )

    if len(encoded) == 4:
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = encoded
    elif len(encoded) == 2:
        prompt_embeds, pooled_prompt_embeds = encoded
        negative_prompt_embeds = None
        negative_pooled_prompt_embeds = None
    else:
        raise ValueError(f"Unexpected SD3 encode_prompt output length: {len(encoded)}")

    prompt_embeds_mask = torch.ones(
        prompt_embeds.shape[0],
        prompt_embeds.shape[1],
        dtype=torch.int32,
        device=prompt_embeds.device,
    )
    result = {
        "prompt_embeds": tensor_to_bytes(prompt_embeds[0]),
        "prompt_embeds_mask": tensor_to_bytes(prompt_embeds_mask[0]),
        "pooled_prompt_embeds": tensor_to_bytes(pooled_prompt_embeds[0]),
        "negative_prompt_embeds": None,
        "negative_prompt_embeds_mask": None,
        "negative_pooled_prompt_embeds": None,
    }
    if negative_prompt_embeds is not None:
        negative_prompt_embeds_mask = torch.ones(
            negative_prompt_embeds.shape[0],
            negative_prompt_embeds.shape[1],
            dtype=torch.int32,
            device=negative_prompt_embeds.device,
        )
        result["negative_prompt_embeds"] = tensor_to_bytes(negative_prompt_embeds[0])
        result["negative_prompt_embeds_mask"] = tensor_to_bytes(negative_prompt_embeds_mask[0])
        result["negative_pooled_prompt_embeds"] = tensor_to_bytes(negative_pooled_prompt_embeds[0])
    return result
