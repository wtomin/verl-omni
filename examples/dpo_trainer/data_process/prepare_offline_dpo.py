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

"""Generate offline DPO triples from one prompt file using a frozen reference pipeline.

The output schema is one logical preference pair per row:

    {prompt, negative_prompt, img_win, img_lose, win_score, lose_score}

Training expands each row to adjacent ``win, lose`` samples and re-encodes the
prompt with the diffusion text encoder.
"""

import argparse
import asyncio
import importlib.util
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

DEFAULT_SYSTEM_PROMPT = "You are a helpful image generation assistant."


def _read_prompts_from_txt(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _prompt_to_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        user_parts = []
        for message in prompt:
            if isinstance(message, dict) and message.get("role") == "user":
                user_parts.append(str(message.get("content", "")))
        if user_parts:
            return "\n".join(user_parts)
    return "" if prompt is None else str(prompt)


def _read_prompts(path: Path, prompt_key: str) -> list[str]:
    if path.suffix == ".txt":
        return _read_prompts_from_txt(path)
    if path.suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif path.suffix == ".json":
        df = pd.read_json(path)
    else:
        df = pd.read_parquet(path)
    return [_prompt_to_text(prompt) for prompt in df[prompt_key].tolist()]


def _build_messages(prompt: str, system_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def _load_reward_fn(path: str | None, name: str | None):
    if path is None or name is None:
        return None
    module_path = Path(path)
    if module_path.exists():
        spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load reward function module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module_name = path[:-3].replace("/", ".") if path.endswith(".py") else path
        module = importlib.import_module(module_name)
    return getattr(module, name)


async def _score_image(reward_fn, image: Image.Image, prompt: str, args: argparse.Namespace) -> float:
    if reward_fn is None:
        return 0.0
    image_array = np.asarray(image).astype("float32") / 255.0
    result = reward_fn(
        data_source=args.data_source,
        solution_image=image_array,
        ground_truth=prompt,
        extra_info={"raw_prompt": prompt, "prompt": prompt},
        reward_router_address=args.reward_router_address,
        model_name=args.reward_model_name,
    )
    if asyncio.iscoroutine(result):
        result = await result
    if isinstance(result, dict):
        return float(result.get("score", 0.0))
    return float(result)


def _make_generator(seed: int, device: str) -> torch.Generator:
    generator_device = device if device != "cpu" else "cpu"
    return torch.Generator(device=generator_device).manual_seed(seed)


async def _generate_split(args: argparse.Namespace, split: str) -> Path:
    from diffusers import StableDiffusion3Pipeline

    input_path = Path(os.path.expanduser(args.input_file))
    output_path = Path(os.path.expanduser(args.output_file))
    prompts = _read_prompts(input_path, args.prompt_key)
    if args.max_samples > 0:
        prompts = prompts[: args.max_samples]

    if args.image_dir is None:
        image_dir = output_path.parent / "images" / output_path.stem
    else:
        image_dir = Path(os.path.expanduser(args.image_dir))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_path, torch_dtype=dtype)
    pipe.to(args.device)
    pipe.set_progress_bar_config(disable=args.disable_progress)
    reward_fn = _load_reward_fn(args.reward_function_path, args.reward_function_name)

    rows = []
    for prompt_idx, prompt in enumerate(prompts):
        candidates = []
        for sample_idx in range(args.num_images_per_prompt):
            seed = args.seed + prompt_idx * args.num_images_per_prompt + sample_idx
            image = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                max_sequence_length=args.max_sequence_length,
                generator=_make_generator(seed, args.device),
            ).images[0]
            image_path = image_dir / f"{prompt_idx:06d}_{sample_idx:02d}.png"
            image.save(image_path)
            score = await _score_image(reward_fn, image, prompt, args)
            candidates.append({"path": str(image_path), "score": score, "seed": seed})

        candidates.sort(key=lambda item: item["score"], reverse=True)
        win = candidates[0]
        lose = candidates[-1]
        rows.append(
            {
                "data_source": args.data_source,
                "prompt": _build_messages(prompt, args.system_prompt),
                "negative_prompt": _build_messages(args.negative_prompt, args.system_prompt),
                "img_win": os.path.relpath(win["path"], output_path.parent),
                "img_lose": os.path.relpath(lose["path"], output_path.parent),
                "win_score": win["score"],
                "lose_score": lose["score"],
                "reward_model": {"style": "model", "ground_truth": prompt},
                "extra_info": {
                    "split": split,
                    "index": prompt_idx,
                    "raw_prompt": prompt,
                    "raw_negative_prompt": args.negative_prompt,
                    "num_candidates": len(candidates),
                    "win_seed": win["seed"],
                    "lose_seed": lose["seed"],
                    "candidate_scores": [item["score"] for item in candidates],
                },
            }
        )

    pd.DataFrame(rows).to_parquet(output_path)
    print(f"Wrote {len(rows)} offline DPO pairs to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate offline DPO triples with a frozen diffusion model.")
    parser.add_argument("--input_file", required=True, help="Prompt file. Supports .txt, .json, .jsonl and parquet.")
    parser.add_argument("--output_file", required=True, help="Parquet file to write.")
    parser.add_argument("--image_dir", default=None, help="Directory to write generated images.")
    parser.add_argument("--prompt_key", default="prompt", help="Prompt column for parquet/json inputs.")
    parser.add_argument("--model_path", default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--data_source", default="offline_dpo")
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--negative_prompt", default=" ")
    parser.add_argument("--num_images_per_prompt", type=int, default=4)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--max_sequence_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--reward_function_path", default=None)
    parser.add_argument("--reward_function_name", default=None)
    parser.add_argument("--reward_router_address", default=None)
    parser.add_argument("--reward_model_name", default=None)
    parser.add_argument("--disable_progress", action="store_true")
    parser.add_argument("--split", default=None, help="Optional split name stored in extra_info.split.")
    args = parser.parse_args()

    if args.num_images_per_prompt < 2:
        raise ValueError("--num_images_per_prompt must be at least 2 for DPO pair construction.")
    if (args.reward_function_path is None) != (args.reward_function_name is None):
        raise ValueError("Set both --reward_function_path and --reward_function_name, or neither.")

    output_path = Path(os.path.expanduser(args.output_file))
    split = args.split or output_path.stem
    asyncio.run(_generate_split(args, split))


if __name__ == "__main__":
    main()
