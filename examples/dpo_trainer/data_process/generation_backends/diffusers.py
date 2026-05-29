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

"""In-process Diffusers generation backend for offline DPO data preparation."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pyarrow as pa
import torch
from pipeline_utils import get_pipeline_utils

from .common import ScoreImages, run_split_loop, score_and_write_dpo_row


async def generate_split(
    args: argparse.Namespace,
    split: str,
    *,
    prompts: list[str],
    output_path: Path,
    image_dir: Path,
    start_idx: int,
    resume_base_table: pa.Table | None,
    score_images: ScoreImages,
) -> Path:
    pipeline_utils = get_pipeline_utils(args)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    pipe = pipeline_utils.load_pipeline(args, dtype)
    pipe.to(args.device)
    pipe.set_progress_bar_config(disable=args.disable_progress)

    async def process_prompt(writer, prompt_idx: int, prompt: str, seeds: list[int]) -> None:
        generators = [
            torch.Generator(device=args.device if args.device != "cpu" else "cpu").manual_seed(seed) for seed in seeds
        ]
        gen_t0 = time.perf_counter()
        images = pipe(**pipeline_utils.build_generate_kwargs(args, prompt, generators)).images
        batch_gen_s = time.perf_counter() - gen_t0
        per_image_batch_gen_s = batch_gen_s / len(images)

        generated = []
        generation_latency_s = []
        for sample_idx, (image, seed) in enumerate(zip(images, seeds, strict=True)):
            post_t0 = time.perf_counter()
            image_path = image_dir / f"{prompt_idx:06d}_{sample_idx:02d}.png"
            image.save(image_path)
            generated.append(
                {
                    "path": str(image_path),
                    "image": image,
                    "seed": seed,
                    "latents": pipeline_utils.tensor_to_bytes(pipeline_utils.encode_image_latent(pipe, image, args)),
                }
            )
            generation_latency_s.append(per_image_batch_gen_s + (time.perf_counter() - post_t0))
        await score_and_write_dpo_row(
            writer,
            args=args,
            split=split,
            prompt_idx=prompt_idx,
            prompt=prompt,
            output_path=output_path,
            prompt_tensors=pipeline_utils.encode_prompt_tensors(pipe, prompt, args.negative_prompt, args),
            generated=generated,
            score_images=score_images,
            generation_latency_s=generation_latency_s,
        )

    await run_split_loop(
        args,
        prompts=prompts,
        output_path=output_path,
        start_idx=start_idx,
        resume_base_table=resume_base_table,
        process_prompt=process_prompt,
    )
    return output_path
