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

"""Shared offline DPO split loop: parquet checkpointing, scoring, row writes."""

from __future__ import annotations

import argparse
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pyarrow as pa
import torch
from parquet_checkpoint_utils import ChunkedParquetWriter, ParquetWriterShutdownGuard

GENERATION_SERVER_CHOICES = ("diffusers", "vllm_omni")
VLLM_OMNI_PIPELINES = ("qwen_image", "sd3")

PromptHandler = Callable[
    [ChunkedParquetWriter, int, str, list[int]],
    Awaitable[None],
]
ScoreImages = Callable[[list[Any], str], Awaitable[tuple[list[float], list[float]]]]


def log_per_image_latencies(
    *,
    prompt_idx: int,
    seeds: list[int],
    generation_latency_s: list[float],
    scoring_latency_s: list[float],
) -> None:
    """Print per-candidate generation and reward scoring latency."""
    for sample_idx, (seed, gen_s, score_s) in enumerate(
        zip(seeds, generation_latency_s, scoring_latency_s, strict=True)
    ):
        print(
            f"[prompt {prompt_idx:06d} image {sample_idx:02d} seed={seed}] "
            f"generation_latency={gen_s:.3f}s scoring_latency={score_s:.3f}s"
        )


def build_messages(prompt: str, system_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def _sample_seeds(rng: torch.Generator, count: int) -> list[int]:
    return [int(torch.randint(0, 2**31, (1,), generator=rng).item()) for _ in range(count)]


async def run_split_loop(
    args: argparse.Namespace,
    *,
    prompts: list[str],
    output_path: Path,
    start_idx: int,
    resume_base_table: pa.Table | None,
    process_prompt: PromptHandler,
) -> None:
    """Iterate prompts with resume-aware seeding; ``process_prompt`` writes each DPO row."""
    seed_rng = torch.Generator().manual_seed(args.seed)
    for _ in range(start_idx):
        _sample_seeds(seed_rng, args.num_images_per_prompt)

    with ChunkedParquetWriter(output_path, args.parquet_flush_every, base_table=resume_base_table) as writer:
        with ParquetWriterShutdownGuard(writer):
            for prompt_idx in range(start_idx, len(prompts)):
                seeds = _sample_seeds(seed_rng, args.num_images_per_prompt)
                await process_prompt(writer, prompt_idx, prompts[prompt_idx], seeds)


async def score_and_write_dpo_row(
    writer: ChunkedParquetWriter,
    *,
    args: argparse.Namespace,
    split: str,
    prompt_idx: int,
    prompt: str,
    output_path: Path,
    prompt_tensors: dict[str, bytes | None],
    generated: list[dict[str, Any]],
    score_images: ScoreImages,
    generation_latency_s: list[float],
) -> None:
    """Rank candidates by reward score and append one win/lose parquet row."""
    scores, scoring_latency_s = await score_images([item["image"] for item in generated], prompt)
    log_per_image_latencies(
        prompt_idx=prompt_idx,
        seeds=[item["seed"] for item in generated],
        generation_latency_s=generation_latency_s,
        scoring_latency_s=scoring_latency_s,
    )
    candidates = [
        {"path": item["path"], "latents": item["latents"], "score": score, "seed": item["seed"]}
        for item, score in zip(generated, scores, strict=True)
    ]
    candidates.sort(key=lambda item: item["score"], reverse=True)
    win, lose = candidates[0], candidates[-1]
    writer.write(
        {
            "data_source": args.data_source,
            "pipeline": args.pipeline,
            "prompt": build_messages(prompt, args.system_prompt),
            "negative_prompt": build_messages(args.negative_prompt, args.system_prompt),
            "img_win": os.path.relpath(win["path"], output_path.parent),
            "img_lose": os.path.relpath(lose["path"], output_path.parent),
            "img_win_latents": win["latents"],
            "img_lose_latents": lose["latents"],
            **prompt_tensors,
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
                "generation_server": args.generation_server,
            },
        }
    )
    writer.commit_checkpoint()


def pack_generated_samples(
    raw: list[dict[str, Any]],
    *,
    prompt_idx: int,
    image_dir: Path,
    pipeline_utils,
) -> list[dict[str, Any]]:
    """Save images to disk and normalize latents to parquet bytes."""
    generated: list[dict[str, Any]] = []
    for sample_idx, item in enumerate(raw):
        image_path = image_dir / f"{prompt_idx:06d}_{sample_idx:02d}.png"
        item["image"].save(image_path)
        latents = item["latents"]
        if not isinstance(latents, bytes | bytearray):
            latents = pipeline_utils.tensor_to_bytes(latents)
        row = {
            "path": str(image_path),
            "image": item["image"],
            "seed": item["seed"],
            "latents": latents,
        }
        for key, value in item.items():
            if key not in {"image", "seed", "latents", "path"}:
                row[key] = value
        generated.append(row)
    return generated


def pack_rollout_prompt_tensors(sample: dict[str, Any], pipeline_utils, pipeline: str) -> dict[str, bytes | None]:
    """Serialize prompt embeddings returned by a vLLM-Omni DPO rollout."""

    def _bytes(tensor: Any, *, as_int_mask: bool = False) -> bytes | None:
        if tensor is None:
            return None
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)
        if as_int_mask:
            tensor = tensor.to(dtype=torch.int32)
        return pipeline_utils.tensor_to_bytes(tensor)

    tensors = {
        "prompt_embeds": _bytes(sample["prompt_embeds"]),
        "prompt_embeds_mask": _bytes(sample["prompt_embeds_mask"], as_int_mask=True),
        "negative_prompt_embeds": _bytes(sample["negative_prompt_embeds"]),
        "negative_prompt_embeds_mask": _bytes(sample["negative_prompt_embeds_mask"], as_int_mask=True),
    }
    if pipeline == "sd3":
        tensors["pooled_prompt_embeds"] = _bytes(sample.get("pooled_prompt_embeds"))
        tensors["negative_pooled_prompt_embeds"] = _bytes(sample.get("negative_pooled_prompt_embeds"))
    return tensors
