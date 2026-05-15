#!/usr/bin/env python3
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
"""Validate SD3 DPO agent-loop rollout outputs on real test prompts.

This script is intentionally standalone: it reads a real prompt parquet, runs
the same vLLM-Omni SD3 DPO custom pipeline used by ``diffusion_single_turn_agent``,
then verifies the rollout tensors can drive the SD3 DPO training adapter and
DPO loss on GPU.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from tensordict import TensorDict

from verl_omni.agent_loop.prompt_utils import stringify_prompt_messages
from verl_omni.pipelines.sd3_dpo.diffusers_training_adapter import StableDiffusion3DPO
from verl_omni.trainer.diffusion.diffusion_algos import compute_diffusion_loss_dpo
from verl_omni.workers.config import DiffusionActorConfig, DiffusionLossConfig

CUSTOM_PIPELINE_CLASS = "verl_omni.pipelines.sd3_dpo.vllm_omni_rollout_adapter.StableDiffusion3DPOPipeline"
REQUIRED_CUSTOM_OUTPUT_KEYS = ("image_latents", "prompt_embeds", "pooled_prompt_embeds")


@dataclass(frozen=True)
class PromptCase:
    uid: str
    prompt: str
    negative_prompt: str
    row_index: int


@dataclass(frozen=True)
class RolloutSample:
    uid: str
    sample_index: int
    score: float
    custom_output: dict
    image_paths: tuple[Path, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="datasets/pickscore/test.parquet",
        help="Real test parquet path. Defaults to the PickScore test split used by the DPO example.",
    )
    parser.add_argument("--model", default="stabilityai/stable-diffusion-3.5-medium", help="SD3/SD3.5 model path.")
    parser.add_argument("--num-prompts", type=int, default=1, help="Number of test prompts to validate.")
    parser.add_argument(
        "--num-samples-per-prompt",
        type=int,
        default=2,
        help="Rollout samples per prompt. Must be >=2 so DPO chosen/rejected pairs can be formed.",
    )
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--dpo-beta", type=float, default=2000.0)
    parser.add_argument(
        "--output-dir",
        default="outputs/sd3_dpo_agentloop_validation",
        help="Directory used to save generated rollout images.",
    )
    parser.add_argument(
        "--skip-training-forward",
        action="store_true",
        help="Only validate rollout custom_output. By default, the script also loads the diffusers transformer.",
    )
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_prompt_cases(args: argparse.Namespace) -> list[PromptCase]:
    data_path = Path(args.data).expanduser()
    if not data_path.exists():
        raise FileNotFoundError(f"Real test parquet not found: {data_path}")

    dataframe = pd.read_parquet(data_path)
    cases: list[PromptCase] = []
    for row_index, row in dataframe.iterrows():
        prompt = stringify_prompt_messages(row.get("prompt"))
        negative_prompt = stringify_prompt_messages(row.get("negative_prompt", ""))
        if not prompt:
            continue
        cases.append(
            PromptCase(
                uid=f"prompt-{row_index}",
                prompt=prompt,
                negative_prompt=negative_prompt,
                row_index=int(row_index),
            )
        )
        if len(cases) >= args.num_prompts:
            break

    if len(cases) < args.num_prompts:
        raise ValueError(f"Only found {len(cases)} non-empty prompts in {data_path}; need {args.num_prompts}.")
    return cases


def server_custom_prompt(prompt: str, negative_prompt: str) -> dict:
    from verl.utils.tokenizer import normalize_token_ids

    custom_prompt = {"prompt_ids": normalize_token_ids([0]), "prompt": prompt}
    if negative_prompt is not None:
        custom_prompt["negative_prompt"] = negative_prompt
    return custom_prompt


def server_sampling_params(args: argparse.Namespace, seed: int):
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    sampling_params = {
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "max_sequence_length": args.max_sequence_length,
        "seed": seed,
        "logprobs": False,
    }
    sampling_kwargs = {}
    extra_args = {}
    for key, value in sampling_params.items():
        if hasattr(OmniDiffusionSamplingParams, key):
            sampling_kwargs[key] = value
        else:
            extra_args[key] = value
    sampling_kwargs["extra_args"] = extra_args
    return OmniDiffusionSamplingParams(**sampling_kwargs)


def validate_custom_output(custom_output: dict, *, expect_negative: bool) -> None:
    for key in REQUIRED_CUSTOM_OUTPUT_KEYS:
        value = custom_output.get(key)
        if not isinstance(value, torch.Tensor):
            raise AssertionError(f"custom_output[{key!r}] must be a tensor, got {type(value)!r}")
        if value.device.type != "cpu":
            raise AssertionError(f"custom_output[{key!r}] should be on CPU after rollout, got {value.device}")
        if not torch.isfinite(value.float()).all():
            raise AssertionError(f"custom_output[{key!r}] contains non-finite values")

    if custom_output["image_latents"].ndim != 4:
        raise AssertionError(f"image_latents should be BCHW, got {tuple(custom_output['image_latents'].shape)}")
    if custom_output["prompt_embeds"].ndim != 3:
        raise AssertionError(f"prompt_embeds should be BLD, got {tuple(custom_output['prompt_embeds'].shape)}")
    if custom_output["pooled_prompt_embeds"].ndim != 2:
        raise AssertionError(
            f"pooled_prompt_embeds should be BD, got {tuple(custom_output['pooled_prompt_embeds'].shape)}"
        )

    if expect_negative:
        for key in ("negative_prompt_embeds", "negative_pooled_prompt_embeds"):
            value = custom_output.get(key)
            if not isinstance(value, torch.Tensor):
                raise AssertionError(f"guidance_scale > 1 requires custom_output[{key!r}]")


def score_for_pair_selection(custom_output: dict) -> float:
    """A deterministic local score used only to form chosen/rejected DPO pairs."""
    latents = custom_output["image_latents"].float()
    return -latents.square().mean().item()


def _to_uint8_hwc_array(image: Any) -> np.ndarray:
    array = image.detach().cpu().float().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3:
        raise ValueError(f"Cannot convert image with shape {array.shape} to HWC image")

    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, 1.0) * 255.0
    array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[-1] == 1:
        array = array[..., 0]
    return array


def save_image(image: Any, path: Path) -> None:
    if hasattr(image, "save"):
        image.save(path)
        return
    if isinstance(image, dict):
        for key in ("image", "pil_image", "data"):
            if key in image:
                save_image(image[key], path)
                return
    for attr in ("image", "pil_image", "data"):
        if hasattr(image, attr):
            save_image(getattr(image, attr), path)
            return

    from PIL import Image

    Image.fromarray(_to_uint8_hwc_array(image)).save(path)


def save_generated_images(images: Any, *, args: argparse.Namespace, prompt_case: PromptCase, sample_index: int) -> tuple[Path, ...]:
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_items = images if isinstance(images, Sequence) and not isinstance(images, (str, bytes)) else [images]
    saved_paths: list[Path] = []
    for image_index, image in enumerate(image_items):
        path = output_dir / (
            f"prompt_{prompt_case.row_index:06d}_{prompt_case.uid}_"
            f"sample_{sample_index:02d}_image_{image_index:02d}.png"
        )
        save_image(image, path)
        saved_paths.append(path)
    return tuple(saved_paths)


async def run_rollouts(args: argparse.Namespace, prompt_cases: list[PromptCase]) -> list[RolloutSample]:
    from vllm_omni.entrypoints.async_omni import AsyncOmni

    engine = AsyncOmni(
        model=args.model,
        custom_pipeline_args={"pipeline_class": CUSTOM_PIPELINE_CLASS},
        enforce_eager=True,
        dtype=args.dtype,
    )
    samples: list[RolloutSample] = []
    try:
        for prompt_case in prompt_cases:
            for sample_index in range(args.num_samples_per_prompt):
                final_output = None
                sample_seed = args.seed + prompt_case.row_index * 1000 + sample_index
                async for output in engine.generate(
                    prompt=server_custom_prompt(prompt_case.prompt, prompt_case.negative_prompt),
                    request_id=f"sd3_dpo_validate_{uuid4().hex[:8]}",
                    sampling_params_list=[server_sampling_params(args, seed=sample_seed)],
                    output_modalities=["image"],
                ):
                    final_output = output

                if final_output is None:
                    raise AssertionError("vLLM-Omni generation produced no final output")
                if not final_output.images:
                    raise AssertionError("vLLM-Omni generation produced no image")

                custom_output = final_output.custom_output or {}
                validate_custom_output(custom_output, expect_negative=args.guidance_scale > 1.0)
                image_paths = save_generated_images(
                    final_output.images,
                    args=args,
                    prompt_case=prompt_case,
                    sample_index=sample_index,
                )
                samples.append(
                    RolloutSample(
                        uid=prompt_case.uid,
                        sample_index=sample_index,
                        score=score_for_pair_selection(custom_output),
                        custom_output=custom_output,
                        image_paths=image_paths,
                    )
                )
                print(
                    f"[rollout] uid={prompt_case.uid} sample={sample_index} "
                    f"score={samples[-1].score:.6f} latents={tuple(custom_output['image_latents'].shape)} "
                    f"saved={','.join(str(path) for path in image_paths)}"
                )
    finally:
        engine.shutdown()
    return samples


def select_adjacent_dpo_pairs(samples: list[RolloutSample]) -> list[RolloutSample]:
    by_uid: dict[str, list[RolloutSample]] = {}
    for sample in samples:
        by_uid.setdefault(sample.uid, []).append(sample)

    pairs: list[RolloutSample] = []
    for uid, group in by_uid.items():
        if len(group) < 2:
            raise ValueError(f"DPO validation needs at least two samples for {uid}")
        ordered = sorted(group, key=lambda item: item.score, reverse=True)
        chosen = ordered[0]
        rejected = ordered[-1]
        if chosen.score < rejected.score:
            raise AssertionError(f"Invalid pair ordering for {uid}: {chosen.score} < {rejected.score}")
        pairs.extend([chosen, rejected])
        print(
            f"[pair] uid={uid} chosen_sample={chosen.sample_index} rejected_sample={rejected.sample_index} "
            f"chosen_score={chosen.score:.6f} rejected_score={rejected.score:.6f}"
        )
    return pairs


def stack_custom_outputs(samples: list[RolloutSample], key: str, *, device: str, dtype: torch.dtype) -> torch.Tensor:
    tensors = [sample.custom_output[key].to(device=device, dtype=dtype) for sample in samples]
    return torch.cat(tensors, dim=0)


def run_training_and_dpo_checks(args: argparse.Namespace, paired_samples: list[RolloutSample]) -> None:
    from diffusers import FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel

    device = "cuda"
    dtype = torch_dtype(args.dtype)
    model_config = SimpleNamespace(
        local_path=args.model,
        pipeline=SimpleNamespace(
            height=args.height,
            width=args.width,
            max_sequence_length=args.max_sequence_length,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        ),
    )

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model, subfolder="scheduler")
    StableDiffusion3DPO.set_timesteps(scheduler, model_config, device)
    transformer = SD3Transformer2DModel.from_pretrained(
        args.model,
        subfolder="transformer",
        torch_dtype=dtype,
    ).to(device=device)
    transformer.eval()

    micro_batch_data = {
        "image_latents": stack_custom_outputs(paired_samples, "image_latents", device=device, dtype=dtype),
        "prompt_embeds": stack_custom_outputs(paired_samples, "prompt_embeds", device=device, dtype=dtype),
        "pooled_prompt_embeds": stack_custom_outputs(
            paired_samples,
            "pooled_prompt_embeds",
            device=device,
            dtype=dtype,
        ),
    }
    if args.guidance_scale > 1.0:
        micro_batch_data["negative_prompt_embeds"] = stack_custom_outputs(
            paired_samples,
            "negative_prompt_embeds",
            device=device,
            dtype=dtype,
        )
        micro_batch_data["negative_pooled_prompt_embeds"] = stack_custom_outputs(
            paired_samples,
            "negative_pooled_prompt_embeds",
            device=device,
            dtype=dtype,
        )

    with torch.no_grad():
        model_output = StableDiffusion3DPO.forward_batch(
            module=transformer,
            scheduler=scheduler,
            model_config=model_config,
            micro_batch=TensorDict(micro_batch_data, batch_size=[len(paired_samples)]),
        )

    if not torch.isfinite(model_output["mse"]).all():
        raise AssertionError("training adapter produced non-finite MSE")
    if model_output["mse"].shape != (len(paired_samples),):
        raise AssertionError(f"unexpected mse shape: {tuple(model_output['mse'].shape)}")

    actor_config = DiffusionActorConfig(
        strategy="fsdp",
        ppo_micro_batch_size_per_gpu=len(paired_samples),
        rollout_n=args.num_samples_per_prompt,
        diffusion_loss=DiffusionLossConfig(loss_mode="dpo", dpo_beta=args.dpo_beta),
    )
    scores = torch.tensor([sample.score for sample in paired_samples], device=device, dtype=torch.float32).unsqueeze(-1)
    uids = np.array([sample.uid for sample in paired_samples], dtype=object)
    dpo_loss, dpo_metrics = compute_diffusion_loss_dpo(
        policy_mse=model_output["mse"],
        sample_level_scores=scores,
        config=actor_config,
        index=uids,
    )
    if not torch.isfinite(dpo_loss):
        raise AssertionError("DPO loss is not finite")

    print(f"[training] mse_shape={tuple(model_output['mse'].shape)} timesteps={model_output['timesteps'].tolist()}")
    print(f"[dpo] loss={dpo_loss.detach().item():.6f} metrics={dpo_metrics}")


def main() -> None:
    args = parse_args()
    if args.num_samples_per_prompt < 2:
        raise ValueError("--num-samples-per-prompt must be >= 2 for DPO pair validation")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this validation script")

    prompt_cases = load_prompt_cases(args)
    print(f"[data] loaded {len(prompt_cases)} prompts from {Path(args.data).expanduser()}")
    samples = asyncio.run(run_rollouts(args, prompt_cases))
    paired_samples = select_adjacent_dpo_pairs(samples)

    gc.collect()
    torch.cuda.empty_cache()

    if args.skip_training_forward:
        print("[ok] rollout custom_output validation passed; skipped training adapter check")
        return

    run_training_and_dpo_checks(args, paired_samples)
    print("[ok] SD3 DPO agent-loop rollout outputs are compatible with GPU training and DPO loss")


if __name__ == "__main__":
    main()
