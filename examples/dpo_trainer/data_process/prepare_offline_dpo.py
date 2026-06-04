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

    {prompt, negative_prompt, img_win, img_lose, win_score, lose_score,
     img_win_latents, img_lose_latents, prompt_embeds, ...}

Training expands each row to adjacent ``win, lose`` samples and consumes the
precomputed diffusion latents and text embeddings directly.
"""

import argparse
import asyncio
import importlib.util
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from parquet_checkpoint_utils import (
    ChunkedParquetWriter,
    ParquetWriterShutdownGuard,
    load_resume_checkpoint,
)
from PIL import Image
from pipeline_utils import get_pipeline_utils

try:
    import torch_npu
except ImportError:
    torch_npu = None

DEFAULT_SYSTEM_PROMPT = "You are a helpful image generation assistant."
DEFAULT_REWARD_SERVER_COMMAND = "vllm serve {model} --host {host} --port {port} --dtype bfloat16 --enforce-eager"
# Due to memory size differences between NPU and GPU devices, we adjust
# --gpu-memory-utilization and --max-model-len for NPU backends.
DEFAULT_REWARD_SERVER_COMMAND_NPU = (
    "vllm serve {model} --host {host} --port {port} --dtype bfloat16 "
    "--gpu-memory-utilization 0.85 --max-model-len 40000"
)


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


def _extract_reward_score(result: Any) -> float:
    if isinstance(result, dict):
        return float(result.get("score", 0.0))
    return float(result)


async def _score_images(
    reward_fn,
    images: list[Image.Image],
    prompt: str,
    args: argparse.Namespace,
) -> list[float]:
    """Score candidate images concurrently via repeated ``compute_score`` calls."""
    if not images:
        return []
    if reward_fn is None:
        return [0.0] * len(images)

    import aiohttp

    async def _score_one(image: Image.Image, session: aiohttp.ClientSession) -> float:
        result = reward_fn(
            data_source=args.data_source,
            solution_image=np.asarray(image).astype("float32") / 255.0,
            ground_truth=prompt,
            extra_info={"raw_prompt": prompt, "prompt": prompt, "aiohttp_session": session},
            reward_router_address=args.reward_router_address,
            model_name=args.reward_model_name,
        )
        if asyncio.iscoroutine(result):
            result = await result
        return _extract_reward_score(result)

    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        return list(await asyncio.gather(*[_score_one(image, session) for image in images]))


def _make_generator(seed: int, device: str) -> torch.Generator:
    generator_device = device if device != "cpu" else "cpu"
    return torch.Generator(device=generator_device).manual_seed(seed)


def _make_generators(seeds: list[int], device: str) -> list[torch.Generator]:
    return [_make_generator(seed, device) for seed in seeds]


def _sample_image_seeds(rng: torch.Generator, count: int) -> list[int]:
    return [int(torch.randint(0, 2**31, (1,), generator=rng).item()) for _ in range(count)]


def _uses_classifier_free_guidance(args: argparse.Namespace) -> bool:
    if args.pipeline == "qwen_image":
        return args.true_cfg_scale is not None and args.true_cfg_scale > 1.0
    return args.guidance_scale is not None and args.guidance_scale > 1.0


def _next_prompt_cfg(args: argparse.Namespace, rng: torch.Generator) -> tuple[str, argparse.Namespace]:
    drop = args.random_drop_negative_prompt and float(torch.rand(1, generator=rng).item()) < 0.5
    if not drop:
        return args.negative_prompt, args
    prompt_args = argparse.Namespace(**vars(args))
    prompt_args.negative_prompt = ""
    scale_key = "true_cfg_scale" if args.pipeline == "qwen_image" else "guidance_scale"
    setattr(prompt_args, scale_key, 1.0)
    return "", prompt_args


def _router_url(host: str, port: int, path: str) -> str:
    return f"http://{host}:{port}{path}"


def _wait_for_reward_server(host: str, port: int, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    url = _router_url(host, port, "/v1/models")
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if 200 <= response.status < 500:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(5)
    raise TimeoutError(f"Reward server did not become ready at {url} within {timeout_s}s: {last_error}")


def _apply_gpu_device_defaults(args: argparse.Namespace) -> None:
    """Prefer reward on the first NPU/GPU and image gen on the second; share device 0 when only one is visible."""
    if torch_npu is not None and torch.npu.is_available():
        n = torch.npu.device_count()
    elif torch.cuda.is_available():
        n = torch.cuda.device_count()
    else:
        n = 0

    if n <= 1:
        args.reward_gpu = 0
        args.image_gpu = 0
    else:
        args.reward_gpu = 0 if args.reward_gpu is None else args.reward_gpu
        args.image_gpu = 1 if args.image_gpu is None else args.image_gpu

    if args.device is None:
        if torch_npu is not None and torch.npu.is_available():
            args.device = f"npu:{args.image_gpu}"
        elif torch.cuda.is_available():
            args.device = f"cuda:{args.image_gpu}"
        else:
            args.device = "cpu"




@contextmanager
def _maybe_launch_reward_server(args: argparse.Namespace):
    if not args.launch_reward_server:
        yield
        return

    if args.reward_model_name is None:
        raise ValueError("--launch_reward_server requires --reward_model_name.")

    is_npu = torch_npu is not None and torch.npu.is_available()
    if is_npu and args.reward_server_command == DEFAULT_REWARD_SERVER_COMMAND:
        server_command = DEFAULT_REWARD_SERVER_COMMAND_NPU
    else:
        server_command = args.reward_server_command
    command = server_command.format(
        model=args.reward_model_name,
        host=args.reward_server_host,
        port=args.reward_server_port,
    )
    env = os.environ.copy()
    env.setdefault("VLLM_USE_DEEP_GEMM", "0")
    if is_npu:
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(args.reward_gpu)
        env["CUDA_VISIBLE_DEVICES"] = str(args.reward_gpu)
        dev_msg = f"ASCEND_RT_VISIBLE_DEVICES={env.get('ASCEND_RT_VISIBLE_DEVICES')}"
    elif torch.cuda.is_available():
        env["CUDA_VISIBLE_DEVICES"] = str(args.reward_gpu)
        dev_msg = f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}"
    else:
        dev_msg = "CPU Mode"
    print(f"Launching reward server (Device index {args.reward_gpu}, {dev_msg}): {command}")
    process = subprocess.Popen(shlex.split(command), env=env)
    try:
        _wait_for_reward_server(args.reward_server_host, args.reward_server_port, args.reward_server_startup_timeout)
        print(f"Reward server is ready at {args.reward_router_address}")
        yield
    finally:
        print("Stopping reward server.")
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


async def _generate_split(args: argparse.Namespace, split: str) -> Path:
    pipeline_utils = get_pipeline_utils(args)
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

    resume = args.resume
    resume_base_table: pa.Table | None = None
    if resume:
        start_idx, resume_base_table = load_resume_checkpoint(output_path)
    else:
        start_idx = 0
    if start_idx >= len(prompts):
        kept_rows = resume_base_table.num_rows if resume_base_table is not None else start_idx
        print(
            f"Output parquet {output_path} already contains {kept_rows} completed prompts; "
            f"nothing left to process ({len(prompts)} prompts requested)."
        )
        return output_path
    if start_idx > 0 or resume_base_table is not None:
        kept_rows = resume_base_table.num_rows if resume_base_table is not None else start_idx
        print(f"Resuming from prompt index {start_idx}/{len(prompts)} ({kept_rows} rows kept in {output_path}).")

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    pipe = pipeline_utils.load_pipeline(args, dtype)
    pipe.to(args.device)
    print("Compiling repeated blocks...")
    pipe.transformer.compile_repeated_blocks(fullgraph=True)
    pipe.set_progress_bar_config(disable=args.disable_progress)
    reward_fn = _load_reward_fn(args.reward_function_path, args.reward_function_name)
    seed_rng = torch.Generator().manual_seed(args.seed)
    for _ in range(start_idx):
        _next_prompt_cfg(args, seed_rng)
        _sample_image_seeds(seed_rng, args.num_images_per_prompt)

    with ChunkedParquetWriter(output_path, args.parquet_flush_every, base_table=resume_base_table) as writer:
        with ParquetWriterShutdownGuard(writer):
            for prompt_idx in range(start_idx, len(prompts)):
                prompt = prompts[prompt_idx]
                print(
                    f"[{prompt_idx + 1}/{len(prompts)}] Generating {args.num_images_per_prompt} images "
                    f"for prompt: {prompt!r}"
                )
                negative_prompt, prompt_args = _next_prompt_cfg(args, seed_rng)
                if _uses_classifier_free_guidance(prompt_args):
                    print(f"  negative_prompt: {negative_prompt!r}")
                prompt_tensors = pipeline_utils.encode_prompt_tensors(pipe, prompt, negative_prompt, prompt_args)
                seeds = _sample_image_seeds(seed_rng, args.num_images_per_prompt)
                images = pipe(
                    **pipeline_utils.build_generate_kwargs(prompt_args, prompt, _make_generators(seeds, args.device))
                ).images
                generated: list[dict[str, Any]] = []
                for image, seed in zip(images, seeds, strict=True):
                    generated.append({"image": image, "seed": seed})

                scores = await _score_images(reward_fn, [item["image"] for item in generated], prompt, args)
                candidates = []
                for item, score in zip(generated, scores, strict=True):
                    candidates.append(
                        {
                            "image": item["image"],
                            "score": score,
                            "seed": item["seed"],
                        }
                    )

                candidates.sort(key=lambda item: item["score"], reverse=True)
                win = candidates[0]
                lose = candidates[-1]
                win_path = image_dir / f"{prompt_idx:06d}_win.png"
                lose_path = image_dir / f"{prompt_idx:06d}_lose.png"
                win["image"].save(win_path)
                lose["image"].save(lose_path)
                win["path"] = str(win_path)
                lose["path"] = str(lose_path)
                win_latents = pipeline_utils.tensor_to_bytes(
                    pipeline_utils.encode_image_latent(pipe, win["image"], args)
                )
                lose_latents = pipeline_utils.tensor_to_bytes(
                    pipeline_utils.encode_image_latent(pipe, lose["image"], args)
                )
                score_diff = win["score"] - lose["score"]
                print(
                    f"  reward scores: win={win['score']:.4f}, reject={lose['score']:.4f}, "
                    f"diff(win-reject)={score_diff:.4f}"
                )
                writer.write(
                    {
                        "data_source": args.data_source,
                        "pipeline": args.pipeline,
                        "prompt": _build_messages(prompt, args.system_prompt),
                        "negative_prompt": _build_messages(negative_prompt, args.system_prompt),
                        "img_win": os.path.relpath(win["path"], output_path.parent),
                        "img_lose": os.path.relpath(lose["path"], output_path.parent),
                        "img_win_latents": win_latents,
                        "img_lose_latents": lose_latents,
                        **prompt_tensors,
                        "win_score": win["score"],
                        "lose_score": lose["score"],
                        "reward_model": {"style": "model", "ground_truth": prompt},
                        "extra_info": {
                            "split": split,
                            "index": prompt_idx,
                            "raw_prompt": prompt,
                            "raw_negative_prompt": negative_prompt,
                            "num_candidates": len(candidates),
                            "win_seed": win["seed"],
                            "lose_seed": lose["seed"],
                            "candidate_scores": [item["score"] for item in candidates],
                        },
                    }
                )
                writer.commit_checkpoint()

    print(f"Wrote {writer.row_count} offline DPO pairs to {output_path}")
    if start_idx > 0 or resume_base_table is not None:
        kept_rows = resume_base_table.num_rows if resume_base_table is not None else start_idx
        print(f"Added {writer.row_count - kept_rows} new rows (resumed from index {start_idx}).")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate offline DPO triples with a frozen diffusion model.")
    parser.add_argument("--input_file", required=True, help="Prompt file. Supports .txt, .json, .jsonl and parquet.")
    parser.add_argument("--output_file", required=True, help="Parquet file to write.")
    parser.add_argument("--image_dir", default=None, help="Directory to write generated images.")
    parser.add_argument("--prompt_key", default="prompt", help="Prompt column for parquet/json inputs.")
    parser.add_argument(
        "--pipeline",
        choices=["auto", "sd3", "qwen_image"],
        default="auto",
        help="Reference image pipeline used to generate tensors for offline DPO. `auto` infers from --model_path.",
    )
    parser.add_argument("--model_path", default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--data_source", default="offline_dpo")
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument(
        "--random_drop_negative_prompt",
        action="store_true",
        help="Randomly drop negative_prompt for 50%% of prompts and disable CFG for those samples.",
    )
    parser.add_argument("--num_images_per_prompt", type=int, default=4)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--true_cfg_scale", type=float, default=4.0, help="Qwen-Image True-CFG scale.")
    parser.add_argument("--max_sequence_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42, help="Master seed for sampling independent per-image seeds.")
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Diffusers device (e.g. npu:1, cuda:1). "
            "Default: npu:0 / cuda:0 on a single visible card; npu:1 / cuda:1 "
            "on the second card when two or more are visible."
        ),
    )
    parser.add_argument(
        "--reward-gpu",
        type=int,
        default=None,
        dest="reward_gpu",
        help=(
            "Physical CUDA device index for the vLLM reward server (via CUDA_VISIBLE_DEVICES). "
            "Default: 0, or forced to 0 when only one GPU is visible."
        ),
    )
    parser.add_argument(
        "--image-gpu",
        type=int,
        default=None,
        dest="image_gpu",
        help=(
            "CUDA device index for image generation when --device is not set. "
            "Default: 0 with one visible GPU, else 1 (second GPU)."
        ),
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument(
        "--parquet_flush_every",
        type=int,
        default=32,
        help="Number of DPO rows to buffer in memory before flushing to parquet.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing --output_file parquet when present (default: enabled).",
    )
    parser.add_argument("--reward_function_path", default=None)
    parser.add_argument("--reward_function_name", default=None)
    parser.add_argument("--reward_router_address", default=None)
    parser.add_argument("--reward_model_name", default=None)
    parser.add_argument(
        "--launch_reward_server",
        action="store_true",
        help="Launch an OpenAI-compatible reward server subprocess before scoring.",
    )
    parser.add_argument("--reward_server_host", default="127.0.0.1")
    parser.add_argument("--reward_server_port", type=int, default=8000)
    parser.add_argument(
        "--reward_server_command",
        default=DEFAULT_REWARD_SERVER_COMMAND,
        help=(
            "Command template used with --launch_reward_server. "
            "Available placeholders: {model}, {host}, {port}, {max_num_seqs}."
        ),
    )
    parser.add_argument("--reward_server_startup_timeout", type=int, default=900)
    parser.add_argument("--disable_progress", action="store_true")
    parser.add_argument("--split", default=None, help="Optional split name stored in extra_info.split.")
    args = parser.parse_args()
    get_pipeline_utils(args)
    _apply_gpu_device_defaults(args)
    print(f"Image generation device: {args.device}.")

    if args.num_images_per_prompt < 2:
        raise ValueError("--num_images_per_prompt must be at least 2 for DPO pair construction.")
    if (args.reward_function_path is None) != (args.reward_function_name is None):
        raise ValueError("Set both --reward_function_path and --reward_function_name, or neither.")
    if args.launch_reward_server and args.reward_router_address is None:
        args.reward_router_address = f"{args.reward_server_host}:{args.reward_server_port}"
    if args.reward_function_path is not None and args.reward_router_address is None:
        raise ValueError(
            "Reward scoring requires --reward_router_address, or use --launch_reward_server to start one automatically."
        )

    output_path = Path(os.path.expanduser(args.output_file))
    split = args.split or output_path.stem
    with _maybe_launch_reward_server(args):
        asyncio.run(_generate_split(args, split))


if __name__ == "__main__":
    main()
