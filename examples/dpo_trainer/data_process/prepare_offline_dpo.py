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
from generation_backends import (
    add_generation_arguments,
    print_generation_backend_info,
    validate_generation_config,
)
from generation_backends import (
    generate_split as backend_generate_split,
)
from parquet_checkpoint_utils import load_resume_checkpoint
from PIL import Image
from pipeline_utils import get_pipeline_utils

DEFAULT_SYSTEM_PROMPT = "You are a helpful image generation assistant."
DEFAULT_REWARD_SERVER_COMMAND = (
    "vllm serve {model} --host {host} --port {port} --dtype bfloat16 --enforce-eager --max-num-seqs {max_num_seqs}"
)


def _read_prompts(path: Path, prompt_key: str) -> list[str]:
    if path.suffix == ".txt":
        with path.open(encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    if path.suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif path.suffix == ".json":
        df = pd.read_json(path)
    else:
        df = pd.read_parquet(path)

    def _to_text(prompt: Any) -> str:
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, list):
            user_parts = [
                str(message.get("content", ""))
                for message in prompt
                if isinstance(message, dict) and message.get("role") == "user"
            ]
            if user_parts:
                return "\n".join(user_parts)
        return "" if prompt is None else str(prompt)

    return [_to_text(prompt) for prompt in df[prompt_key].tolist()]


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
) -> tuple[list[float], list[float]]:
    """Score candidate images concurrently; return scores and per-image scoring latency (seconds)."""
    if not images:
        return [], []

    import aiohttp

    async def _score_one(image: Image.Image, session: aiohttp.ClientSession) -> tuple[float, float]:
        t0 = time.perf_counter()
        if reward_fn is None:
            return 0.0, time.perf_counter() - t0
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
        return _extract_reward_score(result), time.perf_counter() - t0

    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = await asyncio.gather(*[_score_one(image, session) for image in images])
    scores, latencies = zip(*results, strict=True)
    return list(scores), list(latencies)


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
    """Prefer reward on the first GPU and image gen on the second; share GPU 0 when only one is visible."""
    n = torch.cuda.device_count()
    if n <= 1:
        args.reward_gpu = 0
        args.image_gpu = 0
    else:
        args.reward_gpu = 0 if args.reward_gpu is None else args.reward_gpu
        args.image_gpu = 1 if args.image_gpu is None else args.image_gpu

    if args.device is None:
        args.device = f"cuda:{args.image_gpu}" if torch.cuda.is_available() else "cpu"


def _format_reward_server_command(args: argparse.Namespace) -> str:
    values = {
        "model": args.reward_model_name,
        "host": args.reward_server_host,
        "port": args.reward_server_port,
        "max_num_seqs": args.num_images_per_prompt,
    }
    try:
        return args.reward_server_command.format(**values)
    except KeyError:
        return args.reward_server_command.format(
            model=values["model"],
            host=values["host"],
            port=values["port"],
        )


@contextmanager
def _maybe_launch_reward_server(args: argparse.Namespace):
    if not args.launch_reward_server:
        yield
        return

    if args.reward_model_name is None:
        raise ValueError("--launch_reward_server requires --reward_model_name.")

    command = _format_reward_server_command(args)
    env = os.environ.copy()
    env.setdefault("VLLM_USE_DEEP_GEMM", "0")
    if torch.cuda.is_available():
        env["CUDA_VISIBLE_DEVICES"] = str(args.reward_gpu)
    print(
        f"Launching reward server (CUDA device index {args.reward_gpu}, "
        f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}): {command}"
    )
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


async def _generate_split(args: argparse.Namespace, split: str, reward_fn) -> Path:
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

    if args.resume:
        start_idx, resume_base_table = load_resume_checkpoint(output_path)
    else:
        start_idx, resume_base_table = 0, None

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

    async def score_images(images: list[Image.Image], prompt: str) -> list[float]:
        return await _score_images(reward_fn, images, prompt, args)

    output_path = await backend_generate_split(
        args,
        split,
        prompts=prompts,
        output_path=output_path,
        image_dir=image_dir,
        start_idx=start_idx,
        resume_base_table=resume_base_table,
        score_images=score_images,
    )

    row_count = pa.parquet.read_metadata(output_path).num_rows
    print(f"Wrote {row_count} offline DPO pairs to {output_path}")
    if start_idx > 0 or resume_base_table is not None:
        kept_rows = resume_base_table.num_rows if resume_base_table is not None else start_idx
        print(f"Added {row_count - kept_rows} new rows (resumed from index {start_idx}).")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate offline DPO triples with a frozen diffusion model.")
    parser.add_argument("--input_file", required=True, help="Prompt file. Supports .txt, .json, .jsonl and parquet.")
    parser.add_argument("--output_file", required=True, help="Parquet file to write.")
    parser.add_argument("--image_dir", default=None, help="Directory to write generated images.")
    parser.add_argument("--prompt_key", default="prompt", help="Prompt column for parquet/json inputs.")
    add_generation_arguments(parser)
    parser.add_argument(
        "--pipeline",
        choices=["auto", "sd3", "qwen_image"],
        default="auto",
        help="Reference image pipeline used to generate tensors for offline DPO. `auto` infers from --model_path.",
    )
    parser.add_argument("--model_path", default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--data_source", default="offline_dpo")
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--negative_prompt", default="low quality, blurry, distorted, text artifacts, watermark")
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
            "Diffusers device (e.g. cuda:1). "
            "Default: cuda:0 on a single visible GPU; cuda:1 on the second GPU when two or more are visible."
        ),
    )
    parser.add_argument(
        "--reward_gpu",
        type=int,
        default=None,
        help=(
            "Physical CUDA device index for the vLLM reward server (via CUDA_VISIBLE_DEVICES). "
            "Default: 0, or forced to 0 when only one GPU is visible."
        ),
    )
    parser.add_argument(
        "--image_gpu",
        type=int,
        default=None,
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
    validate_generation_config(args)
    print_generation_backend_info(args)

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
    reward_fn = _load_reward_fn(args.reward_function_path, args.reward_function_name)
    with _maybe_launch_reward_server(args):
        asyncio.run(_generate_split(args, split, reward_fn))


if __name__ == "__main__":
    main()
