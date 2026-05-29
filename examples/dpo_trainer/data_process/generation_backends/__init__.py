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

"""Generation backend dispatch for offline DPO data preparation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Awaitable, Callable

import pyarrow as pa

from . import diffusers as diffusers_backend
from . import vllm_omni as vllm_omni_backend
from .config import add_generation_arguments, print_generation_backend_info, validate_generation_config

__all__ = [
    "add_generation_arguments",
    "generate_split",
    "print_generation_backend_info",
    "validate_generation_config",
]


async def generate_split(
    args: argparse.Namespace,
    split: str,
    *,
    prompts: list[str],
    output_path: Path,
    image_dir: Path,
    start_idx: int,
    resume_base_table: pa.Table | None,
    score_images: Callable[[list[Any], str], Awaitable[list[float]]],
) -> Path:
    """Run offline DPO generation for one split using the selected backend."""
    common_kwargs = {
        "prompts": prompts,
        "output_path": output_path,
        "image_dir": image_dir,
        "start_idx": start_idx,
        "resume_base_table": resume_base_table,
        "score_images": score_images,
    }

    if args.generation_server == "diffusers":
        return await diffusers_backend.generate_split(args, split, **common_kwargs)
    if args.generation_server == "vllm_omni":
        with vllm_omni_backend.launch_generation_server(args) as generation_server:
            return await vllm_omni_backend.generate_split(
                args, split, generation_server=generation_server, **common_kwargs
            )
    raise ValueError(f"Unknown generation server: {args.generation_server!r}")
