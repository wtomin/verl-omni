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

"""CLI and validation for offline DPO generation backends."""

import argparse

from .common import GENERATION_SERVER_CHOICES, VLLM_OMNI_PIPELINES


def add_generation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--generation_server",
        choices=GENERATION_SERVER_CHOICES,
        default="diffusers",
        help=(
            "Image generation backend. `diffusers` runs the reference pipeline in-process; "
            "`vllm_omni` uses the DPO rollout adapter via concurrent Ray requests "
            f"(supports --pipeline {' or '.join(VLLM_OMNI_PIPELINES)}; requires --launch_generation_server)."
        ),
    )
    parser.add_argument(
        "--launch_generation_server",
        action="store_true",
        help="Launch a local vLLM-Omni Ray generation server (only for --generation_server vllm_omni).",
    )
    parser.add_argument(
        "--tokenizer_path",
        default=None,
        help="Tokenizer path for vllm_omni prompt encoding (default: <model_path>/tokenizer).",
    )
    parser.add_argument(
        "--custom_chat_template",
        default=None,
        help="Optional chat template override for vllm_omni prompt tokenization.",
    )
    parser.add_argument("--generation_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--generation_gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--generation_max_num_batched_tokens", type=int, default=8192)
    parser.add_argument("--generation_max_num_seqs", type=int, default=16)
    parser.add_argument("--generation_max_model_len", type=int, default=1058)


def validate_generation_config(args: argparse.Namespace) -> None:
    if args.generation_server == "vllm_omni":
        if args.pipeline not in VLLM_OMNI_PIPELINES:
            raise ValueError(f"--generation_server vllm_omni supports --pipeline {' or '.join(VLLM_OMNI_PIPELINES)}.")
        if not args.launch_generation_server:
            raise ValueError("--generation_server vllm_omni requires --launch_generation_server.")
    elif args.launch_generation_server:
        raise ValueError("--launch_generation_server is only valid with --generation_server vllm_omni.")


def print_generation_backend_info(args: argparse.Namespace) -> None:
    if args.generation_server == "diffusers":
        print(f"Image generation device (diffusers): {args.device}.")
    else:
        print(f"Image generation server: vllm_omni on GPU {args.image_gpu}.")
