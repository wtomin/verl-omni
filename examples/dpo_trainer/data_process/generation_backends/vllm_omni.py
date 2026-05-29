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

"""vLLM-Omni Ray generation backend for offline DPO data preparation."""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import pyarrow as pa
import ray
import torch
from omegaconf import OmegaConf
from PIL import Image
from pipeline_utils import get_pipeline_utils
from transformers import AutoTokenizer
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import RolloutMode

from verl_omni.utils.fs import resolve_model_local_dir
from verl_omni.workers.rollout.replica import DiffusionOutput
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer

from .common import (
    ScoreImages,
    build_messages,
    pack_generated_samples,
    pack_rollout_prompt_tensors,
    run_split_loop,
    score_and_write_dpo_row,
)

PIPELINE_VLLM_CONFIG = {
    "qwen_image": {
        "external_lib": "verl_omni.pipelines.qwen_image_dpo",
        "import_module": "verl_omni.pipelines.qwen_image_dpo",
        "use_prompt_ids": True,
        "rollout_extra_keys": (
            "prompt_embeds",
            "prompt_embeds_mask",
            "negative_prompt_embeds",
            "negative_prompt_embeds_mask",
        ),
    },
    "sd3": {
        "external_lib": "verl_omni.pipelines.sd3_dpo",
        "import_module": "verl_omni.pipelines.sd3_dpo",
        "use_prompt_ids": False,
        "rollout_extra_keys": (
            "prompt_embeds",
            "prompt_embeds_mask",
            "pooled_prompt_embeds",
            "negative_prompt_embeds",
            "negative_prompt_embeds_mask",
            "negative_pooled_prompt_embeds",
        ),
    },
}


def _resolve_tokenizer_path(args: argparse.Namespace) -> str:
    """Resolve tokenizer to a local path or valid HF repo id (not ``org/model/tokenizer``)."""
    if args.tokenizer_path:
        return os.path.expanduser(args.tokenizer_path)
    local_model_dir = resolve_model_local_dir(os.path.expanduser(args.model_path))
    tokenizer_subdir = os.path.join(local_model_dir, "tokenizer")
    return tokenizer_subdir if os.path.isdir(tokenizer_subdir) else local_model_dir


@dataclass(frozen=True)
class VllmOmniBackend:
    args: argparse.Namespace
    spec: dict[str, Any]
    server: Any
    pipeline_utils: Any
    tokenizer: AutoTokenizer | None

    @classmethod
    def create(cls, args: argparse.Namespace, server: Any) -> VllmOmniBackend:
        if args.pipeline not in PIPELINE_VLLM_CONFIG:
            raise ValueError(
                f"--generation_server vllm_omni does not support pipeline={args.pipeline!r}. "
                f"Supported: {sorted(PIPELINE_VLLM_CONFIG)}."
            )
        spec = PIPELINE_VLLM_CONFIG[args.pipeline]
        __import__(spec["import_module"])
        tokenizer = None
        if spec["use_prompt_ids"]:
            tokenizer = AutoTokenizer.from_pretrained(_resolve_tokenizer_path(args), trust_remote_code=True)
            if args.custom_chat_template:
                tokenizer.chat_template = args.custom_chat_template
        return cls(
            args=args,
            spec=spec,
            server=server,
            pipeline_utils=get_pipeline_utils(args),
            tokenizer=tokenizer,
        )

    def _tokenize(self, text: str) -> list[int]:
        assert self.tokenizer is not None
        messages = build_messages(text, self.args.system_prompt)
        token_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        return normalize_token_ids(token_ids)

    def _pad_token_ids(self, token_ids: list[int] | None) -> list[int] | None:
        if token_ids is None:
            return None
        assert self.tokenizer is not None
        padded = self.tokenizer.pad(
            {"input_ids": token_ids},
            padding="max_length",
            max_length=self.args.generation_prompt_length,
            return_tensors="pt",
            return_attention_mask=False,
        )["input_ids"]
        if padded.dim() == 2:
            padded = padded.squeeze(0)
        return normalize_token_ids(padded.tolist())

    def prompt_ids_for_request(self, prompt: str) -> tuple[list[int] | None, list[int] | None]:
        if not self.spec["use_prompt_ids"]:
            return None, None
        negative_ids = self._tokenize(self.args.negative_prompt) if self.args.true_cfg_scale > 1.0 else None
        prompt_ids = self._tokenize(prompt)
        return self._pad_token_ids(prompt_ids), self._pad_token_ids(negative_ids)

    def _sampling_params(self, seed: int) -> dict[str, Any]:
        params = {
            "height": self.args.height,
            "width": self.args.width,
            "num_inference_steps": self.args.num_inference_steps,
            "guidance_scale": self.args.guidance_scale,
            "max_sequence_length": self.args.max_sequence_length,
            "seed": seed,
            "logprobs": False,
        }
        if self.args.pipeline == "qwen_image":
            params["true_cfg_scale"] = self.args.true_cfg_scale
        return params

    async def rollout_candidates(
        self, *, prompt: str, prompt_ids: list[int] | None, negative_prompt_ids: list[int] | None, seeds: list[int]
    ) -> tuple[list[dict[str, Any]], list[float]]:
        loop = asyncio.get_event_loop()

        async def _generate_one(seed: int) -> tuple[DiffusionOutput, float]:
            kwargs: dict[str, Any] = {
                "sampling_params": self._sampling_params(seed),
                "request_id": f"offline_dpo_{uuid4().hex}",
            }
            if self.spec["use_prompt_ids"]:
                kwargs["prompt_ids"] = prompt_ids
                kwargs["negative_prompt_ids"] = negative_prompt_ids
            else:
                kwargs["prompt"] = prompt
                kwargs["negative_prompt"] = self.args.negative_prompt
            ref = self.server.generate.remote(**kwargs)
            t0 = time.perf_counter()
            output = await loop.run_in_executor(None, lambda r=ref: ray.get(r))
            return output, time.perf_counter() - t0

        concurrency = max(1, self.args.generation_concurrency)
        results: list[tuple[DiffusionOutput, float]] = []
        for offset in range(0, len(seeds), concurrency):
            seed_batch = seeds[offset : offset + concurrency]
            results.extend(await asyncio.gather(*[_generate_one(seed) for seed in seed_batch]))
        candidates = []
        generation_latency_s = []
        for seed, (output, latency_s) in zip(seeds, results, strict=True):
            generation_latency_s.append(latency_s)
            extra = output.extra_fields or {}
            latents = extra.get("latents")
            if latents is None:
                raise RuntimeError(
                    f"vLLM-Omni DPO rollout did not return `latents` for pipeline={self.args.pipeline!r}. "
                    f"Ensure {self.spec['external_lib']} is loaded."
                )
            item = {"seed": seed, "image": _diffusion_output_to_pil(output.diffusion_output), "latents": latents}
            for key in self.spec["rollout_extra_keys"]:
                item[key] = extra.get(key)
            candidates.append(item)
        return candidates, generation_latency_s


def _diffusion_pipeline_cfg(args: argparse.Namespace) -> dict[str, Any]:
    cfg = {
        "_target_": "verl_omni.workers.config.diffusion.rollout.DiffusionPipelineConfig",
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "max_sequence_length": args.max_sequence_length,
    }
    if args.pipeline == "qwen_image":
        cfg["true_cfg_scale"] = args.true_cfg_scale
    return cfg


def _build_rollout_config(args: argparse.Namespace, spec: dict[str, Any]) -> Any:
    return OmegaConf.create(
        {
            "_target_": "verl_omni.workers.config.diffusion.DiffusionRolloutConfig",
            "name": "vllm_omni",
            "mode": "async",
            "tensor_model_parallel_size": args.generation_tensor_parallel_size,
            "data_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "gpu_memory_utilization": args.generation_gpu_memory_utilization,
            "max_num_batched_tokens": args.generation_max_num_batched_tokens,
            "max_num_seqs": args.generation_concurrency,
            "max_model_len": args.generation_max_model_len,
            "prompt_length": args.generation_prompt_length,
            "dtype": args.dtype,
            "load_format": "auto",
            "enforce_eager": True,
            "enable_chunked_prefill": False,
            "enable_prefix_caching": False,
            "enable_sleep_mode": False,
            "free_cache_engine": False,
            "disable_log_stats": True,
            "n": 1,
            "external_lib": spec["external_lib"],
            "pipeline": _diffusion_pipeline_cfg(args),
        }
    )


def _build_model_config(args: argparse.Namespace, spec: dict[str, Any]) -> Any:
    model_cfg: dict[str, Any] = {
        "_target_": "verl_omni.workers.config.diffusion.DiffusionModelConfig",
        "path": os.path.expanduser(args.model_path),
        "trust_remote_code": True,
        "load_tokenizer": args.pipeline == "qwen_image",
        "algorithm": "dpo",
        "external_lib": spec["external_lib"],
        "pipeline": _diffusion_pipeline_cfg(args),
    }
    if args.pipeline == "qwen_image":
        # Let DiffusionModelConfig resolve tokenizer_path from ``path`` when unset
        if args.tokenizer_path:
            model_cfg["tokenizer_path"] = os.path.expanduser(args.tokenizer_path)
        if args.custom_chat_template:
            model_cfg["custom_chat_template"] = args.custom_chat_template
    return OmegaConf.create(model_cfg)


def _diffusion_output_to_pil(diffusion_output: Any) -> Image.Image:
    if not isinstance(diffusion_output, torch.Tensor):
        diffusion_output = torch.tensor(diffusion_output)
    if diffusion_output.ndim != 3:
        raise ValueError(f"Expected CHW diffusion output, got shape {tuple(diffusion_output.shape)}")
    tensor = diffusion_output
    if tensor.max() > 1.0:
        tensor = tensor.float() / 255.0
    array = (tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array)


@contextmanager
def launch_generation_server(args: argparse.Namespace):
    if not args.launch_generation_server:
        raise ValueError("vllm_omni generation requires --launch_generation_server.")

    spec = PIPELINE_VLLM_CONFIG[args.pipeline]
    __import__(spec["import_module"])
    ray.init(ignore_reinit_error=True)
    server = None
    try:
        server_cls = ray.remote(vLLMOmniHttpServer)
        server = server_cls.options(
            runtime_env={
                "env_vars": {
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                    "NCCL_CUMEM_ENABLE": "0",
                    "TOKENIZERS_PARALLELISM": "true",
                }
            },
            max_concurrency=args.generation_concurrency,
        ).remote(
            config=_build_rollout_config(args, spec),
            model_config=_build_model_config(args, spec),
            rollout_mode=RolloutMode.STANDALONE,
            workers=[],
            replica_rank=0,
            node_rank=0,
            gpus_per_node=1,
            nnodes=1,
            cuda_visible_devices=str(args.image_gpu),
        )
        ray.get(server.launch_server.remote())
        print(
            f"vLLM-Omni generation server ready on GPU {args.image_gpu} "
            f"(model={args.model_path}, pipeline={args.pipeline}, algorithm=dpo)."
        )
        yield server
    finally:
        if server is not None:
            del server
        if ray.is_initialized():
            ray.shutdown()


async def generate_split(
    args: argparse.Namespace,
    split: str,
    *,
    prompts: list[str],
    output_path: Path,
    image_dir: Path,
    start_idx: int,
    resume_base_table: pa.Table | None,
    generation_server: Any,
    score_images: ScoreImages,
) -> Path:
    backend = VllmOmniBackend.create(args, generation_server)

    async def process_prompt(writer, prompt_idx: int, prompt: str, seeds: list[int]) -> None:
        prompt_ids, negative_prompt_ids = backend.prompt_ids_for_request(prompt)
        raw, generation_latency_s = await backend.rollout_candidates(
            prompt=prompt,
            prompt_ids=prompt_ids,
            negative_prompt_ids=negative_prompt_ids,
            seeds=seeds,
        )
        pack_t0 = time.perf_counter()
        generated = pack_generated_samples(
            raw, prompt_idx=prompt_idx, image_dir=image_dir, pipeline_utils=backend.pipeline_utils
        )
        pack_per_image_s = (time.perf_counter() - pack_t0) / len(generated)
        generation_latency_s = [lat + pack_per_image_s for lat in generation_latency_s]
        await score_and_write_dpo_row(
            writer,
            args=args,
            split=split,
            prompt_idx=prompt_idx,
            prompt=prompt,
            output_path=output_path,
            prompt_tensors=pack_rollout_prompt_tensors(generated[0], backend.pipeline_utils, args.pipeline),
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
