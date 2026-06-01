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

"""Validate offline DPO checkpoints by plotting training step vs reward score."""

import argparse
import ast
import asyncio
import csv
import importlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from data_process.pipeline_utils import get_pipeline_utils
from PIL import Image

DEFAULT_REWARD_SERVER_COMMAND = "vllm serve {model} --host {host} --port {port} --dtype bfloat16 --enforce-eager"
CHECKPOINT_FILE_NAMES = (
    "model_world_size_1_rank_0.pt",
    "adapter_model.safetensors",
    "adapter_model.bin",
    "diffusion_pytorch_model.safetensors",
    "diffusion_pytorch_model.bin",
    "model.safetensors",
    "pytorch_model.bin",
)
LORA_WEIGHT_FILE_NAMES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "pytorch_lora_weights.safetensors",
    "pytorch_lora_weights.bin",
)

DEFAULT_QWEN_IMAGE_LORA_TARGET_MODULES = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "img_mlp.net.0.proj",
    "img_mlp.net.2",
    "txt_mlp.net.0.proj",
    "txt_mlp.net.2",
]


def _read_prompts(path: Path, max_prompts: int, max_prompt_lines: int) -> list[str]:
    with path.open(encoding="utf-8") as f:
        lines = [line for idx, line in enumerate(f) if max_prompt_lines <= 0 or idx < max_prompt_lines]
    prompts = [line.strip() for line in lines if line.strip()]
    return prompts[:max_prompts] if max_prompts > 0 else prompts


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


def _apply_gpu_device_defaults(args: argparse.Namespace) -> None:
    """Prefer reward on GPU 0 and image generation on GPU 1 when multiple GPUs are visible."""
    n = torch.cuda.device_count()
    if n <= 1:
        args.reward_gpu = 0 if args.reward_gpu is None else args.reward_gpu
        args.image_gpu = 0 if args.image_gpu is None else args.image_gpu
    else:
        args.reward_gpu = 0 if args.reward_gpu is None else args.reward_gpu
        args.image_gpu = 1 if args.image_gpu is None else args.image_gpu

    if args.device is None:
        args.device = f"cuda:{args.image_gpu}" if torch.cuda.is_available() else "cpu"


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


@contextmanager
def _maybe_launch_reward_server(args: argparse.Namespace):
    if not args.launch_reward_server:
        yield
        return

    if args.reward_model_name is None:
        raise ValueError("--launch_reward_server requires --reward_model_name.")

    command = args.reward_server_command.format(
        model=args.reward_model_name,
        host=args.reward_server_host,
        port=args.reward_server_port,
    )
    env = os.environ.copy()
    env.setdefault("VLLM_USE_DEEP_GEMM", "0")
    if torch.cuda.is_available() and args.reward_gpu is not None:
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


def _parse_step(path: Path) -> int:
    for part in [path.name, *[p.name for p in path.parents]]:
        match = re.search(r"(?:global_)?step[_-]?(\d+)", part)
        if match:
            return int(match.group(1))
    return -1


def _is_checkpoint_file(path: Path) -> bool:
    if not path.is_file() or path.suffix not in {".pt", ".bin", ".safetensors"}:
        return False
    if path.name in CHECKPOINT_FILE_NAMES or path.name in LORA_WEIGHT_FILE_NAMES:
        return True
    return re.fullmatch(r"model_world_size_\d+_rank_\d+\.pt", path.name) is not None


def _is_checkpoint_like(path: Path) -> bool:
    if path.is_file():
        return _is_checkpoint_file(path)
    if not path.is_dir():
        return False
    if (path / "adapter_config.json").exists() and _has_lora_weight_file(path):
        return True
    if (path / "huggingface").is_dir():
        return True
    if any(path.glob("model_world_size_*_rank_*.pt")):
        return True
    return any((path / name).exists() for name in CHECKPOINT_FILE_NAMES)


def _discover_checkpoints(root: Path) -> list[Path]:
    if _is_checkpoint_like(root):
        return [root]

    candidates: list[Path] = []
    for step_dir in sorted(root.glob("global_step_*"), key=_parse_step):
        actor_dir = step_dir / "actor"
        if _is_checkpoint_like(actor_dir):
            candidates.append(actor_dir)
            continue
        candidates.extend(path for path in actor_dir.rglob("*") if _is_checkpoint_like(path))

    if not candidates:
        candidates = [path for path in root.rglob("*") if _is_checkpoint_like(path)]

    deduped = []
    seen = set()
    for path in sorted(candidates, key=lambda p: (_parse_step(p), str(p))):
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(path)
    return deduped


def _load_state_dict_file(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        for key in ("state_dict", "model", "module"):
            if key in state and isinstance(state[key], dict):
                return state[key]
        return state
    raise TypeError(f"Unsupported checkpoint object type from {path}: {type(state)}")


def _shard_rank(path: Path) -> int:
    match = re.search(r"rank_(\d+)", path.name)
    if match is None:
        raise ValueError(f"Unexpected FSDP shard filename: {path.name}")
    return int(match.group(1))


def _list_actor_model_shards(checkpoint: Path) -> list[Path]:
    return sorted(checkpoint.glob("model_world_size_*_rank_*.pt"), key=_shard_rank)


def _infer_fsdp_world_size(checkpoint: Path, shards: list[Path] | None = None) -> int:
    if shards is None:
        shards = _list_actor_model_shards(checkpoint)

    config_path = checkpoint / "fsdp_config.json"
    if config_path.exists():
        with config_path.open(encoding="utf-8") as f:
            world_size = json.load(f).get("world_size")
            if world_size is not None:
                return int(world_size)

    if not shards:
        raise FileNotFoundError(f"No FSDP model shards found under {checkpoint}")
    match = re.fullmatch(r"model_world_size_(\d+)_rank_(\d+)\.pt", shards[0].name)
    if match is None:
        raise ValueError(f"Unexpected FSDP shard filename: {shards[0].name}")
    return int(match.group(1))


def _is_multi_shard_fsdp_checkpoint(checkpoint: Path) -> bool:
    if not checkpoint.is_dir():
        return False
    shards = _list_actor_model_shards(checkpoint)
    if not shards:
        return False
    try:
        return _infer_fsdp_world_size(checkpoint, shards) > 1
    except (FileNotFoundError, ValueError):
        return len(shards) > 1


def _merge_by_fsdp_placement(tensors: list[torch.Tensor], placement: Any) -> torch.Tensor:
    if placement.is_replicate():
        return tensors[0]
    if placement.is_partial():
        raise NotImplementedError("Partial FSDP placement is not supported during checkpoint merge.")
    if placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    raise NotImplementedError(f"Unsupported FSDP placement during checkpoint merge: {placement}")


def _merge_fsdp_shards_with_verl_library(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    """Merge FSDP shards using ``verl.model_merger.FSDPModelMerger`` merge logic."""
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    checkpoint_dir = checkpoint_dir.resolve()
    world_size = _infer_fsdp_world_size(checkpoint_dir)
    rank_zero_state_dict = _load_state_dict_file(checkpoint_dir / f"model_world_size_{world_size}_rank_0.pt")

    merger = object.__new__(FSDPModelMerger)
    mesh, mesh_dim_names = merger._extract_device_mesh_info(rank_zero_state_dict, world_size)
    total_shards, mesh_shape = merger._calculate_shard_configuration(mesh, mesh_dim_names)
    print(
        f"Merging {total_shards} FSDP shards with verl.model_merger library from {checkpoint_dir} "
        f"(world_size={world_size})"
    )
    return merger._load_and_merge_state_dicts(world_size, total_shards, mesh_shape, mesh_dim_names)


def _merge_fsdp_shards_with_verl_cli(
    checkpoint_dir: Path,
    target_dir: Path,
    *,
    use_cpu_initialization: bool,
) -> Path:
    """Merge sharded FSDP checkpoints via ``python -m verl.model_merger merge``.

    See https://verl.readthedocs.io/en/latest/advance/checkpoint.html
    """
    command = [
        sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(checkpoint_dir.resolve()),
        "--target_dir",
        str(target_dir.resolve()),
        "--trust-remote-code",
    ]
    if use_cpu_initialization:
        command.append("--use_cpu_initialization")
    print(f"Merging FSDP shards with verl.model_merger CLI: {' '.join(command)}")
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(
            "verl.model_merger merge failed. "
            "See https://verl.readthedocs.io/en/latest/advance/checkpoint.html "
            f"for manual merge instructions. CLI stderr:\n{stderr}"
        )
    lora_adapter_dir = target_dir / "lora_adapter"
    if lora_adapter_dir.is_dir() and (lora_adapter_dir / "adapter_config.json").exists():
        return lora_adapter_dir
    return target_dir


def _merge_fsdp_shards_for_validation(checkpoint_dir: Path, args: argparse.Namespace) -> dict[str, torch.Tensor]:
    backend = args.fsdp_merge_backend
    if backend in ("auto", "verl"):
        try:
            return _merge_fsdp_shards_with_verl_library(checkpoint_dir)
        except ImportError as exc:
            if backend == "verl":
                raise ImportError(
                    "fsdp_merge_backend=verl requires verl with model_merger installed. "
                    "Install verl or use --fsdp_merge_backend local."
                ) from exc
            print(f"verl.model_merger is unavailable ({exc}); falling back to local FSDP merge.")
        except Exception as exc:
            if backend == "verl":
                raise RuntimeError(f"verl.model_merger FSDP merge failed: {exc}") from exc
            print(f"verl.model_merger FSDP merge failed ({exc}); falling back to local FSDP merge.")

    return _merge_fsdp_sharded_state_dict(checkpoint_dir)


def _load_checkpoint_from_verl_cli_merge(pipe, checkpoint: Path, args: argparse.Namespace, target_dir: Path) -> str:
    merged_path = _merge_fsdp_shards_with_verl_cli(
        checkpoint,
        target_dir,
        use_cpu_initialization=args.fsdp_merge_use_cpu,
    )
    if merged_path.name == "lora_adapter":
        if hasattr(pipe, "load_lora_weights"):
            pipe.load_lora_weights(str(merged_path))
            return "merged_fsdp_lora_adapter"
        if hasattr(pipe.transformer, "load_lora_adapter"):
            pipe.transformer.load_lora_adapter(str(merged_path))
            return "merged_fsdp_transformer_lora_adapter"
    for name in CHECKPOINT_FILE_NAMES:
        candidate = merged_path / name
        if candidate.exists():
            return _load_transformer_state_dict(pipe, _load_state_dict_file(candidate), args, checkpoint)
    if (merged_path / "transformer").is_dir():
        return _load_huggingface_checkpoint(pipe, merged_path, args)
    raise FileNotFoundError(
        f"verl.model_merger wrote no loadable weights under {merged_path}. "
        "For diffusion LoRA checkpoints, prefer --fsdp_merge_backend auto or local."
    )


def _merge_fsdp_sharded_state_dict(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    """Merge FSDP actor shards into one state dict for single-GPU validation."""
    checkpoint_dir = checkpoint_dir.resolve()
    shards = _list_actor_model_shards(checkpoint_dir)
    world_size = _infer_fsdp_world_size(checkpoint_dir, shards)
    if world_size <= 1:
        if not shards:
            raise FileNotFoundError(f"No FSDP model shards found under {checkpoint_dir}")
        return _load_state_dict_file(shards[0])
    if len(shards) < world_size:
        raise FileNotFoundError(f"Expected {world_size} FSDP shards under {checkpoint_dir}, found {len(shards)}.")

    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        from torch.distributed._tensor import DTensor

    model_state_dict_lst = [
        _load_state_dict_file(checkpoint_dir / f"model_world_size_{world_size}_rank_{rank}.pt")
        for rank in range(world_size)
    ]

    pivot_key = sorted(model_state_dict_lst[0])[0]
    pivot_tensor = model_state_dict_lst[0][pivot_key]
    if isinstance(pivot_tensor, DTensor):
        mesh_dim_names = pivot_tensor.device_mesh.mesh_dim_names
    else:
        mesh_dim_names = ("fsdp",)

    merged: dict[str, Any] = {}
    param_placements: dict[str, tuple[Any, ...]] = {}

    for key in model_state_dict_lst[0]:
        merged[key] = []
        for shard_state in model_state_dict_lst:
            tensor = shard_state[key]
            if isinstance(tensor, DTensor):
                merged[key].append(tensor._local_tensor.bfloat16())
                placements = tuple(tensor.placements)
                if mesh_dim_names[0] in ("dp", "ddp"):
                    placements = placements[1:]
                if key not in param_placements:
                    param_placements[key] = placements
                elif param_placements[key] != placements:
                    raise RuntimeError(f"Inconsistent FSDP placements for parameter {key!r}.")
            else:
                merged[key].append(tensor.bfloat16() if tensor.is_floating_point() else tensor)

    for key in sorted(merged):
        shards_for_key = merged[key]
        if not isinstance(shards_for_key, list):
            continue
        if key in param_placements:
            placements = param_placements[key]
            if len(placements) != 1:
                raise NotImplementedError("FSDP + tensor parallelism checkpoint merge is not supported.")
            merged[key] = _merge_by_fsdp_placement(shards_for_key, placements[0])
        else:
            merged[key] = torch.cat(shards_for_key, dim=0)

    print(f"Merged {world_size} FSDP shards from {checkpoint_dir} into a full state dict.")
    return merged


def _load_merged_fsdp_checkpoint(pipe, checkpoint: Path, args: argparse.Namespace) -> str:
    if args.fsdp_merge_backend == "verl_cli":
        with tempfile.TemporaryDirectory(prefix="verl_fsdp_merge_") as tmp_dir:
            return _load_checkpoint_from_verl_cli_merge(pipe, checkpoint, args, Path(tmp_dir))

    state_dict = _merge_fsdp_shards_for_validation(checkpoint, args)
    load_kind = _load_transformer_state_dict(pipe, state_dict, args, checkpoint)
    return f"merged_fsdp_{load_kind}"


def _find_actor_model_shard(checkpoint: Path) -> Path | None:
    shards = _list_actor_model_shards(checkpoint)
    return shards[0] if shards else None


def _has_lora_weight_file(path: Path) -> bool:
    return any((path / name).exists() for name in LORA_WEIGHT_FILE_NAMES)


def _find_lora_adapter_config_dir(source: Path) -> Path | None:
    base = source if source.is_dir() else source.parent
    candidates = (
        base,
        base / "huggingface",
        base.parent / "huggingface",
    )
    for candidate in candidates:
        if (candidate / "adapter_config.json").exists():
            return candidate
    return None


def _strip_state_dict_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = (
        "module.",
        "model.",
        "transformer.",
        "_fsdp_wrapped_module.",
        "base_model.model.",
    )
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        cleaned[new_key] = value
    return cleaned


def _state_dict_has_lora(state_dict: dict[str, torch.Tensor]) -> bool:
    return any("lora_" in key for key in state_dict)


def _parse_target_modules(value: str | None) -> list[str]:
    if value is None or value.strip() == "":
        return DEFAULT_QWEN_IMAGE_LORA_TARGET_MODULES
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(parsed, str):
        return [parsed]
    if isinstance(parsed, list | tuple):
        return [str(item) for item in parsed]
    raise ValueError(f"Unsupported --lora_target_modules value: {value!r}")


def _ensure_lora_adapter(pipe, args: argparse.Namespace, adapter_config_dir: Path | None = None) -> None:
    if hasattr(pipe.transformer, "peft_config") and pipe.transformer.peft_config:
        return

    if adapter_config_dir is not None:
        from peft import PeftConfig

        pipe.transformer.add_adapter(PeftConfig.from_pretrained(str(adapter_config_dir)))
        return

    if args.lora_rank <= 0:
        raise ValueError(
            "Checkpoint contains LoRA weights but no adapter_config.json was found next to the checkpoint. "
            "Pass --lora_rank, --lora_alpha, and optionally --lora_target_modules "
            "matching the training run."
        )

    from peft import LoraConfig

    pipe.transformer.add_adapter(
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=_parse_target_modules(args.lora_target_modules),
            bias="none",
        )
    )


def _load_transformer_state_dict(
    pipe, state_dict: dict[str, torch.Tensor], args: argparse.Namespace, source: Path
) -> str:
    if _state_dict_has_lora(state_dict):
        _ensure_lora_adapter(pipe, args, adapter_config_dir=_find_lora_adapter_config_dir(source))
        load_kind = "transformer_lora_state_dict"
    else:
        load_kind = "transformer_state_dict"

    cleaned = _strip_state_dict_prefixes(state_dict)
    missing, unexpected = pipe.transformer.load_state_dict(cleaned, strict=False)
    print(f"Loaded {load_kind} from {source}; missing={len(missing)}, unexpected={len(unexpected)}")
    return load_kind


def _load_huggingface_checkpoint(pipe, huggingface_dir: Path, args: argparse.Namespace) -> str:
    if (huggingface_dir / "adapter_config.json").exists():
        if not _has_lora_weight_file(huggingface_dir):
            raise FileNotFoundError(
                f"{huggingface_dir} contains adapter_config.json but no LoRA weight file. "
                "Falling back to the actor FSDP shard if one is available."
            )
        if hasattr(pipe, "load_lora_weights"):
            pipe.load_lora_weights(str(huggingface_dir))
            return "huggingface_lora_adapter"
        if hasattr(pipe.transformer, "load_lora_adapter"):
            pipe.transformer.load_lora_adapter(str(huggingface_dir))
            return "huggingface_transformer_lora_adapter"
        raise RuntimeError("HuggingFace checkpoint looks like LoRA, but this pipeline cannot load LoRA weights.")

    if any((huggingface_dir / name).exists() for name in CHECKPOINT_FILE_NAMES):
        checkpoint_file = next(
            huggingface_dir / name for name in CHECKPOINT_FILE_NAMES if (huggingface_dir / name).exists()
        )
        load_kind = _load_transformer_state_dict(pipe, _load_state_dict_file(checkpoint_file), args, checkpoint_file)
        return f"huggingface_{load_kind}"

    transformer_dir = huggingface_dir / "transformer"
    if transformer_dir.is_dir():
        if hasattr(pipe.transformer, "load_lora_adapter") and (transformer_dir / "adapter_config.json").exists():
            pipe.transformer.load_lora_adapter(str(transformer_dir))
            return "huggingface_transformer_lora_adapter"
        loaded = pipe.transformer.__class__.from_pretrained(str(transformer_dir))
        pipe.transformer.load_state_dict(loaded.state_dict(), strict=True)
        return "huggingface_transformer_pretrained"

    raise FileNotFoundError(f"No supported HuggingFace checkpoint files found under {huggingface_dir}")


def _load_checkpoint_into_pipeline(pipe, checkpoint: Path, args: argparse.Namespace) -> str:
    if checkpoint.is_dir() and (checkpoint / "adapter_config.json").exists() and _has_lora_weight_file(checkpoint):
        if hasattr(pipe, "load_lora_weights"):
            pipe.load_lora_weights(str(checkpoint))
            return "lora_adapter"
        if hasattr(pipe.transformer, "load_lora_adapter"):
            pipe.transformer.load_lora_adapter(str(checkpoint))
            return "transformer_lora_adapter"
        raise RuntimeError("Checkpoint looks like a LoRA adapter, but this pipeline cannot load LoRA weights.")

    if checkpoint.is_dir() and (checkpoint / "huggingface").is_dir():
        try:
            return _load_huggingface_checkpoint(pipe, checkpoint / "huggingface", args)
        except FileNotFoundError as exc:
            print(f"Skipping incomplete HuggingFace checkpoint export: {exc}")

    checkpoint_file = checkpoint
    if checkpoint.is_dir():
        if _is_multi_shard_fsdp_checkpoint(checkpoint):
            return _load_merged_fsdp_checkpoint(pipe, checkpoint, args)

        for name in CHECKPOINT_FILE_NAMES:
            candidate = checkpoint / name
            if candidate.exists():
                checkpoint_file = candidate
                break
        else:
            shard = _find_actor_model_shard(checkpoint)
            if shard is None:
                raise FileNotFoundError(f"No supported checkpoint file found under {checkpoint}")
            checkpoint_file = shard

    return _load_transformer_state_dict(pipe, _load_state_dict_file(checkpoint_file), args, checkpoint_file)


def _load_pipeline(args: argparse.Namespace, checkpoint: Path | None):
    pipeline_utils = get_pipeline_utils(args)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    pipe = pipeline_utils.load_pipeline(args, dtype)
    pipe.to(args.device)
    pipe.set_progress_bar_config(disable=args.disable_progress)
    load_kind = "base"
    if checkpoint is not None:
        load_kind = _load_checkpoint_into_pipeline(pipe, checkpoint, args)
    return pipe, pipeline_utils, load_kind


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_curve(summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    if not summary_rows:
        return
    import matplotlib.pyplot as plt

    rows = sorted(summary_rows, key=lambda row: row["step"])
    steps = [row["step"] for row in rows]
    scores = [row["mean_reward"] for row in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(steps, scores, marker="o")
    plt.xlabel("Training step")
    plt.ylabel("Reward score")
    plt.title("Validation Reward vs Training Step")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


async def _validate_checkpoint(
    args: argparse.Namespace,
    checkpoint: Path | None,
    prompts: list[str],
    reward_fn,
    image_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    step = 0 if checkpoint is None else _parse_step(checkpoint)
    pipe, pipeline_utils, load_kind = _load_pipeline(args, checkpoint)

    per_prompt_rows = []
    scores = []
    ckpt_image_dir = image_dir / f"step_{step}"
    ckpt_image_dir.mkdir(parents=True, exist_ok=True)

    try:
        for prompt_idx, prompt in enumerate(prompts):
            generator = _make_generator(args.seed + max(step, 0) * 100000 + prompt_idx, args.device)
            kwargs = pipeline_utils.build_generate_kwargs(args, prompt, generator)
            image = pipe(**kwargs).images[0]
            image_path = ckpt_image_dir / f"{prompt_idx:06d}.png"
            image.save(image_path)
            score = await _score_image(reward_fn, image, prompt, args)
            scores.append(score)
            per_prompt_rows.append(
                {
                    "step": step,
                    "checkpoint": "" if checkpoint is None else str(checkpoint),
                    "load_kind": load_kind,
                    "prompt_index": prompt_idx,
                    "prompt": prompt,
                    "reward": score,
                    "image_path": str(image_path),
                }
            )
    finally:
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "step": step,
        "checkpoint": "" if checkpoint is None else str(checkpoint),
        "load_kind": load_kind,
        "num_prompts": len(prompts),
        "mean_reward": float(np.mean(scores)) if scores else 0.0,
        "std_reward": float(np.std(scores)) if scores else 0.0,
    }
    return summary, per_prompt_rows


async def _main_async(args: argparse.Namespace) -> None:
    output_dir = Path(os.path.expanduser(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    prompts = _read_prompts(Path(os.path.expanduser(args.prompt_file)), args.max_prompts, args.max_prompt_lines)
    if not prompts:
        raise ValueError("Validation prompt file is empty.")

    reward_fn = _load_reward_fn(args.reward_function_path, args.reward_function_name)

    checkpoints: list[Path | None]
    if args.include_base:
        checkpoints = [None]
    else:
        checkpoints = []

    if args.checkpoint_paths:
        checkpoints.extend(Path(os.path.expanduser(path)) for path in args.checkpoint_paths)
    if args.checkpoint_dir:
        checkpoints.extend(_discover_checkpoints(Path(os.path.expanduser(args.checkpoint_dir))))

    if not checkpoints:
        raise ValueError("No checkpoints found. Set --checkpoint_dir, --checkpoint_paths, or --include_base.")

    summary_rows = []
    detail_rows = []
    for checkpoint in checkpoints:
        try:
            summary, details = await _validate_checkpoint(args, checkpoint, prompts, reward_fn, image_dir)
            summary_rows.append(summary)
            detail_rows.extend(details)
            print(f"step={summary['step']} mean_reward={summary['mean_reward']:.6f}")
        except Exception as exc:
            error_row = {
                "step": -1 if checkpoint is None else _parse_step(checkpoint),
                "checkpoint": "" if checkpoint is None else str(checkpoint),
                "error": repr(exc),
            }
            summary_rows.append(error_row)
            print(f"Failed to validate {checkpoint}: {exc}")
            if not args.continue_on_error:
                raise

    _write_jsonl(output_dir / "validation_summary.jsonl", summary_rows)
    _write_jsonl(output_dir / "validation_details.jsonl", detail_rows)
    _write_csv(output_dir / "validation_summary.csv", summary_rows)
    _write_csv(output_dir / "validation_details.csv", detail_rows)
    _plot_curve([row for row in summary_rows if "mean_reward" in row], output_dir / "reward_curve.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate offline DPO checkpoints on a prompt text file.")
    parser.add_argument("--prompt_file", required=True, help="Validation prompt .txt file, one prompt per line.")
    parser.add_argument(
        "--checkpoint_dir", default=None, help="Root directory containing global_step_*/actor checkpoints."
    )
    parser.add_argument("--checkpoint_paths", nargs="*", default=None, help="Explicit checkpoint files or directories.")
    parser.add_argument("--include_base", action="store_true", help="Also validate the base model without checkpoint.")
    parser.add_argument("--output_dir", required=True, help="Directory for logs, generated images, and reward curve.")
    parser.add_argument("--pipeline", choices=["auto", "sd3", "qwen_image"], default="auto")
    parser.add_argument("--model_path", default="Qwen/Qwen-Image")
    parser.add_argument("--data_source", default="offline_dpo_validation")
    parser.add_argument("--negative_prompt", default=" ")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=35)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--true_cfg_scale", type=float, default=4.0)
    parser.add_argument("--max_sequence_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--max_prompts", type=int, default=-1)
    parser.add_argument(
        "--max_prompt_lines",
        type=int,
        default=-1,
        help="Only read the first N lines from --prompt_file before filtering empty prompts. Use <=0 for all lines.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=0,
        help="LoRA rank to reconstruct adapters when checkpoint has LoRA weights but no adapter_config.json.",
    )
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument(
        "--lora_target_modules",
        default=None,
        help="Python list or comma-separated target modules. Defaults to Qwen-Image DPO LoRA targets.",
    )
    parser.add_argument("--reward_function_path", default="verl_omni/utils/reward_score/unified_reward.py")
    parser.add_argument("--reward_function_name", default="compute_score_unified_reward")
    parser.add_argument("--reward_router_address", default=None)
    parser.add_argument("--reward_model_name", default="CodeGoat24/UnifiedReward-2.0-qwen3vl-8b")
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
        help="Command template used with --launch_reward_server. Available placeholders: {model}, {host}, {port}.",
    )
    parser.add_argument("--reward_server_startup_timeout", type=int, default=900)
    parser.add_argument(
        "--reward_gpu",
        type=int,
        default=None,
        help="Physical CUDA device index for the reward server. If unset, inherit CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--image_gpu",
        type=int,
        default=None,
        help="Physical CUDA device index for image generation when --device is not set.",
    )
    parser.add_argument("--disable_progress", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument(
        "--fsdp_merge_backend",
        choices=["auto", "verl", "local", "verl_cli"],
        default="auto",
        help=(
            "How to merge multi-GPU FSDP actor shards before validation. "
            "'auto' uses verl.model_merger merge logic when available, otherwise a local merge. "
            "'verl_cli' exports HuggingFace weights via `python -m verl.model_merger merge` "
            "(see https://verl.readthedocs.io/en/latest/advance/checkpoint.html)."
        ),
    )
    parser.add_argument(
        "--fsdp_merge_use_cpu",
        action="store_true",
        help="Pass --use_cpu_initialization to verl.model_merger when --fsdp_merge_backend=verl_cli.",
    )
    args = parser.parse_args()
    get_pipeline_utils(args)
    _apply_gpu_device_defaults(args)
    print(f"Image generation device: {args.device}.")
    if args.launch_reward_server and args.reward_router_address is None:
        args.reward_router_address = f"{args.reward_server_host}:{args.reward_server_port}"
    if args.reward_function_path is not None and args.reward_router_address is None:
        raise ValueError(
            "Reward scoring requires --reward_router_address, or use --launch_reward_server to start one automatically."
        )
    with _maybe_launch_reward_server(args):
        asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
