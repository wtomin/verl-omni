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

"""Local tensor dump helpers for the Qwen-Image FlowGRPO nightly."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch

DEFAULT_DUMP_STEPS = (1, 2)
VALID_DUMP_MODES = {"signature", "full"}


def parse_dump_steps(value: str | None) -> tuple[int, ...]:
    """Parse DEBUG_DUMP_STEPS into a stable, de-duplicated tuple."""
    if value is None or value.strip() == "":
        return DEFAULT_DUMP_STEPS

    steps: list[int] = []
    for raw_step in value.split(","):
        raw_step = raw_step.strip()
        if not raw_step:
            continue
        step = int(raw_step)
        if step <= 0:
            raise ValueError(f"DEBUG_DUMP_STEPS must contain positive integers, got {step}")
        steps.append(step)
    return tuple(sorted(set(steps)))


def get_dump_config_from_env() -> tuple[str | None, tuple[int, ...], str]:
    dump_dir = os.environ.get("DEBUG_DUMP_DIR")
    dump_steps = parse_dump_steps(os.environ.get("DEBUG_DUMP_STEPS"))
    mode = os.environ.get("DEBUG_DUMP_MODE", "signature").strip().lower()
    if mode not in VALID_DUMP_MODES:
        raise ValueError(f"DEBUG_DUMP_MODE must be one of {sorted(VALID_DUMP_MODES)}, got {mode!r}")
    return dump_dir, dump_steps, mode


def _rank_id() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def _to_local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if hasattr(tensor, "to_local"):
        tensor = tensor.to_local()
    return tensor.detach().cpu()


def _tensor_signature(tensor: torch.Tensor) -> dict[str, Any]:
    tensor = _to_local_tensor(tensor)
    meta: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
    }
    if tensor.numel() == 0:
        return meta

    stats_tensor = tensor.float()
    meta.update(
        {
            "mean": float(stats_tensor.mean()),
            "std": float(stats_tensor.std(unbiased=False)),
            "min": float(stats_tensor.min()),
            "max": float(stats_tensor.max()),
            "norm": float(stats_tensor.norm()),
        }
    )
    return meta


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, torch.Tensor):
        return _tensor_signature(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return repr(value)


def _flatten_tensors(prefix: str, value: Any, out: dict[str, torch.Tensor]) -> None:
    if isinstance(value, torch.Tensor):
        out[prefix] = value
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _flatten_tensors(f"{prefix}.{key}" if prefix else str(key), item, out)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _flatten_tensors(f"{prefix}.{index}" if prefix else str(index), item, out)


class DebugDumper:
    """Write per-rank debug artifacts from the process that owns the tensors."""

    def __init__(
        self,
        dump_dir: str | os.PathLike[str] | None,
        dump_steps: Iterable[int] | None = None,
        mode: str = "signature",
    ) -> None:
        self.dump_dir = Path(dump_dir) if dump_dir else None
        self.dump_steps = set(dump_steps or DEFAULT_DUMP_STEPS)
        self.mode = mode
        self.enabled = self.dump_dir is not None

    @classmethod
    def from_env(cls) -> DebugDumper:
        dump_dir, dump_steps, mode = get_dump_config_from_env()
        return cls(dump_dir=dump_dir, dump_steps=dump_steps, mode=mode)

    def should_dump(self, step: int | None) -> bool:
        return self.enabled and step is not None and int(step) in self.dump_steps

    def _save(self, step: int | None, subdir: str, tensors: Mapping[str, Any]) -> None:
        if not self.should_dump(step):
            return
        assert self.dump_dir is not None

        flat_tensors: dict[str, torch.Tensor] = {}
        for key, value in tensors.items():
            _flatten_tensors(str(key), value, flat_tensors)

        out_dir = self.dump_dir / subdir / f"step_{int(step)}" / f"rank_{_rank_id()}"
        out_dir.mkdir(parents=True, exist_ok=True)

        meta = {name: _tensor_signature(tensor) for name, tensor in flat_tensors.items()}
        extras = {
            str(key): _jsonable(value)
            for key, value in tensors.items()
            if not isinstance(value, torch.Tensor) and str(key) not in flat_tensors
        }
        if extras:
            meta["_non_tensor"] = extras

        if self.mode == "full":
            cpu_tensors = {name: _to_local_tensor(tensor) for name, tensor in flat_tensors.items()}
            torch.save(cpu_tensors, out_dir / "tensors.pt")

        with (out_dir / "meta.json").open("w", encoding="utf-8") as file:
            json.dump(meta, file, indent=2, sort_keys=True)

    def dump_driver_forward(self, step: int | None, batch: Any) -> None:
        keys = (
            "responses",
            "rollout_log_probs",
            "all_latents",
            "all_log_probs",
            "old_log_probs",
            "ref_log_prob",
            "advantages",
            "sample_level_rewards",
        )
        batch_tensors = getattr(batch, "batch", {})
        tensors = {key: batch_tensors[key] for key in keys if key in batch_tensors}
        self._save(step, "driver_forward", tensors)

    def dump_actor_forward(self, step: int | None, output: Mapping[str, Any]) -> None:
        tensors: dict[str, Any] = {}
        if "model_output" in output:
            tensors["model_output"] = output["model_output"]
        if "loss" in output:
            tensors["loss"] = output["loss"]
        self._save(step, "actor_forward", tensors)

    def dump_lora_gradients(self, step: int | None, named_params: Iterable[tuple[str, torch.nn.Parameter]]) -> None:
        grads: dict[str, torch.Tensor] = {}
        for name, param in named_params:
            if "lora_" not in name or param.grad is None:
                continue
            grads[name] = param.grad
        self._save(step, "lora_gradients", grads)
