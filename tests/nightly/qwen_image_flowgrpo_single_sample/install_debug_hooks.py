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

"""Monkey-patch nightly-only debug dumps into the diffusion trainer and engine."""

from __future__ import annotations

import functools
import os
from typing import Any

from verl.utils import tensordict_utils as tu

from tests.nightly.qwen_image_flowgrpo_single_sample.debug_dumper import (
    DebugDumper,
    get_dump_config_from_env,
)

_WRAPPED_ATTR = "_qwen_image_flowgrpo_nightly_debug_wrapped"


def _is_wrapped(func: Any) -> bool:
    return bool(getattr(func, _WRAPPED_ATTR, False))


def _mark_wrapped(func: Any) -> Any:
    setattr(func, _WRAPPED_ATTR, True)
    return func


def _debug_metadata(step: int | None) -> dict[str, Any]:
    dump_dir, dump_steps, mode = get_dump_config_from_env()
    if not dump_dir or step is None:
        return {}
    return {
        "debug_step": int(step),
        "debug_dump_dir": dump_dir,
        "debug_dump_steps": list(dump_steps),
        "debug_dump_mode": mode,
    }


def _patch_policy_gradient_trainer() -> None:
    from verl_omni.trainer.diffusion.ray_diffusion_trainer import BaseRayDiffusionTrainer

    original_update_actor = BaseRayDiffusionTrainer._update_actor
    if _is_wrapped(original_update_actor):
        return

    @functools.wraps(original_update_actor)
    def wrapped_update_actor(self, batch):
        step = getattr(self, "global_steps", None)
        metadata = _debug_metadata(step)
        if metadata:
            DebugDumper.from_env().dump_driver_forward(step, batch)
            batch.meta_info.update(metadata)
        return original_update_actor(self, batch)

    BaseRayDiffusionTrainer._update_actor = _mark_wrapped(wrapped_update_actor)


def _patch_diffusers_engine() -> None:
    from verl_omni.workers.engine.fsdp.diffusers_impl import DiffusersFSDPEngine

    original_run = DiffusersFSDPEngine._run_forward_backward_batch
    if not _is_wrapped(original_run):

        @functools.wraps(original_run)
        def wrapped_run_forward_backward_batch(self, data, loss_function, forward_only, *, timesteps_key):
            step = tu.get_non_tensor_data(data, "debug_step", default=None)
            dump_dir = tu.get_non_tensor_data(data, "debug_dump_dir", default=os.environ.get("DEBUG_DUMP_DIR"))
            dump_steps = tu.get_non_tensor_data(data, "debug_dump_steps", default=None)
            dump_mode = tu.get_non_tensor_data(
                data,
                "debug_dump_mode",
                default=os.environ.get("DEBUG_DUMP_MODE", "signature"),
            )

            self._nightly_debug_step = int(step) if step is not None else None
            self._nightly_debug_dumper = DebugDumper(dump_dir=dump_dir, dump_steps=dump_steps, mode=dump_mode)

            output = original_run(self, data, loss_function, forward_only, timesteps_key=timesteps_key)
            if isinstance(output, dict):
                self._nightly_debug_dumper.dump_actor_forward(self._nightly_debug_step, output)
            return output

        DiffusersFSDPEngine._run_forward_backward_batch = _mark_wrapped(wrapped_run_forward_backward_batch)

    original_optimizer_step = DiffusersFSDPEngine.optimizer_step
    if not _is_wrapped(original_optimizer_step):

        @functools.wraps(original_optimizer_step)
        def wrapped_optimizer_step(self):
            try:
                step = getattr(self, "_nightly_debug_step", None)
                dumper = getattr(self, "_nightly_debug_dumper", None)
                if dumper is not None and hasattr(self.module, "named_parameters"):
                    dumper.dump_lora_gradients(step, self.module.named_parameters())
                return original_optimizer_step(self)
            finally:
                self._nightly_debug_step = None
                self._nightly_debug_dumper = None

        DiffusersFSDPEngine.optimizer_step = _mark_wrapped(wrapped_optimizer_step)


def install_debug_hooks() -> None:
    """Install all nightly debug hooks.

    The hooks are no-ops unless DEBUG_DUMP_DIR is set. This function is safe to call in the driver,
    Ray task runner, and Ray worker processes.
    """
    _patch_policy_gradient_trainer()
    _patch_diffusers_engine()
