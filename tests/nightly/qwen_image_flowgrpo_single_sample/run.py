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

"""Nightly entrypoint for Qwen-Image FlowGRPO single-sample regression."""

from __future__ import annotations

import os
import sys
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
HOOK_FQN = "tests.nightly.qwen_image_flowgrpo_single_sample.install_debug_hooks.install_debug_hooks"


def _prepend_pythonpath(path: str) -> str:
    current = os.environ.get("PYTHONPATH")
    if current:
        parts = current.split(os.pathsep)
        if path in parts:
            return current
        return os.pathsep.join([path, current])
    return path


def _merge_runtime_env(runtime_env: MutableMapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(runtime_env or {})
    env_vars = dict(merged.get("env_vars", {}))

    repo_root = str(REPO_ROOT)
    env_vars["PYTHONPATH"] = _prepend_pythonpath(repo_root)
    for key in ("DEBUG_DUMP_DIR", "DEBUG_DUMP_STEPS", "DEBUG_DUMP_MODE"):
        if key in os.environ:
            env_vars[key] = os.environ[key]

    merged["env_vars"] = env_vars
    merged["worker_process_setup_hook"] = HOOK_FQN
    return merged


def _patch_ray_init() -> None:
    import ray

    original_ray_init = ray.init
    if getattr(original_ray_init, "_qwen_image_flowgrpo_nightly_wrapped", False):
        return

    def wrapped_ray_init(*args, **kwargs):
        kwargs["runtime_env"] = _merge_runtime_env(kwargs.get("runtime_env"))
        return original_ray_init(*args, **kwargs)

    wrapped_ray_init._qwen_image_flowgrpo_nightly_wrapped = True
    ray.init = wrapped_ray_init


def _ensure_import_path() -> None:
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    os.environ["PYTHONPATH"] = _prepend_pythonpath(repo_root)


def main() -> None:
    _ensure_import_path()

    from tests.nightly.qwen_image_flowgrpo_single_sample.install_debug_hooks import install_debug_hooks
    from verl_omni.trainer.main_diffusion import main as diffusion_main

    install_debug_hooks()
    _patch_ray_init()
    diffusion_main()


if __name__ == "__main__":
    main()
