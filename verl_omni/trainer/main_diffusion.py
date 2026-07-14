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
"""Entrypoint for diffusion model RL training."""

import hydra
import ray
from omegaconf import OmegaConf
from verl.utils.device import auto_set_device

from verl_omni.trainer.ray_task_runner import RayTrainerTaskRunner, launch_ray_task_runner
from verl_omni.utils.diffusion_attention import fallback_fa3_if_unavailable, validate_attention_consistency

# Backward-compatible alias used by docs and external imports.
TaskRunner = RayTrainerTaskRunner


@hydra.main(config_path="./config", config_name="diffusion_trainer", version_base=None)
def main(config):
    """Main entry point for diffusion model training with Hydra configuration management."""
    auto_set_device(config)
    OmegaConf.resolve(config)
    fallback_fa3_if_unavailable(config)
    validate_attention_consistency(config)
    run_diffusion(config)


def run_diffusion(config, task_runner_class=None) -> None:
    """Initialize Ray and run distributed diffusion training."""
    if task_runner_class is None:
        task_runner_class = ray.remote(num_cpus=1)(RayTrainerTaskRunner)
    launch_ray_task_runner(config, task_runner_class)


if __name__ == "__main__":
    main()
