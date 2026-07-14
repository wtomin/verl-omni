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
"""Entrypoint for Omni model RL training.

``run_omni`` routes to one of two backends:

* **V1 PPO** (GSPO, GRPO, and other online RL algorithms): delegates to verl's
  ``TaskRunnerV1`` with ``trainer.use_v1=True``.
* **Offline preference** (e.g. offline DPO): uses the shared ``RayTrainerTaskRunner``.
"""

from __future__ import annotations

import hydra
import ray
from omegaconf import OmegaConf
from verl.utils.device import auto_set_device

from verl_omni.trainer.ray_task_runner import RayTrainerTaskRunner, launch_ray_task_runner

__all__ = ["main", "run_omni", "uses_v1_trainer"]


def uses_v1_trainer(config) -> bool:
    """Return True when omni training should use verl's V1 PPO stack."""
    if OmegaConf.select(config, "trainer.use_v1", default=None) is False:
        return False

    sample_source = OmegaConf.select(config, "algorithm.sample_source", default="online")
    trainer_type = OmegaConf.select(config, "algorithm.trainer_type", default="policy_gradient")
    return not (sample_source == "offline" and trainer_type == "direct_preference")


def run_omni(config, task_runner_class=None) -> None:
    """Initialize Ray and run distributed Omni training."""
    if uses_v1_trainer(config):
        from verl.trainer.main_ppo import TaskRunnerV1

        config.trainer.use_v1 = True
        if task_runner_class is None:
            task_runner_class = TaskRunnerV1
        launch_ray_task_runner(
            config,
            task_runner_class,
            enable_transfer_queue_env=True,
            propagate_determinism=True,
        )
        return

    if task_runner_class is None:
        task_runner_class = ray.remote(num_cpus=1)(RayTrainerTaskRunner)
    launch_ray_task_runner(config, task_runner_class)


@hydra.main(config_path="./config", config_name="omni_trainer", version_base=None)
def main(config):
    """Main entry point for Omni model training with Hydra configuration management."""
    auto_set_device(config)
    if uses_v1_trainer(config):
        from verl.trainer.ppo.utils import need_critic, need_reference_policy
        from verl.utils.config import validate_config

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )
    OmegaConf.resolve(config)
    run_omni(config)


if __name__ == "__main__":
    main()
