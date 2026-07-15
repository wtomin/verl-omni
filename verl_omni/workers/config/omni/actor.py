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

from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING
from verl.base_config import BaseConfig
from verl.trainer.config import CheckpointConfig
from verl.utils.profiler import ProfilerConfig
from verl.workers.config.engine import FSDPEngineConfig
from verl.workers.config.optimizer import FSDPOptimizerConfig

from .model import OmniModelConfig

__all__ = [
    "OmniLossConfig",
    "FSDPOmniActorConfig",
]


@dataclass
class OmniLossConfig(BaseConfig):
    loss_mode: str = "dpo"
    beta: float = 0.1
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"
    reference_free: bool = False
    average_log_prob: bool = False
    refer_model_precision: str = "bfloat16"

    def __post_init__(self):
        if self.loss_mode not in {"dpo"}:
            raise ValueError(f"Unsupported omni loss_mode={self.loss_mode!r}; currently supported: ['dpo'].")
        if self.loss_type not in {"sigmoid", "ipo"}:
            raise ValueError(f"Invalid omni DPO loss_type={self.loss_type!r}; expected 'sigmoid' or 'ipo'.")
        if self.beta <= 0:
            raise ValueError(f"Omni DPO beta must be positive, got {self.beta}.")


@dataclass
class FSDPOmniActorConfig(BaseConfig):
    _mutable_fields = BaseConfig._mutable_fields | {
        "ppo_mini_batch_size",
        "ppo_micro_batch_size_per_gpu",
        "engine",
        "model_config",
    }

    strategy: str = "fsdp"
    ppo_mini_batch_size: int = 256
    ppo_micro_batch_size_per_gpu: int = MISSING
    ppo_epochs: int = 1
    shuffle: bool = False
    data_loader_seed: int = 42
    loss_scale_factor: Optional[float] = None
    use_kl_loss: bool = False
    kl_loss_coef: float = 0.001
    rollout_n: int = MISSING
    log_prob_micro_batch_size_per_gpu: Optional[int] = None
    global_batch_info: dict = field(default_factory=dict)
    omni_loss: OmniLossConfig = field(default_factory=OmniLossConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    optim: FSDPOptimizerConfig = field(default_factory=FSDPOptimizerConfig)
    engine: BaseConfig = field(default_factory=BaseConfig)
    model_config: OmniModelConfig = field(default_factory=BaseConfig)
    profiler: Optional[ProfilerConfig] = None

    def __post_init__(self):
        if self.strategy not in {"fsdp", "fsdp2"}:
            raise ValueError(f"FSDP omni actor requires strategy='fsdp' or 'fsdp2', got {self.strategy!r}")
        assert self.rollout_n != MISSING
        self.engine = self.fsdp_config
        object.__setattr__(self.engine, "strategy", self.strategy)
