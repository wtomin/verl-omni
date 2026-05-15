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

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl import DataProto

from verl_omni.trainer.diffusion.ray_diffusion_trainer import RayFlowGRPOTrainer


class _MaterializeDPOFlowWorker:
    def __init__(self, batch_size: int):
        self.dpo_noise = torch.randn(batch_size, 16, 8, 8)
        self.dpo_timesteps = torch.arange(batch_size, dtype=torch.float32)

    def materialize_dpo_flow_batch(self, data: TensorDict) -> TensorDict:
        assert data["prompt_embeds"].is_nested
        batch_size = data.batch_size[0]
        return TensorDict(
            {
                # This duplicate key intentionally has a different sequence length
                # than the original batch. `_materialize_dpo_flow_batch` must not
                # pass it into DataProto.union.
                "prompt_embeds": torch.zeros(batch_size, 256, 4),
                "dpo_noise": self.dpo_noise,
                "dpo_timesteps": self.dpo_timesteps,
            },
            batch_size=[batch_size],
        )


class _MissingDPOFlowKeyWorker:
    def materialize_dpo_flow_batch(self, data: TensorDict) -> TensorDict:
        return TensorDict({"dpo_noise": torch.randn(data.batch_size[0], 16, 8, 8)}, batch_size=data.batch_size)


def _make_trainer(actor_rollout_wg) -> RayFlowGRPOTrainer:
    trainer = object.__new__(RayFlowGRPOTrainer)
    trainer.actor_rollout_wg = actor_rollout_wg
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {
                    "pipeline": {"height": 64, "width": 64},
                    "vae_scale_factor": 8,
                }
            }
        }
    )
    return trainer


def _make_batch(batch_size: int = 2) -> DataProto:
    max_seq_len = 2048
    hidden_size = 4
    valid_seq_len = 256
    prompt_embeds = torch.randn(batch_size, max_seq_len, hidden_size)
    prompt_embeds_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.int32)
    prompt_embeds_mask[:, :valid_seq_len] = 1
    return DataProto.from_tensordict(
        TensorDict(
            {
                "prompt_embeds": prompt_embeds,
                "prompt_embeds_mask": prompt_embeds_mask,
            },
            batch_size=[batch_size],
        )
    )


def test_materialize_dpo_flow_batch_unions_only_dpo_tensors() -> None:
    worker = _MaterializeDPOFlowWorker(batch_size=2)
    trainer = _make_trainer(worker)
    batch = _make_batch()

    result = trainer._materialize_dpo_flow_batch(batch)

    assert result is batch
    assert result.batch["prompt_embeds"].shape == (2, 2048, 4)
    torch.testing.assert_close(result.batch["dpo_noise"], worker.dpo_noise)
    torch.testing.assert_close(result.batch["dpo_timesteps"], worker.dpo_timesteps)


def test_materialize_dpo_flow_batch_requires_noise_and_timesteps() -> None:
    trainer = _make_trainer(_MissingDPOFlowKeyWorker())

    with pytest.raises(KeyError, match="dpo_timesteps"):
        trainer._materialize_dpo_flow_batch(_make_batch())
