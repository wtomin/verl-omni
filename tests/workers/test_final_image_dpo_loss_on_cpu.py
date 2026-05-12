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

import numpy as np
import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from verl_omni.workers.config import DiffusionActorConfig
from verl_omni.workers.utils.losses import diffusion_loss


def test_final_image_dpo_loss_does_not_require_log_probs_or_trajectory():
    batch_size = 4
    model_output = {
        "noise_pred_theta": torch.randn(batch_size, 4, 8, 8, requires_grad=True),
        "noise_pred_ref": torch.randn(batch_size, 4, 8, 8),
    }
    data = TensorDict(
        {
            "fm_velocity_target": torch.randn(batch_size, 4, 8, 8),
        },
        batch_size=batch_size,
    )
    tu.assign_non_tensor(
        data,
        dpo_pair_chosen_indices=np.asarray([0, 2], dtype=np.int64),
        dpo_pair_rejected_indices=np.asarray([1, 3], dtype=np.int64),
        gradient_accumulation_steps=1,
    )

    config = DiffusionActorConfig(strategy="fsdp", ppo_micro_batch_size_per_gpu=1, rollout_n=1)
    config.diffusion_loss.loss_mode = "dpo"

    loss, metrics = diffusion_loss(config=config, model_output=model_output, data=data)

    assert loss.requires_grad
    assert "actor/pg_loss" in metrics
    assert "actor/dpo_loss" in metrics
