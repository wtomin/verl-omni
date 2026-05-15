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
import pytest
import torch
import torch.nn.functional as F

from verl_omni.trainer.diffusion.diffusion_algos import compute_diffusion_loss_dpo
from verl_omni.workers.config.diffusion.actor import DiffusionLossConfig, FSDPDiffusionActorConfig


def _dpo_actor_config(*, dpo_beta: float = 1.0) -> FSDPDiffusionActorConfig:
    return FSDPDiffusionActorConfig(
        strategy="fsdp",
        ppo_micro_batch_size_per_gpu=4,
        rollout_n=2,
        diffusion_loss=DiffusionLossConfig(loss_mode="dpo", dpo_beta=dpo_beta),
    )


def test_compute_diffusion_loss_dpo_matches_manual_reference() -> None:
    """Two identical chosen/rejected pairs; closed-form check vs -logsigmoid."""
    beta = 0.5
    cfg = _dpo_actor_config(dpo_beta=beta)

    # (batch, dim): model matches noise → model_err=0; ref is worse on chosen rows.
    noise = torch.zeros((4, 3), dtype=torch.float32)
    model_noise_pred = noise.clone()
    ref_noise_pred = torch.zeros_like(noise)
    ref_noise_pred[0::2] = 10.0  # chosen: high squared error
    ref_noise_pred[1::2] = 1.0  # rejected: lower than chosen but still non-zero

    sample_level_scores = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)

    loss, metrics = compute_diffusion_loss_dpo(
        noise=noise,
        model_noise_pred=model_noise_pred,
        ref_noise_pred=ref_noise_pred,
        sample_level_scores=sample_level_scores,
        config=cfg,
        index=np.array(["p0", "p0", "p1", "p1"], dtype=object),
    )

    model_err = (model_noise_pred - noise).flatten(1).norm(dim=1).pow(2)
    ref_err = (ref_noise_pred - noise).flatten(1).norm(dim=1).pow(2)
    w_diff = model_err[0] - ref_err[0]
    l_diff = model_err[1] - ref_err[1]
    inside_term = -beta * (w_diff - l_diff)
    expected = -F.logsigmoid(inside_term)

    torch.testing.assert_close(loss, expected)
    assert metrics["actor/dpo_loss"] == pytest.approx(loss.detach().item())
    assert metrics["actor/dpo_accuracy"] == pytest.approx(
        float((inside_term > 0).float().mean().item())
    )


def test_compute_diffusion_loss_dpo_accepts_torch_index() -> None:
    cfg = _dpo_actor_config(dpo_beta=1.0)
    noise = torch.randn((2, 4, 4), dtype=torch.float32)
    model_noise_pred = noise + 0.01 * torch.randn_like(noise)
    ref_noise_pred = noise + 0.02 * torch.randn_like(noise)

    loss, metrics = compute_diffusion_loss_dpo(
        noise=noise,
        model_noise_pred=model_noise_pred,
        ref_noise_pred=ref_noise_pred,
        sample_level_scores=torch.tensor([[0.9], [0.1]], dtype=torch.float32),
        config=cfg,
        index=torch.tensor([7, 7], dtype=torch.long),
    )

    assert loss.shape == ()
    assert "actor/dpo_loss" in metrics
    assert "actor/dpo_accuracy" in metrics


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        (
            {
                "noise": torch.zeros((3, 2)),
                "model_noise_pred": torch.zeros((3, 2)),
                "ref_noise_pred": torch.zeros((3, 2)),
                "sample_level_scores": torch.tensor([1.0, 0.0, 1.0]),
            },
            "even batch",
        ),
        (
            {"sample_level_scores": torch.tensor([0.0, 1.0, 1.0, 0.0])},
            "chosen sample reward",
        ),
        (
            {"index": np.array([0, 1, 0, 1])},
            "same prompt uid",
        ),
    ],
)
def test_compute_diffusion_loss_dpo_validation(extra: dict, match: str) -> None:
    cfg = _dpo_actor_config()
    base = dict(
        noise=torch.zeros((4, 2)),
        model_noise_pred=torch.zeros((4, 2)),
        ref_noise_pred=torch.zeros((4, 2)),
        sample_level_scores=torch.tensor([1.0, 0.0, 1.0, 0.0]),
        config=cfg,
    )
    base.update(extra)

    with pytest.raises(ValueError, match=match):
        compute_diffusion_loss_dpo(**base)
