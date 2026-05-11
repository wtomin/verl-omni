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
"""Diffusion-specific loss functions and KL penalties."""

from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from verl_omni.workers.config import DiffusionActorConfig

DiffusionLossFn = Callable[
    [
        torch.Tensor,  # old_log_prob
        torch.Tensor,  # log_prob
        torch.Tensor,  # advantages
        Optional[DictConfig | DiffusionActorConfig],  # config
    ],
    tuple[torch.Tensor, dict[str, Any]],
]

DIFFUSION_LOSS_REGISTRY: dict[str, DiffusionLossFn] = {}


def register_diffusion_loss(name: str) -> Callable[[DiffusionLossFn], DiffusionLossFn]:
    """Register a diffusion loss function with the given name.

    Args:
        name (str): The name to register the diffusion loss function under.

    Returns:
        function: Decorator function that registers the diffusion loss function.
    """

    def decorator(func: DiffusionLossFn) -> DiffusionLossFn:
        DIFFUSION_LOSS_REGISTRY[name] = func
        return func

    return decorator


def get_diffusion_loss_fn(name):
    """Get the diffusion loss with a given name.

    Args:
        name: `(str)`
            The name of the policy loss.

    Returns:
        `(callable)`: The policy loss function.
    """
    if name not in DIFFUSION_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported diffusion loss mode: {name}. Supported modes are: {list(DIFFUSION_LOSS_REGISTRY.keys())}"
        )
    return DIFFUSION_LOSS_REGISTRY[name]


class DiffusionAdvantageEstimator(str, Enum):
    """Advantage estimators specific to diffusion-based training."""

    FLOW_GRPO = "flow_grpo"
    DPO = "dpo"


DIFFUSION_ADV_ESTIMATOR_REGISTRY: dict[str, Any] = {}


def register_diffusion_adv_est(name_or_enum: str | DiffusionAdvantageEstimator) -> Any:
    """Register a diffusion advantage estimator function with the given name.

    Args:
        name_or_enum: `(str)` or `(DiffusionAdvantageEstimator)`
            The name or enum of the advantage estimator.

    """

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in DIFFUSION_ADV_ESTIMATOR_REGISTRY and DIFFUSION_ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(
                f"Diffusion adv estimator {name} has already been registered: "
                f"{DIFFUSION_ADV_ESTIMATOR_REGISTRY[name]} vs {fn}"
            )
        DIFFUSION_ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return decorator


def get_diffusion_adv_estimator_fn(name_or_enum):
    """Get the diffusion advantage estimator function with a given name."""
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in DIFFUSION_ADV_ESTIMATOR_REGISTRY:
        raise ValueError(
            f"Unknown diffusion advantage estimator: {name}. Supported: {list(DIFFUSION_ADV_ESTIMATOR_REGISTRY.keys())}"
        )
    return DIFFUSION_ADV_ESTIMATOR_REGISTRY[name]


@register_diffusion_adv_est(DiffusionAdvantageEstimator.FLOW_GRPO)
def compute_flow_grpo_outcome_advantage(
    sample_level_rewards: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-4,
    norm_adv_by_std_in_grpo: bool = True,
    global_std: bool = True,
    config: Optional[DictConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        sample_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        global_std: `(bool)`
            whether to use global std for advantage normalization
        config: `(Optional[DictConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = sample_level_rewards.clone()
    assert scores.ndim == 2
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        if global_std:
            batch_std = torch.std(scores)
        else:
            batch_std = None

        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = id2score[idx][0]
                if global_std:
                    id2std[idx] = batch_std
                else:
                    id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                if global_std:
                    id2std[idx] = batch_std
                else:
                    id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]

    return scores, scores


@register_diffusion_loss("flow_grpo")
def compute_diffusion_loss_flow_grpo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    config: Optional[DictConfig | DiffusionActorConfig] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute the clipped policy objective and related metrics for FlowGRPO.

    Adapted from
    https://github.com/yifan123/flow_grpo/blob/main/scripts/train_sd3_fast.py#L885

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size,).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size,).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size,).
        config (verl_omni.workers.config.DiffusionActorConfig):
            Config for the actor.
    """
    assert config is not None
    assert isinstance(config, DiffusionActorConfig)
    loss_cfg = config.diffusion_loss
    advantages = torch.clamp(
        advantages,
        -loss_cfg.adv_clip_max,
        loss_cfg.adv_clip_max,
    )
    log_ratio = log_prob - old_log_prob
    ratio = torch.exp(log_ratio)
    unclipped_loss = -advantages * ratio
    clipped_loss = -advantages * torch.clamp(
        ratio,
        1.0 - loss_cfg.clip_ratio,
        1.0 + loss_cfg.clip_ratio,
    )
    pg_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

    with torch.no_grad():
        ppo_kl = torch.mean(-log_ratio)
        pg_clipfrac = torch.mean((torch.abs(ratio - 1.0) > loss_cfg.clip_ratio).float())
        pg_clipfrac_higher = torch.mean((ratio - 1.0 > loss_cfg.clip_ratio).float())
        pg_clipfrac_lower = torch.mean((1.0 - ratio > loss_cfg.clip_ratio).float())

    pg_metrics = {
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/pg_clipfrac_higher": pg_clipfrac_higher.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


def kl_penalty_image(
    prev_sample_mean: torch.Tensor, ref_prev_sample_mean: torch.Tensor, std_dev_t: torch.Tensor
) -> torch.Tensor:
    """Compute KL divergence given previous sample mean and reference previous sample mean (for images or videos).
    Args:
        prev_sample_mean: (torch.Tensor) shape is (bs, s, c)
        ref_prev_sample_mean: (torch.Tensor) shape is (bs, s, c)
        std_dev_t: (torch.Tensor) shape is (bs, 1, 1)
    """
    kl_loss = ((prev_sample_mean - ref_prev_sample_mean) ** 2).mean(dim=(1, 2), keepdim=True) / (2 * std_dev_t**2)
    return kl_loss.mean()


@register_diffusion_adv_est(DiffusionAdvantageEstimator.DPO)
def compute_dpo_advantage(
    sample_level_rewards: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-4,
    norm_adv_by_std_in_grpo: bool = True,
    global_std: bool = True,
    config: Optional[DictConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for DPO training.
    DPO works with paired data (chosen/rejected), so we need to form pairs
    and compute advantages within each group.

    Args:
        sample_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length). For DPO, response_length should be 1
            since each sample has a single reward.
        index: `(np.ndarray)`
            index array for grouping (same prompt has same index).
        epsilon: `(float)`
            small value to avoid division by zero.
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the advantage by std (similar to GRPO).
        global_std: `(bool)`
            whether to use global std for advantage normalization.
        config: `(Optional[DictConfig])`
            algorithm configuration object, may contain DPO-specific parameters.

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length). For DPO, this is the advantage
            computed within each group (prompt).
        returns: `(torch.Tensor)`
            shape is (bs, response_length). Same as advantages for DPO.
    """
    scores = sample_level_rewards.clone()
    assert scores.ndim == 2
    # For DPO, we expect response_length = 1 (one reward per sample)
    # But we'll handle the general case

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        if global_std:
            batch_std = torch.std(scores)
        else:
            batch_std = None

        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = id2score[idx][0]
                if global_std:
                    id2std[idx] = batch_std
                else:
                    id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                if global_std:
                    id2std[idx] = batch_std
                else:
                    id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]

    return scores, scores


def _spatial_mse_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Channel/spatial mean squared error, one scalar per batch row."""
    if pred.shape != target.shape:
        raise ValueError(f"Mismatched shapes for FM DPO tensors: pred {pred.shape}, target {target.shape}")
    if pred.ndim <= 1:
        return (pred.float() - target.float()).square()
    spatial_dims = tuple(range(1, pred.ndim))
    return (pred.float() - target.float()).square().mean(dim=spatial_dims)


def compute_diffusion_dpo_fm_loss(
    policy_noise_pred: torch.Tensor,
    ref_noise_pred: torch.Tensor,
    fm_velocity_target: torch.Tensor,
    pair_chosen: torch.Tensor,
    pair_rejected: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Explicit diffusion DPO objective from pairwise flow-matching MSE gaps.

    Matches the logits-free loss used when comparing policy vs reference predictions
    to a velocity target ``noise - x_0`` (see diffusion DPO / contrastive reward literature).

    Loss (one term per preference pair):

        ``L = mean( -log σ( -β/2 * ( w_diff - l_diff ) ) )``

    where::

        ``w_diff = θ_err_w - ref_err_w``, ``l_diff = θ_err_l - ref_err_l``

    and θ_err denotes per-sample FM MSE ``mean((pred - target)²)`` over spatial dimensions.

    Args:
        policy_noise_pred: Policy (θ) transformer output velocity prediction, batched like training rollout.
        ref_noise_pred: Reference velocity prediction from the frozen base weights.
        fm_velocity_target: Regression target aligned with ``policy_noise_pred`` (e.g. ``noise - x_0``).
        pair_chosen: Long indices ``(num_pairs,)`` into batch dimension for preferred samples ``w``.
        pair_rejected: Long indices ``(num_pairs,)`` into batch dimension for dispreferred samples ``l``.
        beta: Temperature from DPO preference strength.

    Returns:
        Scalar loss tensor and a metrics dictionary.
    """
    if pair_chosen.numel() == 0:
        return policy_noise_pred.sum() * 0.0, {"actor/dpo_loss": 0.0, "actor/dpo_accuracy": 0.0}

    theta_err = _spatial_mse_per_sample(policy_noise_pred, fm_velocity_target)
    ref_err = _spatial_mse_per_sample(ref_noise_pred.detach(), fm_velocity_target)

    theta_w_err = theta_err.index_select(0, pair_chosen)
    theta_l_err = theta_err.index_select(0, pair_rejected)
    ref_w_err = ref_err.index_select(0, pair_chosen)
    ref_l_err = ref_err.index_select(0, pair_rejected)

    w_diff = theta_w_err - ref_w_err
    l_diff = theta_l_err - ref_l_err
    w_l_diff = w_diff - l_diff
    inside_term = -0.5 * beta * w_l_diff
    dpo_loss = -torch.nn.functional.logsigmoid(inside_term).mean()

    with torch.no_grad():
        implicit_reward_chosen = -0.5 * beta * (theta_w_err - ref_w_err)
        implicit_reward_rejected = -0.5 * beta * (theta_l_err - ref_l_err)
        implicit_accuracy = (implicit_reward_chosen > implicit_reward_rejected).float().mean()

    dpo_metrics = {
        "actor/dpo_loss": dpo_loss.detach().item(),
        "actor/dpo_implicit_accuracy": implicit_accuracy.detach().item(),
        "actor/mean_implicit_reward_chosen": implicit_reward_chosen.mean().detach().item(),
        "actor/mean_implicit_reward_rejected": implicit_reward_rejected.mean().detach().item(),
        "actor/mean_fm_mse_theta": theta_err.mean().detach().item(),
        "actor/mean_fm_mse_ref": ref_err.mean().detach().item(),
    }
    return dpo_loss, dpo_metrics


@register_diffusion_loss("dpo")
def compute_diffusion_loss_dpo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    config: Optional[DictConfig | DiffusionActorConfig] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Fallback DPO surrogate when pairwise FM tensors are unavailable per step.

    This path expects ``advantages`` to hold *(only on participating rows)* the quantity::

        ``(θ_err_w - ref_err_w) - (θ_err_l - ref_err_l)``

    so that::

        ``L = mean( -log σ( -β/2 * advantages ) )``.

    Rows with negligible magnitude are masked out so unrelated batch elements do not dilute gradients.
    """
    assert config is not None

    beta = getattr(config.diffusion_loss, "dpo_beta", 100.0)
    eps = 1e-8
    vals = advantages
    mask = vals.detach().abs() > eps
    if mask.any():
        inside_term = -0.5 * beta * vals[mask]
        dpo_loss = -torch.nn.functional.logsigmoid(inside_term).mean()
    else:
        dpo_loss = torch.zeros((), dtype=advantages.dtype, device=advantages.device)

    with torch.no_grad():
        vd = vals.detach()
        mean_gap = vd.mean() if vd.numel() > 0 else vd.sum()
        std_gap = vd.std(unbiased=False) if vd.numel() > 1 else vd.new_zeros(())
        accuracy = (vd[mask].float() > 0).float().mean() if mask.any() else vd.new_zeros((), dtype=torch.float32)

    dpo_metrics = {
        "actor/dpo_loss": dpo_loss.detach().item(),
        "actor/mean_mse_gap": mean_gap.detach().item(),
        "actor/std_mse_gap": std_gap.detach().item(),
        "actor/dpo_accuracy": accuracy.detach().item(),
    }

    return dpo_loss, dpo_metrics
