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


import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric

from verl_omni.trainer.diffusion.diffusion_algos import (
    compute_diffusion_dpo_fm_loss,
    get_diffusion_loss_fn,
    kl_penalty_image,
)
from verl_omni.workers.config import DiffusionActorConfig


def diffusion_loss(config: DiffusionActorConfig, model_output, data: TensorDict, dp_group=None):
    """Compute loss for diffusion model"""
    log_prob = model_output["log_probs"]

    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    metrics = {}

    # compute policy loss
    loss_mode = config.diffusion_loss.get("loss_mode", "flow_grpo")
    beta = getattr(config.diffusion_loss, "dpo_beta", 100.0)
    old_log_prob = data.get("old_log_probs", None)
    advantages = data.get("advantages", None)

    pc_raw = tu.get_non_tensor_data(data, "dpo_pair_chosen_indices", default=None)
    pr_raw = tu.get_non_tensor_data(data, "dpo_pair_rejected_indices", default=None)
    fm_ready = (
        loss_mode == "dpo"
        and model_output.get("noise_pred_theta") is not None
        and model_output.get("noise_pred_ref") is not None
        and data.get("fm_velocity_target", None) is not None
        and pc_raw is not None
        and pr_raw is not None
        and len(pc_raw) > 0
    )

    if fm_ready:
        device = model_output["noise_pred_theta"].device
        pc = torch.as_tensor(pc_raw, device=device, dtype=torch.long)
        pr = torch.as_tensor(pr_raw, device=device, dtype=torch.long)
        pg_loss, pg_metrics = compute_diffusion_dpo_fm_loss(
            policy_noise_pred=model_output["noise_pred_theta"],
            ref_noise_pred=model_output["noise_pred_ref"],
            fm_velocity_target=data["fm_velocity_target"],
            pair_chosen=pc,
            pair_rejected=pr,
            beta=beta,
        )
    else:
        if advantages is None:
            raise KeyError(f'"advantages" is required for diffusion loss mode {loss_mode!r}')
        if old_log_prob is None:
            if loss_mode != "dpo":
                raise KeyError(f'"old_log_probs" is required for diffusion loss mode {loss_mode!r}')
            old_log_prob = torch.zeros_like(log_prob)
        policy_loss_fn = get_diffusion_loss_fn(loss_mode)
        pg_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            config=config,
        )

    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=AggregationType.MEAN)
    policy_loss = pg_loss

    if config.use_kl_loss:
        ref_prev_sample_mean = data["ref_prev_sample_mean"]
        prev_sample_mean = model_output["prev_sample_mean"]
        std_dev_t = model_output["std_dev_t"]
        kl_loss = kl_penalty_image(
            prev_sample_mean=prev_sample_mean, ref_prev_sample_mean=ref_prev_sample_mean, std_dev_t=std_dev_t
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=AggregationType.MEAN)
        metrics["kl_coef"] = config.kl_loss_coef

    gradient_accumulation_steps = tu.get_non_tensor_data(data, "gradient_accumulation_steps", default=None)
    policy_loss = policy_loss / gradient_accumulation_steps

    return policy_loss, metrics
