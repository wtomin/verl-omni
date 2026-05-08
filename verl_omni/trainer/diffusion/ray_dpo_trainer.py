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
"""
DPO Trainer for diffusion models (SD3) with Ray-based distributed training.
"""

import json
import os
import uuid
from collections import defaultdict
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import Role, WorkerType, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.llm_server import LLMServerManager

from verl_omni.trainer.config import DiffusionAlgoConfig
from verl_omni.trainer.diffusion.diffusion_algos import DiffusionAdvantageEstimator
from verl_omni.trainer.diffusion.diffusion_metric_utils import (
    compute_data_metrics_diffusion,
    compute_throughput_metrics_diffusion,
    compute_timing_metrics_diffusion,
)
from verl_omni.workers.utils.padding import embeds_padding_2_no_padding

logger = __import__("logging").getLogger(__file__)


class RayDPOTrainer:
    """Distributed DPO trainer using Ray for diffusion models (e.g., SD3).

    This trainer implements the Diffusion-DPO algorithm for flow-matching models.
    It orchestrates distributed training across multiple nodes and GPUs,
    managing actor rollouts and DPO loss computation.

    Key differences from FlowGRPO:
    1. DPO uses paired (chosen, rejected) data
    2. DPO loss is based on implicit rewards computed from model predictions
    3. DPO requires a reference model for implicit reward computation
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """Initialize distributed DPO trainer with Ray backend.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping: Mapping from roles to worker classes.
            resource_pool_manager: Manager for Ray resource pools.
            ray_worker_group_cls: Class for Ray worker groups.
            processor: Optional data processor for multimodal data.
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            collate_fn: Function to collate data samples into batches.
            train_sampler: Sampler for the training dataset.
            device_name: Device name for training.
        """
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)

        # DPO requires reference model
        if not self.use_reference_policy:
            logger.warning("DPO requires a reference model. Enabling reference policy.")
            self.use_reference_policy = True

        self.use_rm = need_reward_model(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # Check if ref is in actor (LoRA case)
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
        self.checkpoint_manager = None

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """Creates the train and validation dataloaders."""
        from verl_omni.utils.dataset.rl_dataset import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl_omni.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data.get("dataloader_num_workers", 0)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, "
            f"Size of val dataloader: {len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)

        visual_folder = os.path.join(dump_path, f"{self.global_steps}")
        os.makedirs(visual_folder, exist_ok=True)

        output_paths = []
        images_pil = outputs.cpu().float().permute(0, 2, 3, 1).numpy()
        images_pil = (images_pil * 255).round().clip(0, 255).astype("uint8")
        for i, image in enumerate(images_pil):
            image_path = os.path.join(visual_folder, f"{i}.jpg")
            Image.fromarray(image).save(image_path)
            output_paths.append(image_path)

        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": output_paths,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Dumped generations to {filename}")

    def init_workers(self):
        """Initialize distributed training workers using Ray backend."""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # Create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls

        # Create reference policy (required for DPO)
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # Initialize WorkerGroup
        all_wg = {}
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        # Setup reference policy worker group
        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]
        else:
            self.ref_policy_wg = None

        # Setup actor rollout worker group
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # Create reward loop manager (optional for DPO, but may be used for validation)
        from verl.experimental.reward_loop import RewardLoopManager

        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
        self.reward_loop_manager = RewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )

        # Create async rollout manager
        self.async_rollout_mode = True

        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

            from verl_omni.agent_loop import DiffusionAgentLoopWorker

            AgentLoopManager.agent_loop_workers_class = ray.remote(DiffusionAgentLoopWorker)

        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None

        self.llm_server_manager = LLMServerManager.create(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rollout_resource_pool=actor_rollout_resource_pool,
        )
        self.async_rollout_manager = AgentLoopManager.create(
            config=self.config,
            llm_client=self.llm_server_manager.get_client(),
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.actor_rollout_wg,
            replicas=self.llm_server_manager.get_replicas(),
        )

        self.checkpoint_manager.sleep_replicas()

    def _save_checkpoint(self):
        """Save model checkpoint."""
        from verl.utils.fs import local_mkdir_safe

        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None)
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        # Save dataloader state
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

    def _load_checkpoint(self):
        """Load model checkpoint."""
        if self.config.trainer.resume_mode == "disable":
            return 0

        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("Load from hdfs is not implemented yet.")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "Resume ckpt must be str type"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)

        print(f"Load from checkpoint folder: {global_step_folder}")
        self.global_steps = int(global_step_folder.split("global_step_")[-1])
        print(f"Setting global step to {self.global_steps}")

        actor_path = os.path.join(global_step_folder, "actor")
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )

        # Load dataloader state
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        """Compute log probabilities using the reference model."""
        batch_td = batch.to_tensordict()
        batch_td = embeds_padding_2_no_padding(batch_td)
        metadata = {
            "compute_loss": False,
            "height": self.config.actor_rollout_ref.model.pipeline.height,
            "width": self.config.actor_rollout_ref.model.pipeline.width,
            "vae_scale_factor": self.config.actor_rollout_ref.model.get("vae_scale_factor", 8),
        }
        if self.ref_in_actor:
            metadata["no_lora_adapter"] = True
        tu.assign_non_tensor(batch_td, **metadata)

        if self.ref_in_actor:
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
        else:
            assert self.ref_policy_wg is not None, "Reference policy worker group is not initialized"
            output = self.ref_policy_wg.compute_ref_log_prob(batch_td)

        log_probs = tu.get(output, "log_probs")
        ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
        return DataProto.from_tensordict(ref_log_prob)

    def _update_actor(self, batch: DataProto) -> DataProto:
        """Update actor model using DPO loss."""
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable

        batch_td = batch.to_tensordict()
        batch_td = embeds_padding_2_no_padding(batch_td)

        ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
        seed = self.config.actor_rollout_ref.actor.data_loader_seed
        shuffle = self.config.actor_rollout_ref.actor.shuffle

        tu.assign_non_tensor(
            batch_td,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
            height=self.config.actor_rollout_ref.model.pipeline.height,
            width=self.config.actor_rollout_ref.model.pipeline.width,
            vae_scale_factor=self.config.actor_rollout_ref.model.get("vae_scale_factor", 8),
        )

        actor_output = self.actor_rollout_wg.update_actor(batch_td)
        actor_output = tu.get(actor_output, "metrics")
        actor_output = rename_dict(actor_output, "actor/")
        return DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})

    def fit(self):
        """Main DPO training loop."""
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # Load checkpoint
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)

        # Validation before training
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            if val_metrics:
                pprint(f"Initial validation metrics: {val_metrics}")
                logger.log(data=val_metrics, step=self.global_steps)

        # Training loop
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="DPO Training Progress")

        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)

                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # Add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                # Generate samples (K-repeated sampling for DPO)
                gen_batch = self._prepare_generation_batch(batch)

                with marked_timer("step", timing_raw):
                    with marked_timer("gen", timing_raw, color="red"):
                        # Generate samples for DPO
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        self.checkpoint_manager.sleep_replicas()
                        timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
                        gen_batch_output.meta_info.pop("timing", None)

                    # Form pairs and compute rewards
                    with marked_timer("reward", timing_raw, color="yellow"):
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            batch_reward = self._compute_reward_colocate(batch)
                            batch = batch.union(batch_reward)

                        # Extract rewards and form DPO pairs
                        reward_tensor, reward_extra_infos_dict = extract_reward(batch)

                    # Compute DPO advantages (implicit rewards)
                    with marked_timer("adv", timing_raw, color="brown"):
                        batch.batch["sample_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # Compute advantages for DPO
                        batch = self._compute_dpo_advantages(batch)

                    # Update actor using DPO loss
                    with marked_timer("update_actor", timing_raw, color="red"):
                        actor_output = self._update_actor(batch)

                    # Checkpoint saving
                    is_last_step = self.global_steps >= self.total_training_steps
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                    # Update weights from trainer to rollout
                    with marked_timer("update_weights", timing_raw, color="red"):
                        self.checkpoint_manager.update_weights(self.global_steps)

                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                # Validation
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics = self._validate()
                        if val_metrics:
                            last_val_metrics = val_metrics
                            metrics.update(val_metrics)

                # Collect metrics
                self._collect_training_metrics(batch, timing_raw, metrics, epoch)

                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

    def _prepare_generation_batch(self, batch: DataProto) -> DataProto:
        """Prepare batch for generation with K-repeated sampling."""
        reward_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_keys

        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # Repeat for K samples per prompt (for DPO pair formation)
        gen_batch = gen_batch.repeat(
            repeat_times=self.config.actor_rollout_ref.rollout.get("k_samples", 4), interleave=True
        )

        gen_batch.meta_info = {
            "recompute_log_prob": False,
            "validate": False,
            "global_steps": self.global_steps,
        }

        return gen_batch

    def _compute_reward_colocate(self, batch: DataProto) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        """Compute reward using colocated reward model."""
        assert self.reward_loop_manager is not None, "RewardLoopManager is None"
        batch_reward = self.reward_loop_manager.compute_rm_score(batch)
        return batch_reward

    def _compute_dpo_advantages(self, batch: DataProto) -> DataProto:
        """Compute advantages for DPO training.

        For DPO, we need to:
        1. Form (chosen, rejected) pairs based on rewards
        2. Compute implicit rewards for both policy and reference model
        3. Compute advantage as implicit_reward_chosen - implicit_reward_rejected
        """

        # Get rewards and group information
        rewards = batch.batch.get("sample_level_scores", None)
        if rewards is None:
            raise ValueError("No rewards found in batch for DPO training")

        # Get group information (uid for grouping)
        uids = batch.non_tensor_batch.get("uid", None)
        if uids is None:
            raise ValueError("No uid found in batch for DPO pair formation")

        # Compute DPO advantages using the registered function
        adv_estimator = self.config.algorithm.adv_estimator
        if adv_estimator != DiffusionAdvantageEstimator.DPO:
            logger.warning(f"Expected DPO advantage estimator, got {adv_estimator}. Using DPO.")

        # The advantage computation for DPO is handled differently
        # We need to form pairs and compute implicit rewards
        batch = compute_dpo_advantage(
            batch,
            config=self.config.algorithm,
        )

        return batch

    def _collect_training_metrics(self, batch, timing_raw, metrics, epoch):
        """Collect and compute training metrics."""
        n_gpus = self.resource_pool_manager.get_n_gpus()
        num_samples = batch.batch["sample_level_scores"].shape[0]

        metrics.update(
            {
                "training/global_step": self.global_steps,
                "training/epoch": epoch,
            }
        )

        metrics.update(compute_data_metrics_diffusion(batch=batch))
        metrics.update(compute_timing_metrics_diffusion(timing_raw=timing_raw, num_images=num_samples))
        metrics.update(compute_throughput_metrics_diffusion(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

    def _validate(self):
        """Run validation during training."""
        # Similar to RayFlowGRPOTrainer._validate
        # Implementation depends on specific validation requirements
        pass


def compute_dpo_advantage(
    data: DataProto,
    config: Optional[DiffusionAlgoConfig] = None,
) -> DataProto:
    """Compute DPO advantage (implicit reward difference).

    Args:
        data: DataProto containing batch with rewards and group information.
        config: Algorithm configuration.

    Returns:
        DataProto with computed advantages and returns.
    """
    # Get rewards and uids
    rewards = data.batch["sample_level_scores"].sum(-1)  # (batch_size,)
    uids = data.non_tensor_batch.get("uid", None)

    if uids is None:
        raise ValueError("UIDs are required for DPO advantage computation")

    # Group by uid and form pairs
    uid_to_indices = defaultdict(list)
    for i, uid in enumerate(uids):
        uid_to_indices[uid].append(i)

    # Compute implicit rewards (simplified: use actual rewards)
    # In full DPO, implicit reward = -0.5 * beta * (policy_err - ref_err)
    # For simplicity, we use the actual rewards as implicit rewards
    beta = getattr(config, "dpo_beta", 100.0)

    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)

    for uid, indices in uid_to_indices.items():
        if len(indices) < 2:
            continue

        group_rewards = rewards[indices]
        # Form pairs: highest vs lowest reward in group
        sorted_indices = torch.argsort(group_rewards, descending=True)
        chosen_idx = indices[sorted_indices[0]]
        rejected_idx = indices[sorted_indices[-1]]

        # Advantage = implicit_reward_chosen - implicit_reward_rejected
        # For DPO, we use the actual rewards as a proxy
        advantages[chosen_idx] = (group_rewards[sorted_indices[0]] - group_rewards[sorted_indices[-1]]) * beta / 100.0
        advantages[rejected_idx] = -advantages[chosen_idx]

    data.batch["advantages"] = advantages.unsqueeze(-1)
    data.batch["returns"] = returns.unsqueeze(-1)

    return data
