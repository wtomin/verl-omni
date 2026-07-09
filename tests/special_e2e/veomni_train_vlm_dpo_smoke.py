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
# pyright: reportMissingImports=false
"""VeOmni-native Qwen3-Omni VLM DPO smoke task.

This mirrors VeOmni's ``tasks/train_vlm.py`` entrypoint, but swaps the VLM
forward/backward step for a minimal DPO objective adapted from VeOmni's text DPO
trainer. It is intentionally local to the smoke test so verl-omni can exercise
Qwen3-Omni preference data through VeOmni's VLMTrainer stack without vendoring
VeOmni. Expects Omni-Preference-format parquet (image/video/audio).
"""

from __future__ import annotations

import json
import sys
import types
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

if "transformers.initialization" not in sys.modules:
    try:
        from transformers.initialization import no_init_weights
    except ImportError:
        from transformers.modeling_utils import no_init_weights

        initialization = types.ModuleType("transformers.initialization")
        initialization.no_init_weights = no_init_weights
        sys.modules["transformers.initialization"] = initialization

try:
    import transformers.integrations as _transformers_integrations

    if not hasattr(_transformers_integrations, "use_kernel_func_from_hub") and hasattr(
        _transformers_integrations, "use_kernel_forward_from_hub"
    ):
        _transformers_integrations.use_kernel_func_from_hub = _transformers_integrations.use_kernel_forward_from_hub
    if not hasattr(_transformers_integrations, "use_kernelized_func"):
        _transformers_integrations.use_kernelized_func = lambda kernel_func: (lambda wrapped: wrapped)
except ImportError:
    pass

try:
    import transformers.utils.generic as _transformers_generic

    if not hasattr(_transformers_generic, "is_flash_attention_requested"):
        _transformers_generic.is_flash_attention_requested = lambda config: False
    if not hasattr(_transformers_generic, "maybe_autocast"):
        _transformers_generic.maybe_autocast = lambda *args, **kwargs: nullcontext()
    if not hasattr(_transformers_generic, "merge_with_config_defaults"):
        _transformers_generic.merge_with_config_defaults = lambda func: func
except ImportError:
    pass

if "transformers.utils.output_capturing" not in sys.modules:
    try:
        import transformers.utils.output_capturing  # noqa: F401
    except ImportError:
        output_capturing = types.ModuleType("transformers.utils.output_capturing")

        class OutputRecorder:
            def __init__(self, *args, **kwargs):
                pass

        def capture_outputs(func=None, **kwargs):
            if func is None:
                return lambda wrapped: wrapped
            return func

        output_capturing._CAN_RECORD_REGISTRY = {}
        output_capturing.OutputRecorder = OutputRecorder
        output_capturing.capture_outputs = capture_outputs
        output_capturing.maybe_install_capturing_hooks = lambda model: None
        sys.modules["transformers.utils.output_capturing"] = output_capturing

try:
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeTextConfig,
        Qwen3OmniMoeThinkerConfig,
    )

    if not hasattr(Qwen3OmniMoeTextConfig, "rope_parameters"):

        def _rope_parameters(self):
            rope_parameters = dict(getattr(self, "rope_scaling", None) or {"rope_type": "default"})
            rope_parameters.setdefault("rope_theta", getattr(self, "rope_theta", 10000.0))
            return rope_parameters

        Qwen3OmniMoeTextConfig.rope_parameters = property(_rope_parameters)
    _THINKER_TOKEN_DEFAULTS = {"vision_start_token_id": 6}
    for _name, _value in _THINKER_TOKEN_DEFAULTS.items():
        if not hasattr(Qwen3OmniMoeThinkerConfig, _name):
            setattr(Qwen3OmniMoeThinkerConfig, _name, property(lambda self, value=_value: value))
except ImportError:
    pass

try:
    import transformers.utils as _transformers_utils

    _transformers_utils.auto_docstring = lambda obj=None, **kwargs: (lambda wrapped: wrapped) if obj is None else obj
    if not hasattr(_transformers_utils, "is_grouped_mm_available"):
        _transformers_utils.is_grouped_mm_available = lambda: False
    if not hasattr(_transformers_utils, "torch_compilable_check"):

        def _torch_compilable_check(condition, message):
            if not condition:
                raise ValueError(message)

        _transformers_utils.torch_compilable_check = _torch_compilable_check
except ImportError:
    pass

try:
    from veomni.arguments import MixedPrecisionConfig, parse_args
    from veomni.data import build_data_transform
    from veomni.data.data_collator import PostCollator
    from veomni.data.data_transform import DATA_TRANSFORM_REGISTRY
    from veomni.data.multimodal import PREPROCESSOR_REGISTRY
    from veomni.distributed.clip_grad_norm import veomni_clip_grad_norm
    from veomni.distributed.parallel_state import get_parallel_state
    from veomni.distributed.sequence_parallel import gather_outputs
    from veomni.distributed.torch_parallelize import build_parallelize_model
    from veomni.models import build_foundation_model
    from veomni.ops.batch_invariant_ops import set_batch_invariant_mode
    from veomni.trainer.base import BaseTrainer
    from veomni.trainer.text_dpo_trainer import DPOConfig
    from veomni.trainer.vlm_trainer import VeOmniVLMArguments, VLMTrainer
    from veomni.utils import helper, logging
    from veomni.utils.constants import IGNORE_INDEX
    from veomni.utils.device import synchronize
except ImportError as exc:  # pragma: no cover - exercised only when optional dep is absent.
    raise SystemExit(
        f"VeOmni and its compatible runtime dependencies are required for this smoke test. Import failed with: {exc}"
    ) from exc


logger = logging.get_logger(__name__)

_NON_MODEL_KEYS = set()
_SOURCE_NAMES = (
    "Omni-Preference-Image",
    "Omni-Preference-Video",
    "Omni-Preference-Audio",
)


@dataclass
class VeOmniVLMDPOArguments(VeOmniVLMArguments):
    """Root args for Qwen3-Omni VLM DPO smoke training."""

    dpo_config: DPOConfig = field(default_factory=DPOConfig)


def _register_pass_through_preprocessors() -> None:
    def _pass_through(conversations, **kwargs):
        return conversations

    for source_name in _SOURCE_NAMES:
        try:
            PREPROCESSOR_REGISTRY[source_name]
        except (KeyError, ValueError):
            PREPROCESSOR_REGISTRY.register(source_name)(_pass_through)


def _as_python(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") or text.startswith("{"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
    return value


def _append_content(conversation: list[Any], content: Any, media: dict[str, list[Any]]) -> None:
    content = _as_python(content)
    if isinstance(content, str):
        conversation.append(("text", content))
        return

    for item in content or []:
        item = _as_python(item)
        if not isinstance(item, dict):
            conversation.append(("text", str(item)))
            continue

        item_type = item.get("type")
        if item_type == "text":
            conversation.append(("text", item.get("text", "")))
        elif item_type == "image":
            media["images"].append(item.get("image"))
            conversation.append(("image", None))
        elif item_type == "video":
            media["videos"].append(item.get("video"))
            conversation.append(("video", None))
        elif item_type == "audio":
            media["audios"].append(item.get("audio"))
            conversation.append(("audio", None))


def _build_preference_branch(sample: dict[str, Any], answer: str) -> dict[str, Any]:
    prompt = _as_python(sample.get("prompt", []))
    media: dict[str, list[Any]] = {"images": [], "videos": [], "audios": []}
    conversations: list[list[Any]] = []

    for message in prompt:
        message = _as_python(message)
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "system":
            continue
        conversation = [role or "user"]
        _append_content(conversation, message.get("content", ""), media)
        if len(conversation) > 1:
            conversations.append(conversation)

    conversations.append(["assistant", ("text", answer)])

    branch = {
        "conversations": conversations,
        "source_name": sample.get("source_name"),
    }
    for key, values in media.items():
        if values:
            branch[key] = values
    return branch


def _cat_sequence_tensors(chosen: torch.Tensor, rejected: torch.Tensor) -> torch.Tensor:
    dim = -1 if chosen.ndim > 1 else 0
    return torch.cat([chosen, rejected], dim=dim)


def _merge_chosen_rejected(chosen: dict[str, Any], rejected: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in chosen.keys() | rejected.keys():
        chosen_value = chosen.get(key)
        rejected_value = rejected.get(key)
        if chosen_value is None:
            merged[key] = rejected_value
            continue
        if rejected_value is None:
            merged[key] = chosen_value
            continue
        if not isinstance(chosen_value, torch.Tensor) or not isinstance(rejected_value, torch.Tensor):
            merged[key] = chosen_value
            continue
        if key in {"input_ids", "attention_mask", "labels", "position_ids", "image_mask", "video_mask", "audio_mask"}:
            merged[key] = _cat_sequence_tensors(chosen_value, rejected_value)
        else:
            merged[key] = torch.cat([chosen_value, rejected_value], dim=0)
    return merged


def _mixed_precision_forward_context(args: VeOmniVLMDPOArguments, device_type: str):
    mixed_precision = args.train.accelerator.fsdp_config.mixed_precision
    if not mixed_precision.enable:
        return nullcontext()
    return torch.autocast(device_type=device_type, dtype=getattr(torch, mixed_precision.param_dtype))


def _register_qwen_omni_dpo_transform() -> None:
    try:
        DATA_TRANSFORM_REGISTRY["qwen3_omni_moe_dpo"]
        return
    except (KeyError, ValueError):
        pass

    @DATA_TRANSFORM_REGISTRY.register("qwen3_omni_moe_dpo")
    def process_qwen_omni_dpo(sample: dict[str, Any], **kwargs):
        base_transform = DATA_TRANSFORM_REGISTRY["qwen3_omni_moe"]
        chosen_sample = _build_preference_branch(sample, sample["chosen"])
        rejected_sample = _build_preference_branch(sample, sample["rejected"])
        chosen = base_transform(chosen_sample, **kwargs)[0]
        rejected = base_transform(rejected_sample, **kwargs)[0]
        return [_merge_chosen_rejected(chosen, rejected)]


class VLMDPOTrainer(VLMTrainer):
    """Qwen3-Omni VLMTrainer variant with VeOmni DPO loss."""

    def __init__(self, args: VeOmniVLMDPOArguments):
        self.base = BaseTrainer.__new__(BaseTrainer)
        self.base.args = args

        self.base._setup()
        self._build_model()
        self._freeze_model_module()
        self._build_model_assets()
        self._build_data_transform()
        self.base._build_dataset()
        self._build_collate_fn()
        self.base._build_dataloader()
        self._build_postforward()
        self.base._build_parallelized_model()
        self.base._build_optimizer()
        self.base._build_lr_scheduler()
        self.base._build_training_context()
        self.base._init_callbacks()
        self._build_reference_model()

    def _build_data_transform(self):
        args: VeOmniVLMDPOArguments = self.base.args
        model_type = self.base.model_config.model_type
        if model_type != "qwen3_omni_moe":
            raise ValueError(f"VLM DPO smoke currently supports qwen3_omni_moe only, got {model_type!r}")
        self.base.data_transform = build_data_transform(
            "qwen3_omni_moe_dpo",
            processor=self.base.processor,
            position_id_func=self.base.model.get_position_id_func(),
            **args.data.mm_configs,
        )

    def _build_postforward(self):
        self.post_forward = PostCollator()
        self.sp_enabled = get_parallel_state().sp_enabled

    def _build_reference_model(self):
        args: VeOmniVLMDPOArguments = self.base.args
        logger.info_rank0("Building frozen Qwen3-Omni reference model for VLM DPO")
        self.reference_model = build_foundation_model(
            config_path=args.model.config_path,
            weights_path=args.model.model_path,
            torch_dtype=args.dpo_config.refer_model_precision,
            init_device=args.train.init_device,
            encoder_data_balance=args.model.encoder_data_balance,
            encoder_data_balance_sorting_algo=args.model.encoder_data_balance_sorting_algo,
            ops_implementation=args.model.ops_implementation,
            config_kwargs=args.model.model_config,
        )
        self.reference_model.requires_grad_(False)

        cpu_load_param_name = None
        if hasattr(self.base.model, "get_parallel_plan"):
            cpu_load_param_name = getattr(self.base.model.get_parallel_plan(), "cpu_load_param_name", None)

        self.reference_model = build_parallelize_model(
            self.reference_model,
            init_device=args.train.init_device,
            weights_path=args.model.model_path,
            enable_reshard_after_forward=args.train.accelerator.fsdp_config.reshard_after_forward,
            mixed_precision=MixedPrecisionConfig(enable=False),
            enable_gradient_checkpointing=False,
            basic_modules=list(
                set(getattr(self.reference_model, "_no_split_modules", None) or []) | set(args.model.basic_modules)
            ),
            enable_reentrant=False,
            enable_forward_prefetch=args.train.accelerator.fsdp_config.forward_prefetch,
            enable_fsdp_offload=args.train.accelerator.fsdp_config.offload,
            broadcast_model_weights_from_rank0=args.train.broadcast_model_weights_from_rank0,
            cpu_load_param_name=cpu_load_param_name,
            max_load_broadcast_size=args.train.accelerator.fsdp_config.max_load_broadcast_size,
        )
        self.reference_model.eval()
        helper.print_device_mem_info("VRAM usage after building VLM DPO reference model")

    @staticmethod
    def dpo_loss(
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
        beta: float,
        label_smoothing: float = 0.0,
        loss_type: str = "sigmoid",
        reference_free: bool = False,
    ):
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        if reference_free:
            ref_logratios = 0
        logits = pi_logratios - ref_logratios
        if loss_type == "ipo":
            losses = (logits - 1 / (2 * beta)) ** 2
        else:
            losses = (
                -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
            )
        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
        return losses, chosen_rewards, rejected_rewards

    def concatenated_forward(self, model, micro_batch: dict[str, Any]):
        model_inputs = {key: value for key, value in micro_batch.items() if key not in _NON_MODEL_KEYS}
        outputs = model(**model_inputs, return_log_probs=True, use_cache=False)
        log_probs_packed = outputs.fused_linear_aux.log_probs.squeeze(0)
        seq_lens = self.post_forward.compute_seqlens_func(micro_batch)
        if self.sp_enabled:
            log_probs_packed = gather_outputs(log_probs_packed, gather_dim=0, group=get_parallel_state().sp_group)
        log_probs_packed = log_probs_packed[: sum(seq_lens)]
        log_probs_list = list(log_probs_packed.split(seq_lens, dim=0))

        if self.sp_enabled:
            all_labels = gather_outputs(micro_batch["labels"], gather_dim=-1, group=get_parallel_state().sp_group)
            all_labels = all_labels.view(-1)[: sum(seq_lens)]
            labels_list = list(all_labels.split(seq_lens))
        else:
            all_labels = micro_batch["labels"].view(-1)
            labels_list = []
            offset = 0
            for seq_len in seq_lens:
                seq_labels = all_labels[offset : offset + seq_len]
                labels_list.append(F.pad(seq_labels[1:], (0, 1), value=IGNORE_INDEX))
                offset += seq_len

        all_logps = []
        for seq_log_probs, seq_labels in zip(log_probs_list, labels_list, strict=True):
            loss_mask = seq_labels != IGNORE_INDEX
            logp = (seq_log_probs.float() * loss_mask).sum()
            if self.base.args.dpo_config.average_log_prob:
                logp = logp / loss_mask.sum().clamp(min=1)
            all_logps.append(logp)
        all_logps_t = torch.stack(all_logps)
        return all_logps_t[0::2], all_logps_t[1::2]

    def forward_backward_step(self, micro_batch: dict[str, torch.Tensor]):
        args: VeOmniVLMDPOArguments = self.base.args
        micro_batch = self.base.preforward(micro_batch)

        with torch.no_grad():
            ref_chosen_logps, ref_rejected_logps = self.concatenated_forward(self.reference_model, micro_batch)

        with (
            self.base.model_fwd_context,
            _mixed_precision_forward_context(args, self.base.device.type),
            set_batch_invariant_mode(args.train.enable_batch_invariant_mode),
        ):
            policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(self.base.model, micro_batch)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta=args.dpo_config.beta,
            label_smoothing=args.dpo_config.label_smoothing,
            loss_type=args.dpo_config.loss_type,
            reference_free=args.dpo_config.reference_free,
        )
        loss = losses.mean()
        loss_dict = {
            "dpo_loss": loss.detach(),
            "chosen_rewards": chosen_rewards.mean().detach(),
            "rejected_rewards": rejected_rewards.mean().detach(),
            "reward_accuracy": (chosen_rewards > rejected_rewards).float().mean().detach(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean().detach(),
        }

        with self.base.model_bwd_context, set_batch_invariant_mode(args.train.enable_batch_invariant_mode):
            loss.backward()
        return loss, loss_dict

    def train_step(self, data_iterator: Any):
        args: VeOmniVLMDPOArguments = self.base.args
        self.base.state.global_step += 1
        micro_batches = next(data_iterator)
        self.on_step_begin(micro_batches=micro_batches)
        synchronize()

        total_loss = 0.0
        total_loss_dict: dict[str, float] = defaultdict(float)
        num_micro_steps = len(micro_batches)
        for micro_step, micro_batch in enumerate(micro_batches):
            self.base.model_reshard(micro_step, num_micro_steps)
            loss, loss_dict = self.forward_backward_step(micro_batch)
            total_loss += loss.item()
            for key, value in loss_dict.items():
                total_loss_dict[key] += value.item()

        grad_norm = veomni_clip_grad_norm(self.base.model, args.train.optimizer.max_grad_norm)
        self.base.optimizer.step()
        self.base.lr_scheduler.step()
        self.base.optimizer.zero_grad()
        self.on_step_end(loss=total_loss, loss_dict=total_loss_dict, grad_norm=grad_norm)

    def train(self):
        args: VeOmniVLMDPOArguments = self.base.args
        self.on_train_begin()
        logger.info(
            f"Rank{args.train.local_rank} Start Qwen3-Omni VLM DPO smoke training. "
            f"Start step: {self.base.start_step}. Train steps: {args.train_steps}."
        )
        for epoch in range(self.base.start_epoch, args.train.num_train_epochs):
            if hasattr(self.base.train_dataloader, "set_epoch"):
                self.base.train_dataloader.set_epoch(epoch)
            self.base.state.epoch = epoch
            self.on_epoch_begin()
            data_iterator = iter(self.base.train_dataloader)
            for _ in range(self.base.start_step, args.train_steps):
                try:
                    self.train_step(data_iterator)
                except StopIteration:
                    logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.dataloader.drop_last}")
                    break
            self.on_epoch_end()
            self.base.start_step = 0
        self.on_train_end()
        synchronize()
        self.base.destroy_distributed()


if __name__ == "__main__":
    _register_pass_through_preprocessors()
    _register_qwen_omni_dpo_transform()
    parsed_args = parse_args(VeOmniVLMDPOArguments)
    trainer = VLMDPOTrainer(parsed_args)
    trainer.train()
