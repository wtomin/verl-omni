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
"""FSDP/FSDP2 engine for Qwen3-Omni offline DPO."""

import gc
import logging
import os
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Optional

import torch
import torch.distributed
import torch.nn.functional as F
from tensordict import TensorDict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.tensor import DTensor
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    FSDPModule,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_clip_grad_norm_,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.model import convert_weight_keys
from verl.utils.torch_dtypes import PrecisionType
from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig
from verl.workers.engine.base import BaseEngine, BaseEngineCtx, EngineRegistry
from verl.workers.engine.fsdp.utils import create_device_mesh, get_sharding_strategy
from verl.workers.engine.utils import enable_full_determinism, prepare_micro_batches

from verl_omni.pipelines.utils import prepare_omni_model_inputs
from verl_omni.utils.fsdp_utils import collect_lora_params
from verl_omni.workers.config import OmniModelConfig
from verl_omni.workers.engine.lora_adapter_mixin import LoRAAdapterMixin

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
device_name = get_device_name()

_NON_MODEL_KEYS = {
    "average_log_prob",
    "compute_loss",
    "disable_auto_offload",
    "global_token_num",
    "gradient_accumulation_steps",
    "max_token_len_per_gpu",
    "micro_batch_size_per_gpu",
    "mini_batch_size",
    "num_mini_batch",
    "reference_chosen_logps",
    "reference_rejected_logps",
    "sample_level_rewards",
    "sample_level_scores",
    "sp_size",
    "update_lr_scheduler",
    "use_dynamic_bsz",
    "use_fused_kernels",
    "use_remove_padding",
}


@EngineRegistry.register(model_type="omni_model", backend=["fsdp", "fsdp2"], device=["cuda"])
class OmniFSDPEngine(LoRAAdapterMixin, BaseEngine):
    """FSDP/FSDP2 Qwen3-Omni engine for offline DPO and LoRA training."""

    def __init__(
        self,
        model_config: OmniModelConfig,
        engine_config: FSDPEngineConfig,
        optimizer_config: FSDPOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__()
        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config
        self.mode = None
        self.rank = torch.distributed.get_rank()
        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload
        self._is_lora = self.model_config.lora_rank > 0 or self.model_config.lora_adapter_path is not None
        self._init_device_mesh()
        if self.engine_config.full_determinism:
            enable_full_determinism(seed=self.engine_config.seed)

    @property
    def is_param_offload_enabled(self) -> bool:
        return self._is_offload_param

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return self._is_offload_optimizer

    def _init_device_mesh(self):
        world_size = torch.distributed.get_world_size()
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=self.engine_config.fsdp_size)
        self.ulysses_sequence_parallel_size = 1
        self.ulysses_device_mesh = None
        self.use_ulysses_sp = False

    def initialize(self):
        self._build_model_optimizer()
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.module,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            processing_class=self.model_config.get_processor(),
            checkpoint_config=self.checkpoint_config,
            trust_remote_code=self.model_config.trust_remote_code,
        )
        self.to(
            device="cpu",
            model=self._is_offload_param,
            optimizer=self._is_offload_optimizer,
            grad=self._is_offload_param,
        )
        log_gpu_memory_usage("After offload model/optimizer/grad during init", logger=logger)

    def _strip_modules_for_thinker_only(self, module: torch.nn.Module) -> None:
        for name in getattr(module, "_verl_strip_modules", ()):
            if hasattr(module, name):
                setattr(module, name, None)

    def _build_module(self):
        from transformers import AutoConfig, AutoModelForCausalLM
        from verl.utils.model import print_model_size

        torch_dtype = self.engine_config.model_dtype
        if torch_dtype is None:
            torch_dtype = torch.float32 if not self.engine_config.forward_only else torch.bfloat16
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        init_context = get_init_weight_context_manager(use_meta_tensor=True, mesh=self.device_mesh)
        with init_context():
            hf_config = AutoConfig.from_pretrained(
                self.model_config.config_path or self.model_config.local_path,
                trust_remote_code=self.model_config.trust_remote_code,
            )
            module = AutoModelForCausalLM.from_pretrained(
                self.model_config.model_path or self.model_config.local_path,
                torch_dtype=torch_dtype,
                trust_remote_code=self.model_config.trust_remote_code,
                config=hf_config,
            )
            module.to(torch_dtype)
            self._strip_modules_for_thinker_only(module)
            if self.model_config.enable_gradient_checkpointing:
                if hasattr(module, "gradient_checkpointing_enable"):
                    try:
                        module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                    except TypeError:
                        module.gradient_checkpointing_enable()
                elif hasattr(module, "enable_gradient_checkpointing"):
                    module.enable_gradient_checkpointing()
            module.config.use_cache = False
            module.can_generate = lambda: False

        if self.rank == 0:
            print_model_size(module)
        return module

    def _build_fsdp_module(self, module: torch.nn.Module):
        from torch.distributed.fsdp import CPUOffload, MixedPrecision

        mixed_precision_config = self.engine_config.mixed_precision
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)
        auto_wrap_policy = get_fsdp_wrap_policy(
            module=module,
            config=self.engine_config.wrap_policy,
            is_lora=self._is_lora,
        )
        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        if self.engine_config.strategy == "fsdp":
            cpu_offload = None
            if self.engine_config.forward_only:
                cpu_offload = CPUOffload(offload_params=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False
            module = FSDP(
                module,
                param_init_fn=init_fn,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=self.engine_config.forward_prefetch,
                use_orig_params=self.engine_config.use_orig_params,
                cpu_offload=cpu_offload,
            )
        elif self.engine_config.strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using FSDP2"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype,
                reduce_dtype=reduce_dtype,
                cast_forward_inputs=True,
            )
            offload_policy = None
            if self.engine_config.offload_policy or self.engine_config.forward_only:
                self._is_offload_param = False
                self._is_offload_optimizer = False
                offload_policy = CPUOffloadPolicy(pin_memory=True)
            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": offload_policy,
                "reshard_after_forward": self.engine_config.reshard_after_forward,
            }
            full_state = module.state_dict()
            apply_fsdp2(module, fsdp_kwargs, self.engine_config)
            fsdp2_load_full_state_dict(module, full_state, fsdp_mesh, offload_policy)
        else:
            raise NotImplementedError(f"Unknown strategy {self.engine_config.strategy}")

        if torch.distributed.get_world_size() == 1 and fsdp_version(module) == 1:
            FSDP.set_state_dict_type(
                module,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(),
            )
        elif fsdp_version(module) == 1:
            FSDP.set_state_dict_type(
                module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )
        return module

    def _build_optimizer(self, module):
        from verl.workers.config.optimizer import build_optimizer

        return build_optimizer(module.parameters(), self.optimizer_config)

    def _build_lr_scheduler(self, optimizer):
        from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

        optim_config = self.optimizer_config
        total_steps = optim_config.total_training_steps
        num_warmup_steps = optim_config.lr_warmup_steps
        if num_warmup_steps <= 0:
            num_warmup_steps = int(optim_config.lr_warmup_steps_ratio * total_steps)
        if self.rank == 0:
            print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")
        if optim_config.lr_scheduler_type == "constant":
            return get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps)
        if optim_config.lr_scheduler_type == "cosine":
            return get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                min_lr_ratio=optim_config.min_lr_ratio,
                num_cycles=optim_config.num_cycles,
                zero_indexed_step=optim_config.zero_indexed_step,
            )
        raise NotImplementedError(f"LR scheduler type {optim_config.lr_scheduler_type} is not supported")

    def _build_model_optimizer(self):
        module = self._build_module()
        if self._is_lora:
            module = self._build_lora_module(module)
        torch.distributed.barrier()
        log_gpu_memory_usage("After init Qwen3-Omni model", logger=logger)
        module = self._build_fsdp_module(module)
        log_gpu_memory_usage("After Qwen3-Omni FSDP", logger=logger)

        if self.engine_config.forward_only:
            optimizer = None
            lr_scheduler = None
        else:
            optimizer = self._build_optimizer(module)
            lr_scheduler = self._build_lr_scheduler(optimizer)

        self.module = module
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def train_mode(self, **kwargs):
        return EngineTrainModeCtx(self, **kwargs)

    def eval_mode(self, **kwargs):
        return EngineEvalModeCtx(self, **kwargs)

    def get_data_parallel_rank(self):
        return torch.distributed.get_rank()

    def get_data_parallel_size(self):
        return torch.distributed.get_world_size()

    def get_data_parallel_group(self):
        return torch.distributed.group.WORLD

    def is_mp_src_rank_with_outputs(self):
        return True

    def _model_module(self):
        return getattr(self.module, "_fsdp_wrapped_module", self.module)

    @contextmanager
    def disable_adapter(self):
        module = self._model_module()
        if not hasattr(module, "disable_adapters"):
            yield
            return
        module.disable_adapters()
        try:
            yield
        finally:
            module.enable_adapters()

    def _prepare_model_inputs(self, micro_batch: TensorDict | dict[str, Any]) -> tuple[dict[str, Any], torch.Tensor]:
        labels = micro_batch["labels"]
        model_inputs = {key: value for key, value in micro_batch.items() if key not in _NON_MODEL_KEYS}
        model_inputs.pop("labels", None)
        model_inputs = prepare_omni_model_inputs(
            self.model_config,
            model_inputs,
            dtype=next(self.module.parameters()).dtype,
        )
        return model_inputs, labels

    @staticmethod
    def _sequence_logps(logits: torch.Tensor, labels: torch.Tensor, average_log_prob: bool) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].float()
        shift_labels = labels[:, 1:].contiguous()
        loss_mask = shift_labels != -100
        safe_labels = shift_labels.masked_fill(~loss_mask, 0)
        token_logps = F.log_softmax(shift_logits, dim=-1).gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
        seq_logps = (token_logps * loss_mask).sum(dim=-1)
        if average_log_prob:
            seq_logps = seq_logps / loss_mask.sum(dim=-1).clamp(min=1)
        return seq_logps

    def _concatenated_forward(self, model, micro_batch: TensorDict | dict[str, Any]):
        model_inputs, labels = self._prepare_model_inputs(micro_batch)
        outputs = model(**model_inputs, use_cache=False)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        all_logps = self._sequence_logps(
            logits,
            labels,
            average_log_prob=tu.get_non_tensor_data(micro_batch, "average_log_prob", default=False),
        )
        return all_logps[0::2], all_logps[1::2]

    def _forward_backward_micro_batch(self, micro_batch: TensorDict, loss_function: Callable):
        micro_batch = micro_batch.to(get_device_id())
        tu.assign_non_tensor(
            micro_batch,
            average_log_prob=tu.get_non_tensor_data(micro_batch, "average_log_prob", default=False),
        )
        policy_chosen_logps, policy_rejected_logps = self._concatenated_forward(self.module, micro_batch)
        model_output = {
            "policy_chosen_logps": policy_chosen_logps,
            "policy_rejected_logps": policy_rejected_logps,
            "reference_chosen_logps": micro_batch["reference_chosen_logps"],
            "reference_rejected_logps": micro_batch["reference_rejected_logps"],
        }
        loss, metrics = loss_function(model_output=model_output, data=micro_batch)
        loss.backward()
        return loss.detach(), metrics

    def train_batch(self, data: TensorDict, loss_function: Optional[Callable] = None):
        if loss_function is None:
            raise ValueError("OmniFSDPEngine.train_batch requires a loss_function.")
        config = getattr(loss_function, "keywords", {}).get("config")
        loss_config = getattr(config, "omni_loss", None)
        tu.assign_non_tensor(data, average_log_prob=getattr(loss_config, "average_log_prob", False))
        micro_batches, _ = prepare_micro_batches(
            data=data,
            dp_group=self.get_data_parallel_group(),
            same_micro_num_in_dp=True,
        )
        gradient_accumulation_steps = len(micro_batches)
        losses = []
        metrics: dict[str, list[torch.Tensor]] = {}
        for micro_batch in micro_batches:
            tu.assign_non_tensor(micro_batch, gradient_accumulation_steps=gradient_accumulation_steps)
            loss, micro_metrics = self._forward_backward_micro_batch(micro_batch, loss_function)
            losses.append(loss.item())
            for key, value in micro_metrics.items():
                metrics.setdefault(key, []).append(value)
        grad_norm = self.optimizer_step()
        self.optimizer_zero_grad()
        metrics = {key: torch.stack(value).mean().item() for key, value in metrics.items()}
        metrics["grad_norm"] = grad_norm
        return {"model_output": {}, "loss": losses, "metrics": metrics}

    def infer_batch(self, data: TensorDict, loss_function: Optional[Callable] = None):
        del loss_function
        micro_batches, _ = prepare_micro_batches(
            data=data,
            dp_group=self.get_data_parallel_group(),
            same_micro_num_in_dp=True,
        )
        chosen_logps = []
        rejected_logps = []
        with torch.no_grad():
            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                chosen, rejected = self._concatenated_forward(self.module, micro_batch)
                chosen_logps.append(chosen)
                rejected_logps.append(rejected)
        return {
            "model_output": {
                "chosen_logps": torch.cat(chosen_logps, dim=0),
                "rejected_logps": torch.cat(rejected_logps, dim=0),
            },
            "loss": [0.0],
            "metrics": {},
        }

    def optimizer_zero_grad(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad()

    def optimizer_step(self):
        assert self.optimizer_config.clip_grad is not None
        if isinstance(self.module, FSDP):
            grad_norm = self.module.clip_grad_norm_(self.optimizer_config.clip_grad)
        elif isinstance(self.module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.module.parameters(), max_norm=self.optimizer_config.clip_grad)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.module.parameters(),
                max_norm=self.optimizer_config.clip_grad,
            )
        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()
        if not torch.isfinite(grad_norm):
            logger.warning("grad_norm is not finite: %s", grad_norm)
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()
        return grad_norm.item()

    def lr_scheduler_step(self):
        if self.lr_scheduler is None:
            return None
        self.lr_scheduler.step()
        return self.lr_scheduler.get_last_lr()[0]

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        super().to(device=device, model=model, optimizer=optimizer, grad=grad)
        if self.engine_config.forward_only:
            return
        assert device in (device_name, "cpu")
        if device == device_name:
            if model:
                load_fsdp_model_to_gpu(self.module)
            if optimizer and self.optimizer is not None:
                load_fsdp_optimizer(self.optimizer, device)
            gc.collect()
        elif device == "cpu":
            if model:
                offload_fsdp_model_to_cpu(self.module)
            if optimizer and self.optimizer is not None:
                offload_fsdp_optimizer(self.optimizer)

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs,
    ) -> None:
        origin_module_device = next(self.module.parameters()).device.type
        if self._is_offload_param or origin_module_device == "cpu":
            load_fsdp_model_to_gpu(self.module)
        self.checkpoint_manager.save_checkpoint(
            local_path=local_path,
            hdfs_path=hdfs_path,
            global_step=global_step,
            max_ckpt_to_keep=max_ckpt_to_keep,
        )
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)
        gc.collect()
        aggressive_empty_cache(force_sync=True)

    def load_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        del_local_after_load: int = True,
        **kwargs,
    ) -> None:
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.module)
        self.checkpoint_manager.load_checkpoint(
            local_path=local_path,
            hdfs_path=hdfs_path,
            del_local_after_load=del_local_after_load,
        )
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.optimizer)

    def get_per_tensor_param(
        self,
        layered_summon=False,
        base_sync_done=False,
        adapter_name: str | None = None,
        **kwargs,
    ):
        load_fsdp_model_to_gpu(self.module)
        peft_config = None
        peft_model = self._model_module()
        if hasattr(peft_model, "peft_config"):
            peft_config = peft_model.peft_config.get("default", None)
            adapter_ctx = self.use_adapter(adapter_name) if adapter_name is not None else nullcontext()
            with adapter_ctx:
                params = collect_lora_params(
                    module=self.module,
                    layered_summon=layered_summon,
                    base_sync_done=base_sync_done,
                    is_diffusers=False,
                    adapter_name=adapter_name or "default",
                    layer_prefixes=self.model_config.fsdp_layer_prefixes,
                )
        else:
            params = self.module.state_dict()
        params = convert_weight_keys(params, peft_model)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)
        device = get_device_id()
        export_dtype = PrecisionType.to_dtype(self.engine_config.model_dtype)

        def param_generator():
            for name, param in params.items():
                tensor = param.full_tensor() if isinstance(param, DTensor) else param
                tensor = tensor.to(device, non_blocking=True)
                if tensor.is_floating_point() and export_dtype is not None and tensor.dtype != export_dtype:
                    tensor = tensor.to(export_dtype, non_blocking=True)
                yield name, tensor

        peft_config_dict = peft_config.to_dict() if peft_config is not None else None
        return param_generator(), peft_config_dict


class EngineEvalModeCtx(BaseEngineCtx):
    def __init__(self, engine: OmniFSDPEngine, **kwargs):
        super().__init__(engine=engine, mode="eval", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, OmniFSDPEngine)
        super().__enter__()
        self.engine.module.eval()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, OmniFSDPEngine)
        if self.engine.engine_config.fsdp_size > 1:
            if fsdp_version(self.engine.module) == 1:
                self.engine.module._handle.reshard(True)
            elif fsdp_version(self.engine.module) == 2:
                self.engine.module.reshard()
        super().__exit__(exc_type, exc_value, traceback)


class EngineTrainModeCtx(BaseEngineCtx):
    def __init__(self, engine: OmniFSDPEngine, **kwargs):
        super().__init__(engine=engine, mode="train", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, OmniFSDPEngine)
        super().__enter__()
        self.engine.module.train()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, OmniFSDPEngine)
        self.engine.optimizer_zero_grad()
        super().__exit__(exc_type, exc_value, traceback)
