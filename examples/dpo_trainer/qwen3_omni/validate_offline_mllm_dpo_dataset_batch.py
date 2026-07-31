#!/usr/bin/env python3
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

"""Print one OfflineMLLMDPODataset batch for right and no-padding modes.

Example:
    python examples/dpo_trainer/qwen3_omni/validate_offline_mllm_dpo_dataset_batch.py \
        --model-path /path/to/Qwen3-Omni-30B-A3B-Instruct \
        --data-root /path/to/Omni-Preference/parquet_dpo \
        --split train \
        --batch-size 1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings(
    "ignore",
    message=r"Unrecognized keys in `rope_parameters`.*",
)

DEFAULT_MM_CONFIGS = {
    "scale_factor": 28,
    "image_min_pixels": 3136,
    "image_max_pixels": 602112,
    "video_min_pixels": 100352,
    "video_max_pixels": 602112,
    "max_ratio": 200,
    "min_frames": 4,
    "max_frames": 8,
    "frame_factor": 2,
    "sample_rate": 16000,
    "fps": 2.0,
    "use_audio_in_video": False,
}


def load_repo_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bootstrap_minimal_verl_omni_modules():
    """Load only the modules needed for dataset validation.

    Importing ``verl_omni`` normally pulls optional rollout dependencies such as
    vLLM-Omni. This script only needs the registry, Qwen3-Omni processor adapter,
    and offline DPO dataset, so load those modules directly by path.
    """

    root_mod = sys.modules.setdefault("verl_omni", types.ModuleType("verl_omni"))
    root_mod.__path__ = [str(REPO_ROOT / "verl_omni")]
    pipelines_mod = sys.modules.setdefault("verl_omni.pipelines", types.ModuleType("verl_omni.pipelines"))
    pipelines_mod.__path__ = [str(REPO_ROOT / "verl_omni" / "pipelines")]
    qwen3_mod = sys.modules.setdefault(
        "verl_omni.pipelines.qwen3_omni",
        types.ModuleType("verl_omni.pipelines.qwen3_omni"),
    )
    qwen3_mod.__path__ = [str(REPO_ROOT / "verl_omni" / "pipelines" / "qwen3_omni")]
    utils_mod = sys.modules.setdefault("verl_omni.utils", types.ModuleType("verl_omni.utils"))
    utils_mod.__path__ = [str(REPO_ROOT / "verl_omni" / "utils")]
    dataset_pkg = sys.modules.setdefault("verl_omni.utils.dataset", types.ModuleType("verl_omni.utils.dataset"))
    dataset_pkg.__path__ = [str(REPO_ROOT / "verl_omni" / "utils" / "dataset")]

    model_base = load_repo_module(
        "verl_omni.pipelines.model_base",
        REPO_ROOT / "verl_omni" / "pipelines" / "model_base.py",
    )
    load_repo_module(
        "verl_omni.pipelines.qwen3_omni.thinker_training_adapter",
        REPO_ROOT / "verl_omni" / "pipelines" / "qwen3_omni" / "thinker_training_adapter.py",
    )
    load_repo_module(
        "verl_omni.utils.dataset.qwen3_omni_transform",
        REPO_ROOT / "verl_omni" / "utils" / "dataset" / "qwen3_omni_transform.py",
    )
    dataset_module = load_repo_module(
        "verl_omni.utils.dataset.offline_mllm_dpo_dataset",
        REPO_ROOT / "verl_omni" / "utils" / "dataset" / "offline_mllm_dpo_dataset.py",
    )
    return model_base.OmniModelBase, dataset_module


OmniModelBase, dataset_mod = bootstrap_minimal_verl_omni_modules()
OfflineMLLMDPODataset = dataset_mod.OfflineMLLMDPODataset
ModalityGroupedBatchSampler = dataset_mod.ModalityGroupedBatchSampler
offline_mllm_dpo_collate_fn = dataset_mod.offline_mllm_dpo_collate_fn


@dataclass
class ProcessorModelConfig:
    path: str
    architecture: str
    model_stage: str = "thinker"
    trust_remote_code: bool = True
    external_lib: str | None = None

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Qwen3-Omni model path or HF repo id.")
    parser.add_argument("--data-root", default=None, help="Root containing audio/image/video split parquet files.")
    parser.add_argument(
        "--split",
        default="train",
        help="Split name under each modality directory, e.g. train or test.",
    )
    parser.add_argument(
        "--data-files",
        nargs="+",
        default=None,
        help="Explicit Omni-Preference parquet/json/jsonl files.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Number of preference pairs to collate.")
    parser.add_argument(
        "--sampler-batches",
        type=int,
        default=3,
        help="Number of ModalityGroupedBatchSampler batches to print for each padding mode. Use 0 to skip.",
    )
    parser.add_argument("--sampler-seed", type=int, default=0, help="Random seed for ModalityGroupedBatchSampler.")
    parser.add_argument("--max-samples", type=int, default=-1, help="Optional dataset sample limit.")
    parser.add_argument("--max-length", type=int, default=1024, help="Right-padding/truncation max length.")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--chosen-key", default="chosen")
    parser.add_argument("--rejected-key", default="rejected")
    parser.add_argument(
        "--mm-configs",
        default=None,
        help="JSON string for multimodal transform kwargs. If omitted, uses Qwen3-Omni DPO defaults.",
    )
    parser.add_argument("--external-lib", default=None, help="Optional external lib to import before processor setup.")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    return parser.parse_args()


def build_processor(args: argparse.Namespace):
    from transformers import AutoConfig

    if args.external_lib:
        from verl.utils.import_utils import import_external_libs

        import_external_libs(args.external_lib)

    hf_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    architecture = hf_config.architectures[0]
    model_config = ProcessorModelConfig(
        path=args.model_path,
        architecture=architecture,
        trust_remote_code=args.trust_remote_code,
        external_lib=args.external_lib,
    )
    adapter_cls = OmniModelBase.get_class_by_name(
        model_config.architecture,
        model_config.model_stage,
        model_config.external_lib,
    )
    tokenizer = adapter_cls.configure_tokenizer(args.model_path, model_config)
    processor = adapter_cls.configure_processor(args.model_path, model_config)
    return tokenizer, processor


def resolve_data_files(args: argparse.Namespace) -> list[str]:
    if args.data_files:
        return [str(Path(path).expanduser()) for path in args.data_files]
    if args.data_root is None:
        raise ValueError("Pass either --data-files or --data-root.")

    data_root = Path(args.data_root).expanduser()
    data_files = [data_root / modality / f"{args.split}.parquet" for modality in ("audio", "image", "video")]
    missing = [str(path) for path in data_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing modality parquet file(s): {missing}")
    return [str(path) for path in data_files]


def build_dataset(args: argparse.Namespace, processor, pad_mode: str) -> OfflineMLLMDPODataset:
    mm_configs = json.loads(args.mm_configs) if args.mm_configs is not None else dict(DEFAULT_MM_CONFIGS)
    config = OmegaConf.create(
        {
            "pad_mode": pad_mode,
            "max_length": args.max_length,
            "prompt_key": args.prompt_key,
            "chosen_key": args.chosen_key,
            "rejected_key": args.rejected_key,
            "base_transform": "qwen3_omni_moe",
            "data_source": "offline_mllm_dpo",
            "mm_configs": mm_configs,
        }
    )
    return OfflineMLLMDPODataset(
        data_files=resolve_data_files(args),
        tokenizer=None,
        processor=processor,
        config=config,
        max_samples=args.max_samples,
    )


def first_same_modality_features(dataset: OfflineMLLMDPODataset, batch_size: int):
    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")
    modality = dataset.get_modality(0)
    indices = [index for index in range(len(dataset)) if dataset.get_modality(index) == modality][:batch_size]
    if not indices:
        raise ValueError("Cannot find any sample for the first dataset modality.")
    return modality, indices, [dataset[index] for index in indices]


def describe_value(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        if value.is_nested:
            parts = [f"NestedTensor(layout={value.layout})"]
            try:
                parts.append(f"offsets={value.offsets().tolist()}")
            except Exception:
                pass
            try:
                parts.append(f"values_shape={tuple(value.values().shape)}")
            except Exception:
                pass
            return ", ".join(parts)
        return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype})"
    if isinstance(value, np.ndarray):
        return f"ndarray(shape={value.shape}, dtype={value.dtype})"
    if isinstance(value, dict):
        return f"dict(keys={list(value.keys())})"
    batch_size = getattr(value, "batch_size", None)
    if batch_size is not None:
        return f"{type(value).__name__}(batch_size={tuple(batch_size)})"
    if isinstance(value, list | tuple):
        return f"{type(value).__name__}(len={len(value)})"
    return type(value).__name__


def print_value(key: str, value: Any, indent: int = 2) -> None:
    prefix = " " * indent
    print(f"{prefix}{key}: {describe_value(value)}")
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            print_value(str(child_key), child_value, indent=indent + 2)


def as_list(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return list(tolist())
        except Exception:
            pass
    if isinstance(value, list | tuple):
        return list(value)
    return [value]


def batch_modalities(batch: dict[str, Any]) -> list[str]:
    return sorted({str(item) for item in as_list(batch.get("modality", []))})


def print_collated_batch(title: str, batch: dict[str, Any], dataset_len: int, extra: str) -> None:
    print(f"\n=== {title} ===")
    print(f"dataset_len={dataset_len} {extra}")
    print(f"top_level_keys={list(batch.keys())}")
    for key, value in batch.items():
        print_value(str(key), value)


def print_direct_batch(
    title: str,
    dataset: OfflineMLLMDPODataset,
    collate_fn,
    batch_size: int,
    *,
    pad_mode: str | None = None,
) -> None:
    modality, indices, features = first_same_modality_features(dataset, batch_size)
    batch = collate_fn(features, pad_mode=pad_mode)
    print_collated_batch(
        title,
        batch,
        len(dataset),
        extra=f"modality={modality} pair_indices={indices}",
    )


def print_sampler_batches(
    title: str,
    dataset: OfflineMLLMDPODataset,
    collate_fn,
    batch_size: int,
    sampler_batches: int,
    sampler_seed: int,
    *,
    pad_mode: str | None = None,
) -> None:
    if sampler_batches <= 0:
        return
    sampler = ModalityGroupedBatchSampler(
        data_source=dataset,
        batch_size=batch_size,
        seed=sampler_seed,
        num_batches=sampler_batches,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=lambda features: collate_fn(features, pad_mode=pad_mode),
    )
    for batch_index, batch in enumerate(loader):
        print_collated_batch(
            f"{title} sampler_batch={batch_index}",
            batch,
            len(dataset),
            extra=f"modalities={batch_modalities(batch)}",
        )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    _tokenizer, processor = build_processor(args)
    right_dataset = build_dataset(args, processor, pad_mode="right")
    no_padding_dataset = build_dataset(args, processor, pad_mode="no_padding")

    print_direct_batch(
        "pad_mode=right direct_batch",
        right_dataset,
        offline_mllm_dpo_collate_fn,
        args.batch_size,
        pad_mode="right",
    )
    print_direct_batch(
        "pad_mode=no_padding",
        no_padding_dataset,
        offline_mllm_dpo_collate_fn,
        args.batch_size,
        pad_mode="no_padding",
    )
    print_sampler_batches(
        "pad_mode=right",
        right_dataset,
        offline_mllm_dpo_collate_fn,
        args.batch_size,
        args.sampler_batches,
        args.sampler_seed,
        pad_mode="right",
    )
    print_sampler_batches(
        "pad_mode=no_padding",
        no_padding_dataset,
        offline_mllm_dpo_collate_fn,
        args.batch_size,
        args.sampler_batches,
        args.sampler_seed,
        pad_mode="no_padding",
    )


if __name__ == "__main__":
    main()
