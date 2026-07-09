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

"""Verify Omni-Preference offline MLLM DPO parquet with RLHFDataset + collate fn.

Runs on CPU without model weights. Checks that:

  1. ``RLHFDataset`` loads local parquet and passes ``chosen`` / ``rejected``.
  2. ``offline_mllm_dpo_collate_fn`` expands each pair into adjacent samples.
  3. Batched ``is_chosen`` alternates True/False per preference pair.

Usage
-----
# All three modalities (default parquet root)
python examples/dpo_trainer/qwen3_omni/verify_omni_preference_dpo_pipeline.py \\
    --parquet_root ~/Omni-Preference/parquet_dpo

# Single modality
python examples/dpo_trainer/qwen3_omni/verify_omni_preference_dpo_pipeline.py \\
    --train_files ~/Omni-Preference/parquet_dpo/video/train.parquet
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from verl.utils.dataset.rl_dataset import RLHFDataset
from verl_omni.utils.dataset.offline_mllm_dpo_dataset import offline_mllm_dpo_collate_fn

DEFAULT_MODALITIES = ("image", "video", "audio")


def _dummy_tokenizer():
    tok = MagicMock()
    tok.apply_chat_template = lambda msgs, **kw: "dummy"
    tok.pad_token_id = 0
    tok.eos_token_id = 1
    tok.__call__ = lambda *a, **kw: {"input_ids": torch.tensor([[0, 1, 2]])}
    return tok


def _dummy_processor():
    proc = MagicMock()
    proc.apply_chat_template = lambda msgs, **kw: "dummy"
    proc.tokenizer = _dummy_tokenizer()
    proc.image_processor = MagicMock(patch_size=14)
    return proc


def _prompt_kind(raw_prompt: list[dict]) -> str:
    user_msg = next((m for m in raw_prompt if m.get("role") == "user"), None)
    if user_msg is None:
        return "unknown"
    content = user_msg.get("content")
    if isinstance(content, str):
        return "text"
    if isinstance(content, list):
        media_types = {part.get("type") for part in content if isinstance(part, dict)}
        for kind in ("image", "video", "audio"):
            if kind in media_types:
                return kind
    return "unknown"


def _count_modalities(raw_prompts) -> dict[str, int]:
    counts = {"image": 0, "video": 0, "audio": 0, "text": 0, "unknown": 0}
    for msgs in raw_prompts:
        counts[_prompt_kind(msgs)] += 1
    return counts


def _resolve_train_files(parquet_root: str | None, modalities: list[str], train_files: list[str] | None) -> list[str]:
    if train_files:
        return [os.path.abspath(os.path.expanduser(path)) for path in train_files]

    if parquet_root is None:
        raise ValueError("Provide either --train_files or --parquet_root.")

    root = os.path.abspath(os.path.expanduser(parquet_root))
    paths = [os.path.join(root, modality, "train.parquet") for modality in modalities]
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError("Missing parquet files:\n  " + "\n  ".join(missing))
    return paths


def _verify_dataset(dataset: RLHFDataset) -> str:
    sample = dataset[0]
    print("── Raw item (dataset[0]) ──────────────────────────────────────────")
    for key, value in sample.items():
        snippet = repr(value[0] if isinstance(value, list) and value else value)[:100]
        print(f"  {key:20s}: {type(value).__name__:10s}  {snippet}")

    assert "raw_prompt" in sample, "raw_prompt missing"
    assert "chosen" in sample, "chosen column not passed through by RLHFDataset"
    assert "rejected" in sample, "rejected column not passed through by RLHFDataset"

    kind = _prompt_kind(sample["raw_prompt"])
    assert kind in {"image", "video", "audio", "text"}, f"Unexpected prompt kind: {kind}"

    print(f"\n  ✓ raw_prompt kind = {kind}")
    print(f"  ✓ chosen   = {sample['chosen'][:80]!r}")
    print(f"  ✓ rejected = {sample['rejected'][:80]!r}")
    return kind


def _verify_loader(dataset: RLHFDataset, batch_size: int, num_batches: int) -> None:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=offline_mllm_dpo_collate_fn,
        num_workers=0,
    )

    print(f"\n── {num_batches} batches (batch_size={batch_size} pairs → {batch_size * 2} samples) ──")
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break

        n = len(batch["response"])
        assert n == batch_size * 2, f"Batch {batch_idx}: expected {batch_size * 2} samples, got {n}"

        is_chosen = list(batch["is_chosen"])
        for i in range(0, len(is_chosen), 2):
            assert bool(is_chosen[i]) is True, f"position {i} should be chosen"
            assert bool(is_chosen[i + 1]) is False, f"position {i + 1} should be rejected"

        counts = _count_modalities(batch["raw_prompt"])
        sources = {str(source) for source in batch.get("data_source", [])}
        print(
            f"  batch {batch_idx}: {n} samples ({n // 2} pairs) | "
            f"image={counts['image']} video={counts['video']} audio={counts['audio']} | "
            f"sources={sources}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Omni-Preference offline MLLM DPO parquet pipeline.")
    parser.add_argument(
        "--parquet_root",
        default=os.path.expanduser("~/Omni-Preference/parquet_dpo"),
        help="Root with {image,video,audio}/train.parquet (used when --train_files is omitted).",
    )
    parser.add_argument(
        "--train_files",
        nargs="+",
        default=None,
        help="Explicit train.parquet path(s). Overrides --parquet_root.",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=list(DEFAULT_MODALITIES),
        default=list(DEFAULT_MODALITIES),
        help="Modalities to load when using --parquet_root.",
    )
    parser.add_argument("--batch_size", type=int, default=2, help="DataLoader batch size in preference pairs.")
    parser.add_argument("--max_samples", type=int, default=16, help="Maximum parquet rows to load.")
    parser.add_argument("--num_batches", type=int, default=2, help="Number of batches to iterate.")
    args = parser.parse_args()

    try:
        train_files = _resolve_train_files(args.parquet_root, args.modalities, args.train_files)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    data_cfg = OmegaConf.create(
        {
            "cache_dir": "~/.cache/verl/omni_preference_dpo_verify",
            "prompt_key": "prompt",
            "max_prompt_length": 4096,
            "filter_overlong_prompts": False,
            "truncation": "left",
            "return_raw_chat": False,
        }
    )

    print("Loading parquet files:", flush=True)
    for path in train_files:
        print(f"  - {path}", flush=True)

    dataset = RLHFDataset(
        data_files=train_files,
        tokenizer=_dummy_tokenizer(),
        processor=_dummy_processor(),
        config=data_cfg,
        max_samples=args.max_samples,
    )
    print(f"\nDataset size: {len(dataset)} pairs\n", flush=True)
    if len(dataset) == 0:
        print("ERROR: dataset is empty.", file=sys.stderr)
        sys.exit(1)

    _verify_dataset(dataset)
    _verify_loader(dataset, batch_size=min(args.batch_size, len(dataset)), num_batches=args.num_batches)

    print("\n✓ Omni-Preference DPO parquet verification passed.")
    print("  RLHFDataset               — loads parquet, passes chosen/rejected through")
    print("  offline_mllm_dpo_collate_fn — expands pairs to adjacent chosen/rejected samples")


if __name__ == "__main__":
    main()
