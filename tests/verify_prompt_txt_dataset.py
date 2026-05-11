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
"""Smoke-test ``PromptTxtRLDataset`` loading without Ray / training.

Two modes:

1. **Default (full)** — calls ``verl_omni.utils.dataset.rl_dataset.create_rl_dataset``, matching
   training. Requires a working editable install (``pip install -e .``) and compatible deps
   (same as ``python -m verl_omni.trainer.diffusion.main_dpo``).

2. ``--schema-only`` — reads UTF-8 lines and builds rows via ``build_prompt_txt_row`` only.
   Does **not** import ``verl_omni`` (no Ray / diffusers). Use this to sanity-check paths and
   line counts before fixing the training environment.

Examples::

    # Lightweight (no verl_omni import)
    python tests/verify_prompt_txt_dataset.py --schema-only \\
        --train /abs/path/train.txt --val /abs/path/test.txt

    # Same code path as training (from repo root)
    PYTHONPATH=. python tests/verify_prompt_txt_dataset.py \\
        --train /abs/path/train.txt --val /abs/path/test.txt --max-samples 8

Exit code 0 only if checks succeed.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_build_prompt_txt_row():
    schema_path = _repo_root() / "verl_omni/utils/dataset/prompt_txt_schema.py"
    spec = importlib.util.spec_from_file_location("prompt_txt_schema_verify", schema_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_prompt_txt_row


def _iter_txt_prompts(path: str):
    with open(path, encoding="utf-8") as fp:
        for raw_line in fp:
            text = raw_line.strip()
            if text:
                yield text


def _schema_only_shard(label: str, path: str, build_row, max_samples: int) -> int:
    lines = list(_iter_txt_prompts(path))
    if not lines:
        raise ValueError(f"{label}: no non-empty lines in {path!r}")
    cap = max_samples if max_samples > 0 else len(lines)
    lines = lines[:cap]
    row0 = build_row(lines[0], 0, {})
    print(f"[ok] {label}: path={path!r} non_empty_lines={len(lines)} first_prompt_len_chars={len(lines[0])}")
    print(f"      row keys={sorted(row0.keys())} prompt_roles={[m['role'] for m in row0['prompt']]}")
    return len(lines)


class _DummyTokenizer:
    """Minimal tokenizer stand-in so RLHFDataset can run with ``filter_overlong_prompts=false``."""

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, **kwargs):
        if tokenize:
            return list(range(32))
        return "<dummy>"


def _smoke_data_config() -> dict:
    """Subset of ``legacy_data.yaml`` fields required by ``RLHFDataset.__init__``."""
    return {
        "custom_cls": {
            "path": "pkg://verl_omni.utils.dataset.prompt_txt_rl_dataset",
            "name": "PromptTxtRLDataset",
        },
        "cache_dir": "~/.cache/verl/rlhf",
        "prompt_key": "prompt",
        "negative_prompt_key": "negative_prompt",
        "image_key": "images",
        "video_key": "videos",
        "max_prompt_length": 8192,
        "return_raw_chat": True,
        "return_full_prompt": False,
        "truncation": "error",
        "filter_overlong_prompts": False,
        "apply_chat_template_kwargs": {},
        "filter_overlong_prompts_workers": 1,
        "use_shm": False,
        "need_tools_kwargs": False,
        "filter_prompts": True,
        "return_multi_modal_inputs": True,
        "shuffle": False,
        "seed": None,
        "trust_remote_code": False,
    }


def _full_load_shard(label: str, path: str, data_cfg: dict, tokenizer, max_samples: int) -> int:
    from omegaconf import OmegaConf

    from verl_omni.utils.dataset.rl_dataset import create_rl_dataset

    cfg = OmegaConf.create(data_cfg)
    ds = create_rl_dataset(
        path,
        cfg,
        tokenizer=tokenizer,
        processor=None,
        is_train=(label == "train"),
        max_samples=max_samples,
    )
    n = len(ds)
    sample = ds[0]
    keys = sorted(sample.keys())
    print(f"[ok] {label}: path={path!r} len={n} sample_keys={keys}")
    rp = sample.get("raw_prompt")
    if rp is not None:
        print(f"      raw_prompt first message: {rp[0]}")
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify PromptTxtRLDataset loads train/val txt shards.")
    parser.add_argument("--train", required=True, help="Absolute or relative path to train.txt")
    parser.add_argument("--val", required=True, help="Absolute or relative path to val.txt")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="If >0, cap rows loaded per shard (same meaning as train_max_samples).",
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Only check UTF-8 readability + row schema (no verl_omni / create_rl_dataset).",
    )
    args = parser.parse_args()

    try:
        if args.schema_only:
            build_row = _load_build_prompt_txt_row()
            _schema_only_shard("train", args.train, build_row, args.max_samples)
            _schema_only_shard("val", args.val, build_row, args.max_samples)
            print("Schema-only checks passed (paths readable, non-empty, rows well-formed).")
        else:
            cfg_dict = _smoke_data_config()
            tok = _DummyTokenizer()
            _full_load_shard("train", args.train, cfg_dict, tok, args.max_samples)
            _full_load_shard("val", args.val, cfg_dict, tok, args.max_samples)
            print("Full load passed (create_rl_dataset + __getitem__).")
    except Exception as e:
        print(f"[fail] {type(e).__name__}: {e}", file=sys.stderr)
        raise SystemExit(1) from e

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
