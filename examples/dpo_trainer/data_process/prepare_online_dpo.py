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
"""Preprocess a prompt-only text file into parquet rows for diffusion DPO/RLHF data loading.

The input file is a UTF-8 text file with one prompt per line.

Example:

    python3 examples/dpo_trainer/data_process/prepare_online_dpo.py \
        --input_file dataset/my_prompts/train_prompts.txt \
        --output_file data/my_prompts/train.parquet \
        --data_source online_dpo \
        --system_prompt "You are a helpful image generation assistant."

The output schema follows the existing ``RLHFDataset`` parquet convention used
by VeRL-Omni while keeping the raw prompt in ``extra_info.raw_prompt`` for
reward models that score generated images against the prompt text.
"""

import argparse
import os
from pathlib import Path

import pandas as pd

DEFAULT_SYSTEM_PROMPT = "You are a helpful image generation assistant."


def _read_text(path: str | None) -> str | None:
    if path is None:
        return None
    with open(os.path.expanduser(path), encoding="utf-8") as f:
        return f.read().strip()


def _resolve_system_prompt(args: argparse.Namespace) -> str:
    from_file = _read_text(args.system_prompt_file)
    if from_file is not None:
        return from_file
    return args.system_prompt


def _read_prompts(path: Path) -> list[tuple[int, str]]:
    prompts = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            prompt = line.strip()
            if prompt:
                prompts.append((line_number, prompt))
    if not prompts:
        raise ValueError(f"No non-empty prompts found in {path}")
    return prompts


def _build_rows(
    prompts: list[tuple[int, str]],
    *,
    split: str,
    data_source: str,
    ability: str,
    system_prompt: str,
    negative_prompt: str,
) -> list[dict]:
    rows = []
    for index, (line_number, prompt_text) in enumerate(prompts):
        rows.append(
            {
                "data_source": data_source,
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text},
                ],  # consumed by actor/rollout
                "negative_prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": negative_prompt},
                ],  # consumed by actor/rollout
                "ability": ability,
                # VisualRewardManager currently reads ground_truth unconditionally.
                "reward_model": {"style": "model", "ground_truth": ""},
                "extra_info": {
                    "split": split,
                    "index": index,
                    "line_number": line_number,
                    "raw_prompt": prompt_text,
                    "raw_negative_prompt": negative_prompt,
                },
            }
        )
    return rows


def _write_file(
    *,
    input_path: Path,
    output_path: Path,
    data_source: str,
    ability: str,
    system_prompt: str,
    negative_prompt: str,
    split: str,
) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"Expected input file: {input_path}")

    rows = _build_rows(
        _read_prompts(input_path),
        split=split,
        data_source=data_source,
        ability=ability,
        system_prompt=system_prompt,
        negative_prompt=negative_prompt,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path)
    print(f"Wrote {len(rows)} samples to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert one prompt-only text file to one parquet file.")
    parser.add_argument("--input_file", required=True, help="UTF-8 text file containing one prompt per line.")
    parser.add_argument("--output_file", required=True, help="Parquet file to write.")
    parser.add_argument(
        "--data_source",
        default="online_dpo",
        help="Dataset identifier. This dataset only contains prompts, and rejected/selected samples after rollout.",
    )
    parser.add_argument("--ability", default="online_dpo", help="Task ability tag stored in each parquet row.")
    parser.add_argument(
        "--system_prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt prepended to every user prompt.",
    )
    parser.add_argument(
        "--system_prompt_file",
        default=None,
        help="Optional UTF-8 file containing the system prompt. Overrides --system_prompt.",
    )
    parser.add_argument(
        "--negative_prompt",
        default=" ",
        help="Negative user prompt for classifier-free guidance.",
    )
    parser.add_argument("--split", default=None, help="Optional split name stored in extra_info.split.")
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS destination for the output parquet file.")
    args = parser.parse_args()

    input_path = Path(os.path.expanduser(args.input_file))
    output_path = Path(os.path.expanduser(args.output_file))
    split = args.split or output_path.stem
    system_prompt = _resolve_system_prompt(args)

    _write_file(
        input_path=input_path,
        output_path=output_path,
        data_source=args.data_source,
        ability=args.ability,
        system_prompt=system_prompt,
        negative_prompt=args.negative_prompt,
        split=split,
    )

    if args.hdfs_dir is not None:
        from verl.utils.hdfs_io import copy, makedirs

        makedirs(args.hdfs_dir)
        copy(src=str(output_path), dst=args.hdfs_dir)


if __name__ == "__main__":
    main()
