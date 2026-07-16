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

"""Create the fixed one-row dataset used by the Qwen-Image FlowGRPO nightly."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SYSTEM_PROMPT = (
    "Describe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:"
)
USER_PROMPT = "A high-resolution photograph of a single red apple resting on a wooden table."


def build_row() -> dict:
    return {
        "data_source": "jpeg_compressibility",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        "negative_prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": " "},
        ],
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {"split": "train", "index": 0},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="~/data/qwen_image_single",
        help="Directory for train.parquet and test.parquet.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame([build_row()])
    frame.to_parquet(output_dir / "train.parquet")
    frame.to_parquet(output_dir / "test.parquet")
    print(f"Wrote {output_dir / 'train.parquet'} and {output_dir / 'test.parquet'}")


if __name__ == "__main__":
    main()
