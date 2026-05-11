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
"""Smoke-test UnifiedReward inference through an OpenAI-compatible reward router.

Example:
    PYTHONPATH=. python tests/verify_unified_reward.py \
        --router-address localhost:8080 \
        --image /path/to/generated.png \
        --prompt "a cute cat sitting on a sofa"
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from verl_omni.utils.reward_score.unified_reward import compute_score_unified_reward


def _load_image_tensor(path: str) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    data = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(data).permute(2, 0, 1)


async def _run(args: argparse.Namespace) -> dict:
    image = _load_image_tensor(args.image)
    return await compute_score_unified_reward(
        data_source="unified_reward",
        solution_image=image,
        ground_truth=args.prompt,
        extra_info={"prompt": args.prompt},
        reward_router_address=args.router_address,
        model_name=args.model_name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-address", default="localhost:8080", help="Reward router host:port.")
    parser.add_argument("--image", required=True, help="Path to one generated image.")
    parser.add_argument("--prompt", required=True, help="Text caption/prompt used for scoring.")
    parser.add_argument(
        "--model-name",
        default="CodeGoat24/UnifiedReward-2.0-qwen3vl-8b",
        help="OpenAI model name served by vLLM. Should be the same as SERVING_MODEL_NAME in "
        "``vllm serve --served-model-name SERVING_MODEL_NAME``.",
    )
    args = parser.parse_args()

    if not Path(args.image).is_file():
        raise FileNotFoundError(args.image)

    result = asyncio.run(_run(args))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
