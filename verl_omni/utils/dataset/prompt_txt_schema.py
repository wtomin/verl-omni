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
"""Pure helpers for building RLHF rows from plain-text prompts (no heavy imports)."""

from __future__ import annotations

from typing import Any, Mapping


def build_prompt_txt_row(line: str, index: int, config: Mapping[str, Any]) -> dict[str, Any]:
    """Turn one stripped prompt line into a row compatible with :class:`RLHFDataset`.

    ``extra_info["prompt"]`` duplicates the line text so image rewards (e.g. PickScore via
    :func:`~verl_omni.utils.reward_score.pick_score.compute_pickscore_reward`) still receive
    a non-empty prompt when ``reward_model.ground_truth`` is unset.
    """
    system = config.get("txt_system_prompt", None)
    neg_user = config.get("txt_negative_user_content", " ")
    user_only_neg = config.get("txt_user_only_negative_content", " ")

    if system:
        prompt_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": line},
        ]
        negative_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": neg_user},
        ]
    else:
        prompt_messages = [{"role": "user", "content": line}]
        negative_messages = [{"role": "user", "content": user_only_neg}]

    reward_style = config.get("txt_reward_style", "rule")
    reward_gt = config.get("txt_reward_ground_truth", "")
    data_source = config.get("txt_default_data_source", "pick_score")

    return {
        "data_source": data_source,
        "prompt": prompt_messages,
        "negative_prompt": negative_messages,
        "reward_model": {"style": reward_style, "ground_truth": reward_gt},
        "extra_info": {"index": index, "prompt": line},
    }
