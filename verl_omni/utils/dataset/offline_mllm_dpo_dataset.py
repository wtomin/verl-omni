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

"""Offline MLLM DPO collate utilities for Qwen3-Omni preference training.

Dataset loading reuses the upstream ``RLHFDataset`` directly — no custom
dataset class is required.  ``RLHFDataset.__getitem__`` returns all parquet
columns it does not recognise as media-placeholder columns, so ``chosen`` and
``rejected`` pass through transparently alongside ``raw_prompt``.

Multimodal content (``{"type": "image", …}`` or ``{"type": "video", …}``) in
the structured ``prompt`` field is already in the list-of-dicts format that
``RLHFDataset._build_messages`` passes through unchanged (it only rewrites
string-content messages that contain ``<video>`` / ``<audio>`` placeholder
tokens). Mixed batches with image, text-only, and video rows work without any
special handling: ``maybe_filter_out_long_prompts`` already calls
``_process_multi_modal_info`` for visual media while text-only rows use the
standard chat path.

Usage
-----
Configure the trainer with:

    data.train_files=[image/train.parquet, text/train.parquet, video/train.parquet]
    data.val_files=[image/test.parquet, text/test.parquet, video/test.parquet]
    data.custom_cls.path=pkg://verl_omni.utils.dataset.offline_mllm_dpo_dataset
    data.custom_cls.collate_fn=offline_mllm_dpo_collate_fn

The ``custom_cls.name`` key is **not** needed; ``RLHFDataset`` (the default)
is used for data loading.  Only the collate function is overridden.

Parquet rows are produced by:

    examples/dpo_trainer/data_process/llava_hound_dpo_multisource.py

Required parquet columns:

    prompt      list[dict]   Chat-style messages; user turn holds typed media
                             objects ({"type":"image","image":"…"} or
                             {"type":"video","video":"…"}) or plain text.
    chosen      str          Preferred assistant response.
    rejected    str          Less preferred assistant response.

Optional pass-through columns (returned by RLHFDataset as-is):

    data_source, ability, reward_model, extra_info, uid
"""

from __future__ import annotations

from typing import Any

from verl.utils.dataset.rl_dataset import collate_fn as _upstream_collate_fn

# Sentinel key injected into each batch sample so the collate function can
# identify rows that come from DPO-pair parquet files.
_DPO_PAIR_KEY = "chosen"


def expand_offline_mllm_dpo_features(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand each preference pair into adjacent chosen / rejected samples.

    Each input sample (one parquet row after ``RLHFDataset.__getitem__``) has
    both ``chosen`` and ``rejected`` fields.  This function splits each into
    two consecutive dicts tagged with ``is_chosen: True`` and
    ``is_chosen: False``.  The DPO loss relies on this adjacency to match pairs
    without needing a separate pair-ID lookup.

    Samples that lack a ``chosen`` field are forwarded unchanged (allows the
    collate fn to work safely on mixed or non-DPO batches).
    """
    expanded: list[dict[str, Any]] = []
    for feature in features:
        if _DPO_PAIR_KEY not in feature:
            expanded.append(feature)
            continue

        chosen = feature.pop("chosen")
        rejected = feature.pop("rejected")

        # chosen (preferred) — even position in each pair
        expanded.append({**feature, "response": chosen, "is_chosen": True})
        # rejected (less preferred) — odd position in each pair
        expanded.append({**feature, "response": rejected, "is_chosen": False})

    return expanded


def offline_mllm_dpo_collate_fn(features: list[dict[str, Any]]):
    """Collate offline MLLM DPO pairs, expanding each row to chosen/rejected.

    Wraps the upstream ``collate_fn`` after the pair-expansion step.
    ``RLHFDataset`` is used for data loading; only this collate function is
    registered via ``data.custom_cls.collate_fn``.
    """
    if features and _DPO_PAIR_KEY in features[0]:
        features = expand_offline_mllm_dpo_features(features)
    return _upstream_collate_fn(features)
