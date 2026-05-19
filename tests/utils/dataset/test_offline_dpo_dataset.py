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

import pandas as pd
import torch
from omegaconf import OmegaConf

from verl_omni.utils.dataset.offline_dpo_dataset import (
    OfflineDPODataset,
    expand_offline_dpo_features,
    resolve_materialize_prompts,
)
from verl_omni.utils.dataset.rl_dataset import collate_fn


class _ToyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
        del add_generation_prompt, tokenize, kwargs
        return "\n".join(message["content"] for message in messages if message["role"] == "user")

    def __call__(self, text, add_special_tokens=False, return_tensors="pt", truncation=True, max_length=8):
        del add_special_tokens, return_tensors, truncation
        token_ids = [min(ord(ch), 255) for ch in text][:max_length]
        return {"input_ids": torch.tensor([token_ids], dtype=torch.long)}


def _config():
    return OmegaConf.create(
        {
            "max_prompt_length": 8,
            "prompt_key": "prompt",
            "negative_prompt_key": "negative_prompt",
            "img_win_key": "img_win",
            "img_lose_key": "img_lose",
            "win_score_key": "win_score",
            "lose_score_key": "lose_score",
            "apply_chat_template_kwargs": {},
        }
    )


def test_offline_dpo_dataset_expands_adjacent_win_lose_pairs(tmp_path) -> None:
    data_path = tmp_path / "train.parquet"
    pd.DataFrame(
        [
            {
                "prompt": [{"role": "user", "content": "a cat"}],
                "negative_prompt": [{"role": "user", "content": " "}],
                "img_win": "win.png",
                "img_lose": "lose.png",
                "win_score": 0.9,
                "lose_score": 0.1,
            }
        ]
    ).to_parquet(data_path)

    dataset = OfflineDPODataset(str(data_path), _ToyTokenizer(), config=_config())
    features = expand_offline_dpo_features([dataset[0]])

    assert len(features) == 2
    assert features[0]["uid"] == features[1]["uid"]
    assert features[0]["is_chosen"] is True
    assert features[1]["is_chosen"] is False
    assert features[0]["sample_level_scores"].item() > features[1]["sample_level_scores"].item()
    assert features[0]["image_path"].endswith("win.png")
    assert features[1]["image_path"].endswith("lose.png")


def test_prefers_extra_info_raw_prompt_for_materialize(tmp_path) -> None:
    data_path = tmp_path / "train.parquet"
    chat_like_prompt = (
        "[{'role': 'system', 'content': 'You are a helpful image generation assistant.'}\n"
        " {'role': 'user', 'content': 'wrong caption from prompt column'}]"
    )
    pd.DataFrame(
        [
            {
                "prompt": chat_like_prompt,
                "negative_prompt": [{"role": "user", "content": "neg from column"}],
                "img_win": "win.png",
                "img_lose": "lose.png",
                "win_score": 0.9,
                "lose_score": 0.1,
                "extra_info": {
                    "raw_prompt": "gold tip pyramid in the night",
                    "raw_negative_prompt": " ",
                },
            }
        ]
    ).to_parquet(data_path)

    dataset = OfflineDPODataset(str(data_path), _ToyTokenizer(), config=_config())
    item = dataset[0]

    assert item["prompt_text"] == "gold tip pyramid in the night"
    assert item["negative_prompt_text"] == " "
    features = expand_offline_dpo_features([item])
    assert features[0]["raw_prompt"] == "gold tip pyramid in the night"
    assert features[1]["raw_negative_prompt"] == " "

    raw_prompt, raw_negative = resolve_materialize_prompts(
        chat_like_prompt,
        [{"role": "user", "content": "neg from column"}],
        {"raw_prompt": "caption from extra_info", "raw_negative_prompt": "neg from extra"},
    )
    assert raw_prompt == "caption from extra_info"
    assert raw_negative == "neg from extra"


def test_rl_collate_expands_offline_dpo_pairs(tmp_path) -> None:
    data_path = tmp_path / "train.parquet"
    pd.DataFrame(
        [
            {
                "prompt": [{"role": "user", "content": "a cat"}],
                "img_win": "win.png",
                "img_lose": "lose.png",
                "win_score": 1.0,
                "lose_score": 0.0,
            }
        ]
    ).to_parquet(data_path)

    dataset = OfflineDPODataset(str(data_path), _ToyTokenizer(), config=_config())
    batch = collate_fn([dataset[0]])

    assert batch["prompts"].shape == (2, 8)
    torch.testing.assert_close(batch["sample_level_scores"], torch.tensor([[1.0], [0.0]]))
    assert batch["uid"][0] == batch["uid"][1]
