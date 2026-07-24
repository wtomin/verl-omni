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

import os

os.environ.setdefault("VERL_OMNI_SKIP_AUTO_IMPORTS", "1")

import torch

from verl_omni.pipelines.bagel_flow_grpo.bagel_model import BagelTrainingConfig
from verl_omni.pipelines.bagel_sft.bagel_sft_model import BagelForSFT


def test_bagel_sft_tiny_text_forward():
    config = BagelTrainingConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
        max_position_embeddings=32,
        latent_channel=1,
    )
    model = BagelForSFT(config)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)

    output = model(input_ids=input_ids, attention_mask=attention_mask)

    assert output.logits.shape == (1, 4, 32)
    assert output.image_velocity is None
