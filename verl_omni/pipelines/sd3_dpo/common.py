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

import torch


def maybe_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def prompt_embeds_mask_from_embeds(prompt_embeds: torch.Tensor) -> torch.Tensor:
    return torch.ones(
        prompt_embeds.shape[0],
        prompt_embeds.shape[1],
        dtype=torch.int32,
        device=prompt_embeds.device,
    )
