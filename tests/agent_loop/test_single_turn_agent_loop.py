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
import asyncio
from types import SimpleNamespace

from verl_omni.agent_loop.single_turn_agent_loop import DiffusionSingleTurnAgentLoop


class _TokenizerWithoutChatTemplate:
    chat_template = None

    def __init__(self):
        self.texts = []

    def __call__(self, text, add_special_tokens=False):
        self.texts.append((text, add_special_tokens))
        return {"input_ids": [len(self.texts), len(text)]}


class _ServerManager:
    def __init__(self):
        self.request = None

    async def generate(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            diffusion_output="image",
            log_probs=None,
            num_preempted=None,
            extra_fields={},
        )


def test_single_turn_text_tokenizer_does_not_require_chat_template():
    async def process_vision_info(_raw_prompt):
        return {}

    async def apply_chat_template(*_args, **_kwargs):
        raise AssertionError("chat template should not be used for plain text tokenizers")

    tokenizer = _TokenizerWithoutChatTemplate()
    server_manager = _ServerManager()
    agent_loop = object.__new__(DiffusionSingleTurnAgentLoop)
    agent_loop.tokenizer = tokenizer
    agent_loop.processor = None
    agent_loop.server_manager = server_manager
    agent_loop.process_vision_info = process_vision_info
    agent_loop.apply_chat_template = apply_chat_template

    output = asyncio.run(
        agent_loop.run(
            {},
            raw_prompt=[
                {"role": "system", "content": "Describe the image."},
                {"role": "user", "content": "A small cat."},
            ],
            raw_negative_prompt="low quality",
        )
    )

    assert output.prompt_ids == [1, len("Describe the image.\nA small cat.")]
    assert server_manager.request["prompt"] == "Describe the image.\nA small cat."
    assert server_manager.request["negative_prompt"] == "low quality"
    assert server_manager.request["negative_prompt_ids"] == [2, len("low quality")]
    assert tokenizer.texts == [
        ("Describe the image.\nA small cat.", False),
        ("low quality", False),
    ]
