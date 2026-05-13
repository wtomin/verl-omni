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

"""Prompt helpers shared by diffusion agent loops."""

from collections.abc import Mapping
from typing import Any


def stringify_prompt_part(part: Any) -> str:
    if part is None:
        return ""
    if isinstance(part, str):
        return part
    if isinstance(part, Mapping):
        if "text" in part:
            return stringify_prompt_part(part["text"])
        if "content" in part:
            return stringify_prompt_part(part["content"])
        return ""
    if isinstance(part, (list, tuple)):
        return " ".join(text for item in part if (text := stringify_prompt_part(item)))
    if hasattr(part, "tolist") and not isinstance(part, str):
        return stringify_prompt_part(part.tolist())
    return str(part)


def stringify_prompt_messages(messages: Any) -> str:
    """Extract plain text from chat-style prompt messages for diffusion pipelines."""
    if isinstance(messages, Mapping):
        return stringify_prompt_part(messages.get("content", messages.get("prompt", "")))
    if isinstance(messages, str):
        return messages
    if hasattr(messages, "tolist") and not isinstance(messages, str):
        messages = messages.tolist()
    if isinstance(messages, (list, tuple)):
        return "\n".join(text for message in messages if (text := stringify_prompt_messages(message)))
    return stringify_prompt_part(messages)
