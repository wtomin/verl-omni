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

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_PATH = _REPO_ROOT / "verl_omni/utils/dataset/prompt_txt_schema.py"


@pytest.fixture(scope="module")
def build_prompt_txt_row():
    spec = importlib.util.spec_from_file_location("prompt_txt_schema", _SCHEMA_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_prompt_txt_row


def test_make_example_user_only(build_prompt_txt_row):
    row = build_prompt_txt_row("a red circle", 7, {})
    assert row["data_source"] == "jpeg_compressibility"
    assert row["prompt"] == [{"role": "user", "content": "a red circle"}]
    assert row["negative_prompt"] == [{"role": "user", "content": " "}]
    assert row["reward_model"] == {"style": "rule", "ground_truth": ""}
    assert row["extra_info"] == {"index": 7}


def test_make_example_with_system_prompt(build_prompt_txt_row):
    cfg = {"txt_system_prompt": "SYS:", "txt_negative_user_content": "NEG"}
    row = build_prompt_txt_row("hello", 0, cfg)
    assert row["prompt"] == [
        {"role": "system", "content": "SYS:"},
        {"role": "user", "content": "hello"},
    ]
    assert row["negative_prompt"] == [
        {"role": "system", "content": "SYS:"},
        {"role": "user", "content": "NEG"},
    ]
