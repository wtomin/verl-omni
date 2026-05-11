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
"""RLHF-style dataset: one text prompt per line in ``*.txt`` files.

Use with Hydra ``data.custom_cls`` (see ``trainer/config/data/prompt_txt_data.yaml``).

Each non-empty line becomes a row compatible with :class:`~verl_omni.utils.dataset.rl_dataset.RLHFDataset`
(chat ``prompt`` / ``negative_prompt``, ``data_source``, ``reward_model``, ``extra_info``).
"""

from __future__ import annotations

from typing import Any

import datasets
import numpy as np

from verl_omni.utils.dataset.prompt_txt_schema import build_prompt_txt_row
from verl_omni.utils.dataset.rl_dataset import RLHFDataset


class PromptTxtRLDataset(RLHFDataset):
    """Load prompts from UTF-8 text files (one prompt per line).

    Instantiate only via ``verl.trainer.main_ppo.create_rl_dataset`` /
    ``verl_omni.utils.dataset.rl_dataset.create_rl_dataset`` with ``custom_cls`` set so the
    constructor matches upstream ``(data_files, tokenizer, processor, config, max_samples)``.

    Optional ``config`` keys (all under ``data`` in Hydra):

    - ``txt_default_data_source``: forwarded to ``data_source`` (default ``pick_score``).
    - ``txt_reward_style`` / ``txt_reward_ground_truth``: ``reward_model`` dict fields.
    - ``txt_system_prompt``: if set, prompts become
      ``[system, user(line)]``; negative prompt reuses the same system message.
    - ``txt_negative_user_content``: user-role negative text when using a system prompt
      (default single space ``" "``, matching dummy diffusion data).
    - ``txt_user_only_negative_content``: when no system prompt, negative prompt is a one-turn user
      message with this content (default ``" "``).
    """

    def _make_example(self, line: str, index: int) -> dict[str, Any]:
        return build_prompt_txt_row(line, index, self.config)

    def _read_files_and_tokenize(self):
        rows: list[dict[str, Any]] = []
        global_idx = 0
        for filepath in self.data_files:
            path_str = str(filepath)
            if not path_str.endswith(".txt"):
                raise ValueError(
                    "PromptTxtRLDataset only supports `.txt` shards (one prompt per line). "
                    f"Got {path_str!r}. Drop `custom_cls` or convert to parquet/jsonl for the "
                    "default RLHFDataset."
                )
            with open(filepath, encoding="utf-8") as fp:
                for raw_line in fp:
                    text = raw_line.strip()
                    if not text:
                        continue
                    rows.append(self._make_example(text, global_idx))
                    global_idx += 1

        self.dataframe = datasets.Dataset.from_list(rows)
        total = len(self.dataframe)
        print(f"dataset len: {total}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} samples out of {total}")

        filtered = self.maybe_filter_out_long_prompts(self.dataframe)
        # Upstream RLHFDataset returns None when ``filter_overlong_prompts`` is False.
        if filtered is not None:
            self.dataframe = filtered
