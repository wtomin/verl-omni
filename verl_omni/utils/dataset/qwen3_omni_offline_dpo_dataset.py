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
"""Offline DPO dataset for Qwen3-Omni audio+image caption pairs.

Each parquet row is a logical preference pair:

``prompt`` + ``answer_win``  -> chosen sample
``prompt`` + ``answer_lose`` -> rejected sample

The custom collate function expands rows into adjacent ``[chosen, rejected]``
samples so downstream DPO code can keep pair ordering intact.

Multimodal encoding follows the official Qwen-Omni preprocessing flow::

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=...)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=...)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, ...)
"""

import copy
import os
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from verl.utils.dataset.rl_dataset import collate_fn as _upstream_collate_fn

QWEN3_OMNI_OFFLINE_DPO_PAIR_MARKER = "__qwen3_omni_offline_dpo_pair__"
_DATA_FILE_KEY = "__data_file__"
_SEQUENCE_KEYS = frozenset({"input_ids", "attention_mask", "loss_mask"})
_MROPE_POSITION_IDS_DIM = 3


def _import_process_mm_info():
    try:
        from qwen_omni_utils import process_mm_info
    except ImportError as exc:
        raise ImportError(
            "Qwen3OmniOfflineDPODataset requires `qwen_omni_utils`. Install it using `pip install qwen-omni-utils`."
        ) from exc
    return process_mm_info


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return value
    return [value]


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "as_py"):
        value = value.as_py()
        if isinstance(value, dict):
            return value
    raise TypeError(f"Expected a dict-like value, got {type(value)}")


def _normalize_nested(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_normalize_nested(item) for item in value.tolist()]
    if isinstance(value, list | tuple):
        return [_normalize_nested(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_nested(item) for key, item in value.items()}
    if hasattr(value, "as_py"):
        return _normalize_nested(value.as_py())
    return value


def _drop_none_values(value: Any) -> Any:
    if isinstance(value, list):
        return [_drop_none_values(item) for item in value]
    if isinstance(value, dict):
        return {key: _drop_none_values(item) for key, item in value.items() if item is not None}
    return value


def _read_dataframe(data_files: str | Sequence[str]) -> pd.DataFrame:
    paths = [data_files] if isinstance(data_files, str) else list(data_files)
    frames = []
    for data_file in paths:
        path = Path(os.path.expanduser(str(data_file))).resolve()
        if path.suffix == ".jsonl":
            frame = pd.read_json(path, lines=True)
        elif path.suffix == ".json":
            frame = pd.read_json(path)
        else:
            frame = pd.read_parquet(path)
        frame[_DATA_FILE_KEY] = str(path)
        frames.append(frame)
    if not frames:
        raise ValueError("Qwen3-Omni offline DPO dataset requires at least one data file.")
    return pd.concat(frames, ignore_index=True)


def _resolve_path(path: Any, data_file: str | None) -> str:
    path = os.path.expanduser(str(path))
    if os.path.isabs(path) or data_file is None:
        return path
    return os.path.normpath(os.path.join(os.path.dirname(os.path.expanduser(data_file)), path))


def _resolve_media_reference(value: Any, data_file: str | None, *, use_file_uri: bool) -> Any:
    """Resolve a media reference for ``process_mm_info``.

    Supports the formats documented for Qwen-Omni: local ``file://`` URIs, plain
    paths, http(s) URLs, base64 data URIs, PIL images, numpy audio, and lists of
    video frames.
    """
    if isinstance(value, list):
        return [_resolve_media_reference(item, data_file, use_file_uri=use_file_uri) for item in value]
    if not isinstance(value, str):
        return value
    if value.startswith(("http://", "https://", "data:", "file://")):
        return value
    resolved = _resolve_path(value, data_file)
    if use_file_uri:
        return Path(os.path.abspath(resolved)).as_uri()
    return resolved


def _resolve_message_media_paths(
    messages: list[dict[str, Any]],
    data_file: str | None,
    *,
    use_file_uri: bool,
) -> list[dict[str, Any]]:
    resolved = copy.deepcopy(messages)
    for message in resolved:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            media_type = item.get("type")
            if media_type in {"image", "audio", "video"} and media_type in item:
                item[media_type] = _resolve_media_reference(item[media_type], data_file, use_file_uri=use_file_uri)
    return resolved


def _extra_info_dict(extra_info: Any) -> dict[str, Any]:
    if extra_info is None:
        return {}
    extra_info = _normalize_nested(extra_info)
    if isinstance(extra_info, dict):
        return extra_info
    return {"raw_extra_info": extra_info}


def _resolve_mm_processor_kwargs(row: dict[str, Any], config: DictConfig) -> dict[str, Any]:
    config_kwargs = dict(config.get("mm_processor_kwargs", {}) or {})
    legacy_kwargs = dict(config.get("apply_chat_template_kwargs", {}) or {})
    row_kwargs = _normalize_nested(row.get("mm_processor_kwargs"))
    if not isinstance(row_kwargs, dict):
        row_kwargs = {}
    return {**legacy_kwargs, **config_kwargs, **row_kwargs}


def _to_sequence_tensor(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 1:
        return value
    if value.ndim == 2 and value.shape[0] == 1:
        return value[0]
    raise ValueError(f"Unexpected sequence tensor shape: {tuple(value.shape)}")


def _to_mrope_position_ids(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 3 and value.shape[0] == _MROPE_POSITION_IDS_DIM:
        return value[:, 0, :]
    if value.ndim == 2 and value.shape[0] == _MROPE_POSITION_IDS_DIM:
        return value
    raise ValueError(f"Unexpected mrope position_ids shape: {tuple(value.shape)}")


def _slice_sequence(value: torch.Tensor, *, max_length: int, truncation: str) -> torch.Tensor:
    if value.ndim == 1:
        if truncation == "left":
            return value[-max_length:]
        return value[:max_length]
    if value.ndim == 2 and value.shape[0] == _MROPE_POSITION_IDS_DIM:
        if truncation == "left":
            return value[:, -max_length:]
        return value[:, :max_length]
    raise ValueError(f"Unsupported tensor shape for truncation: {tuple(value.shape)}")


def _pad_sequence(value: torch.Tensor, *, pad_len: int, pad_value: int | float) -> torch.Tensor:
    if pad_len <= 0:
        return value
    if value.ndim == 1:
        pad = torch.full((pad_len,), pad_value, dtype=value.dtype)
        return torch.cat((value, pad), dim=0)
    if value.ndim == 2 and value.shape[0] == _MROPE_POSITION_IDS_DIM:
        pad = torch.full((_MROPE_POSITION_IDS_DIM, pad_len), pad_value, dtype=value.dtype)
        return torch.cat((value, pad), dim=-1)
    raise ValueError(f"Unsupported tensor shape for padding: {tuple(value.shape)}")


class Qwen3OmniOfflineDPODataset(Dataset):
    """Dataset for FineVideo-style Qwen3-Omni offline DPO parquet rows."""

    def __init__(self, data_files, tokenizer, processor=None, config: DictConfig | None = None, max_samples: int = -1):
        if config is None:
            raise ValueError("Qwen3OmniOfflineDPODataset requires a data config.")
        if processor is None:
            raise ValueError(
                "Qwen3OmniOfflineDPODataset requires an Omni processor "
                "(e.g. Qwen3OmniMoeProcessor or AutoProcessor.from_pretrained)."
            )
        self.data_files = [data_files] if isinstance(data_files, str) else list(data_files)
        self.dataframe = _read_dataframe(self.data_files)
        if max_samples is not None and max_samples > 0:
            self.dataframe = self.dataframe.iloc[:max_samples]

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.prompt_key = config.get("prompt_key", "prompt")
        self.answer_win_key = config.get("answer_win_key", "answer_win")
        self.answer_lose_key = config.get("answer_lose_key", "answer_lose")
        self.win_score_key = config.get("win_score_key", "win_score")
        self.lose_score_key = config.get("lose_score_key", "lose_score")
        self.max_length = config.get(
            "max_length",
            config.get("max_prompt_length", 1024) + config.get("max_response_length", 1024),
        )
        self.truncation = config.get("truncation", "error")
        self.use_file_uri_for_media = config.get("use_file_uri_for_media", True)
        self._process_mm_info = _import_process_mm_info()

        required = {
            self.prompt_key,
            self.answer_win_key,
            self.answer_lose_key,
        }
        missing = required - set(self.dataframe.columns)
        if missing:
            raise ValueError(f"Qwen3-Omni offline DPO data is missing required columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.dataframe)

    def _encode_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        mm_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        use_audio_in_video = bool(mm_kwargs.get("use_audio_in_video", False))
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )
        audios, images, videos = self._process_mm_info(messages, use_audio_in_video=use_audio_in_video)
        processor_kwargs = {key: value for key, value in mm_kwargs.items() if key != "use_audio_in_video"}
        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            audio=audios,
            return_tensors="pt",
            padding=False,
            use_audio_in_video=use_audio_in_video,
            **processor_kwargs,
        )
        return dict(inputs)

    def _pad_or_truncate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        position_ids: torch.Tensor,
        extra_inputs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        seq_len = input_ids.shape[0]
        if seq_len > self.max_length:
            if self.truncation == "error":
                raise ValueError(f"Offline DPO sample length {seq_len} exceeds max_length={self.max_length}.")
            if self.truncation not in {"left", "right"}:
                raise ValueError(f"Unsupported truncation mode for offline DPO: {self.truncation}")
            input_ids = _slice_sequence(input_ids, max_length=self.max_length, truncation=self.truncation)
            attention_mask = _slice_sequence(attention_mask, max_length=self.max_length, truncation=self.truncation)
            loss_mask = _slice_sequence(loss_mask, max_length=self.max_length, truncation=self.truncation)
            position_ids = _slice_sequence(position_ids, max_length=self.max_length, truncation=self.truncation)
            seq_len = self.max_length

        if seq_len < self.max_length:
            pad_len = self.max_length - seq_len
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0
            input_ids = _pad_sequence(input_ids, pad_len=pad_len, pad_value=pad_token_id)
            attention_mask = _pad_sequence(attention_mask, pad_len=pad_len, pad_value=0)
            loss_mask = _pad_sequence(loss_mask, pad_len=pad_len, pad_value=0)
            position_ids = _pad_sequence(position_ids, pad_len=pad_len, pad_value=0)

        return input_ids, attention_mask, position_ids, loss_mask, extra_inputs

    def _tokenize_sample(
        self,
        prompt_messages: list[dict[str, Any]],
        response: str,
        mm_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        full_messages = copy.deepcopy(prompt_messages)
        full_messages.append({"role": "assistant", "content": response})

        prompt_inputs = self._encode_messages(
            prompt_messages,
            add_generation_prompt=True,
            mm_kwargs=mm_kwargs,
        )
        full_inputs = self._encode_messages(
            full_messages,
            add_generation_prompt=False,
            mm_kwargs=mm_kwargs,
        )

        input_ids = _to_sequence_tensor(full_inputs.pop("input_ids"))
        attention_mask = _to_sequence_tensor(full_inputs.pop("attention_mask"))
        position_ids_raw = full_inputs.pop("position_ids", None)
        if position_ids_raw is None:
            raise ValueError("Qwen3-Omni processor output is missing required `position_ids`.")
        position_ids = _to_mrope_position_ids(position_ids_raw)

        prefix_len = int(_to_sequence_tensor(prompt_inputs["input_ids"]).shape[0])
        loss_mask = torch.zeros_like(attention_mask)
        loss_mask[prefix_len:] = attention_mask[prefix_len:]

        extra_inputs = {
            key: value for key, value in full_inputs.items() if key not in _SEQUENCE_KEYS and key != "position_ids"
        }

        input_ids, attention_mask, position_ids, loss_mask, extra_inputs = self._pad_or_truncate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            position_ids=position_ids,
            extra_inputs=extra_inputs,
        )

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
        if extra_inputs:
            result["multi_modal_inputs"] = extra_inputs
        return result

    def _build_sample(
        self,
        *,
        prompt_messages: list[dict[str, Any]],
        response: str,
        row: dict[str, Any],
        pair_uid: str,
        is_chosen: bool,
        score: float,
        extra_info: dict[str, Any],
        mm_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        sample = self._tokenize_sample(prompt_messages, response, mm_kwargs)
        sample.update(
            {
                "uid": pair_uid,
                "is_chosen": is_chosen,
                "sample_level_scores": torch.tensor([score], dtype=torch.float32),
                "raw_prompt": prompt_messages,
                "raw_response": response,
                "data_source": row.get("data_source", "finevideo/audio_caption"),
                "reward_model": row.get("reward_model", {"style": "rule", "ground_truth": row[self.answer_win_key]}),
                "extra_info": {**extra_info, "is_chosen": is_chosen},
            }
        )
        return sample

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = self.dataframe.iloc[item].to_dict()
        data_file = row.get(_DATA_FILE_KEY)
        prompt_messages = _drop_none_values(_normalize_nested(row[self.prompt_key]))
        prompt_messages = [_as_dict(message) for message in _as_list(prompt_messages)]
        prompt_messages = _resolve_message_media_paths(
            prompt_messages,
            data_file,
            use_file_uri=self.use_file_uri_for_media,
        )
        mm_kwargs = _resolve_mm_processor_kwargs(row, self.config)

        win_score = float(row.get(self.win_score_key, 1.0))
        lose_score = float(row.get(self.lose_score_key, 0.0))
        if win_score < lose_score:
            raise ValueError(f"Offline DPO row {item} has win_score < lose_score: {win_score} < {lose_score}")

        extra_info = {
            **_extra_info_dict(row.get("extra_info")),
            "index": int(item),
        }
        pair_uid = str(row.get("uid") or uuid.uuid4())
        chosen = self._build_sample(
            prompt_messages=prompt_messages,
            response=str(row[self.answer_win_key]),
            row=row,
            pair_uid=pair_uid,
            is_chosen=True,
            score=win_score,
            extra_info=extra_info,
            mm_kwargs=mm_kwargs,
        )
        rejected = self._build_sample(
            prompt_messages=prompt_messages,
            response=str(row[self.answer_lose_key]),
            row=row,
            pair_uid=pair_uid,
            is_chosen=False,
            score=lose_score,
            extra_info=extra_info,
            mm_kwargs=mm_kwargs,
        )
        return {
            QWEN3_OMNI_OFFLINE_DPO_PAIR_MARKER: True,
            "chosen": chosen,
            "rejected": rejected,
        }


def expand_qwen3_omni_offline_dpo_features(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded = []
    for feature in features:
        if not feature.get(QWEN3_OMNI_OFFLINE_DPO_PAIR_MARKER):
            expanded.append(feature)
            continue
        expanded.append(feature["chosen"])
        expanded.append(feature["rejected"])
    return expanded


def _pad_last_dim(values: list[torch.Tensor], pad_value: int | float = 0) -> list[torch.Tensor]:
    max_len = max(value.shape[-1] for value in values)
    padded = []
    for value in values:
        pad_len = max_len - value.shape[-1]
        if pad_len <= 0:
            padded.append(value)
            continue
        pad_shape = (*value.shape[:-1], pad_len)
        pad = torch.full(pad_shape, pad_value, dtype=value.dtype, device=value.device)
        padded.append(torch.cat((value, pad), dim=-1))
    return padded


def _collate_multi_modal_inputs(values: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({key for value in values for key in value})
    collated = {}
    for key in keys:
        key_values = [value[key] for value in values if key in value]
        if not key_values or not all(isinstance(value, torch.Tensor) for value in key_values):
            collated[key] = np.fromiter(key_values, dtype=object, count=len(key_values))
            continue

        if key in {"input_features", "feature_attention_mask"}:
            key_values = _pad_last_dim(key_values)
        collated[key] = torch.cat(key_values, dim=0)
    return collated


def qwen3_omni_offline_dpo_collate_fn(features):
    if features and isinstance(features[0], dict) and features[0].get(QWEN3_OMNI_OFFLINE_DPO_PAIR_MARKER):
        features = expand_qwen3_omni_offline_dpo_features(features)
    multi_modal_inputs = [feature.pop("multi_modal_inputs", None) for feature in features]
    batch = _upstream_collate_fn(features)
    if any(value is not None for value in multi_modal_inputs):
        batch["multi_modal_inputs"] = _collate_multi_modal_inputs(
            [value for value in multi_modal_inputs if value is not None]
        )
    return batch
