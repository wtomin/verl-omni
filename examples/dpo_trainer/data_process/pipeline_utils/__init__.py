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

"""Registry for offline DPO pipeline-specific tensor utilities."""

import argparse
from types import ModuleType

from . import qwen_image, sd3

_PIPELINE_UTILS = {
    sd3.PIPELINE_KEY: sd3,
    qwen_image.PIPELINE_KEY: qwen_image,
}


def available_pipeline_keys() -> list[str]:
    return sorted(_PIPELINE_UTILS)


def resolve_pipeline_key(pipeline: str, model_path: str) -> str:
    if pipeline != "auto":
        if pipeline not in _PIPELINE_UTILS:
            raise ValueError(f"Unknown pipeline {pipeline!r}. Available: {available_pipeline_keys()}")
        return pipeline

    normalized_model_path = model_path.lower()
    if "qwen" in normalized_model_path:
        return qwen_image.PIPELINE_KEY
    if (
        model_path == sd3.DEFAULT_MODEL_PATH
        or "stable-diffusion-3" in normalized_model_path
        or "sd3" in normalized_model_path
    ):
        return sd3.PIPELINE_KEY

    raise ValueError(
        f"Could not infer offline DPO pipeline from model_path={model_path!r}. "
        f"Set --pipeline explicitly to one of: {available_pipeline_keys()}."
    )


def get_pipeline_utils(args: argparse.Namespace) -> ModuleType:
    pipeline_key = resolve_pipeline_key(args.pipeline, args.model_path)
    args.pipeline = pipeline_key
    utils = _PIPELINE_UTILS[pipeline_key]
    utils.apply_arg_defaults(args)
    return utils
