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

"""Registry for omni-model training-side adapters."""

from abc import ABC, abstractmethod
from typing import Any

import torch

from verl_omni.workers.config import OmniModelConfig

__all__ = ["OmniTrainingAdapterBase"]


class OmniTrainingAdapterBase(ABC):
    """Base class for architecture/algorithm-specific omni training adapters."""

    _registry: dict[tuple[str, str], type["OmniTrainingAdapterBase"]] = {}

    @classmethod
    def register(cls, architecture: str, algorithm: str):
        """Class decorator that registers a subclass for ``(architecture, algorithm)``."""

        def decorator(subclass: type["OmniTrainingAdapterBase"]) -> type["OmniTrainingAdapterBase"]:
            cls._registry[(architecture, algorithm)] = subclass
            return subclass

        return decorator

    @classmethod
    def get_class(cls, model_config: OmniModelConfig) -> type["OmniTrainingAdapterBase"]:
        """Return the registered adapter for the omni model architecture and algorithm."""
        architecture = model_config.architecture
        algorithm = model_config.algorithm
        key = (architecture, algorithm)
        if key not in cls._registry and model_config.external_lib is not None:
            from verl.utils.import_utils import import_external_libs

            import_external_libs(model_config.external_lib)
        try:
            return cls._registry[key]
        except KeyError:
            registered = sorted(cls._registry.keys())
            raise NotImplementedError(
                f"No omni training adapter registered for (architecture={architecture!r}, "
                f"algorithm={algorithm!r}). Registered: {registered}. "
                f"Set actor_rollout_ref.model.external_lib to load your adapter implementation."
            ) from None

    @classmethod
    @abstractmethod
    def prepare_model_inputs(cls, model_inputs: dict[str, Any], dtype: torch.dtype) -> dict[str, Any]:
        """Build architecture-specific model inputs for a VeOmni forward pass."""
        pass
