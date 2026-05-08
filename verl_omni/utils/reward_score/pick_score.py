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
"""
PickScore reward model adapter for verl-omni DPO training.
"""

from typing import Optional

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def _solution_image_to_pil(solution_image) -> Image.Image:
    if isinstance(solution_image, Image.Image):
        return solution_image
    if isinstance(solution_image, torch.Tensor):
        t = solution_image.detach().cpu().float()
        if t.ndim == 4:
            t = t[0]
        if t.ndim != 3:
            raise ValueError(f"Expected image tensor (C,H,W) or (N,C,H,W), got shape {tuple(t.shape)}")
        if t.shape[0] in (1, 3):
            t = t.clamp(0, 1) if t.max() <= 2.0 else t / 255.0
            arr = (t * 255.0).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
            return Image.fromarray(arr.squeeze(-1) if arr.shape[-1] == 1 else arr)
        raise ValueError(f"Expected 1 or 3 channel image (C,H,W), got C={t.shape[0]}")
    raise TypeError(f"Unsupported image type: {type(solution_image)}")


class PickScoreRewardModel:
    """PickScore reward model for scoring image-prompt pairs.

    Uses CLIP ViT-H-14 to compute similarity between text and image embeddings.
    Scores are normalized to [0, 1] range by dividing by 26.

    Reference: https://arxiv.org/abs/2304.11397
    """

    def __init__(
        self,
        device: str = "cuda",
        dtype: str = "bfloat16",
        batch_size: int = 64,
    ):
        self.device = torch.device(device)
        self.batch_size = batch_size

        # Load processor and model
        processor_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_path = "yuvalkirstain/PickScore_v1"

        self.processor = CLIPProcessor.from_pretrained(processor_path)
        self.model = CLIPModel.from_pretrained(model_path).eval().to(self.device)

        # Set dtype
        if dtype == "float16":
            self.model = self.model.half()
        elif dtype == "bfloat16":
            self.model = self.model.to(torch.bfloat16)

        # Get logit scale
        self.logit_scale = self.model.logit_scale.exp()

    def compute_scores(
        self,
        prompts: list[str],
        images: list[Image.Image],
    ) -> torch.Tensor:
        """Compute PickScore for a batch of (prompt, image) pairs.

        Args:
            prompts: List of text prompts.
            images: List of PIL Images.

        Returns:
            Tensor of scores (normalized to [0, 1] range).
        """
        # Process images
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        image_inputs = {k: v.to(device=self.device) for k, v in image_inputs.items()}

        # Process texts
        text_inputs = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(device=self.device) for k, v in text_inputs.items()}

        # Get embeddings
        image_embs = self.model.get_image_features(**image_inputs)
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)

        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)

        # Compute scores
        scores = self.logit_scale * (text_embs * image_embs).sum(dim=-1)

        # Normalize to [0, 1] range
        scores = scores / 26.0

        return scores

    def __call__(
        self,
        prompts: list[str],
        images: list[Image.Image],
    ) -> list[float]:
        """Score a list of (prompt, image) pairs.

        Args:
            prompts: List of text prompts.
            images: List of PIL Images (must match length of prompts).

        Returns:
            List of reward scores.
        """
        assert len(prompts) == len(images), "Number of prompts and images must match"

        all_scores = []
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i : i + self.batch_size]
            batch_images = images[i : i + self.batch_size]

            scores = self.compute_scores(batch_prompts, batch_images)
            all_scores.extend(scores.cpu().tolist())

        return all_scores


_pickscore_model: Optional[PickScoreRewardModel] = None


def get_pickscore_reward_model(
    device: str = "cuda",
    dtype: str = "bfloat16",
    batch_size: int = 64,
) -> PickScoreRewardModel:
    """Return a process-wide cached PickScore model (lazy init)."""
    global _pickscore_model
    if _pickscore_model is None:
        _pickscore_model = PickScoreRewardModel(device=device, dtype=dtype, batch_size=batch_size)
    return _pickscore_model


def compute_pickscore_reward(
    data_source: str,
    solution_image: Image.Image,
    ground_truth: Optional[str] = None,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> dict:
    """Compute PickScore reward for verl-omni reward manager.

    Args:
        data_source: Source of the data.
        solution_image: Generated image.
        ground_truth: Ground truth text (optional).
        extra_info: Extra information dictionary.
        **kwargs: Additional keyword arguments.

    Returns:
        Dictionary with score and extra info.
    """
    # This is a simplified version - in practice, you'd use the prompt from the data
    prompt = ground_truth or (extra_info or {}).get("prompt", "")

    if not prompt:
        return {"score": 0.0, "acc": 0.0}

    image_pil = _solution_image_to_pil(solution_image)
    reward_model = get_pickscore_reward_model()
    score = reward_model([prompt], [image_pil])[0]

    return {"score": float(score), "acc": float(score)}
