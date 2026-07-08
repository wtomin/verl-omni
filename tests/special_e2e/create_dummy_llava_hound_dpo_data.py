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
"""Create a tiny LLaVA-Hound-DPO-style multimodal parquet dataset.

The generated data mirrors the offline MLLM DPO schema used by
``llava_hound_dpo_multisource.py``:

    prompt, chosen, rejected, win_score, lose_score, data_source, ability,
    reward_model, extra_info

It intentionally writes three source folders (image/text/video) so the e2e smoke
test exercises mixed multimodal batches without downloading the real dataset.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
from PIL import Image

SYSTEM_PROMPT = "You are a helpful assistant."


def _content_item(item_type: str, *, text: str | None = None, image="", video="", audio="") -> dict:
    return {"type": item_type, "text": text, "image": image, "video": video, "audio": audio}


def _write_png(path: str, color: tuple[int, int, int]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (64, 64), color=color).save(path)
    return os.path.abspath(path)


def _make_assets(root: str) -> dict[str, str | list[str]]:
    media_dir = os.path.join(root, "media")
    image_path = _write_png(os.path.join(media_dir, "images", "dummy_image.png"), (240, 64, 64))

    video_frame_dir = os.path.join(media_dir, "videos", "dummy_clip")
    frame_paths = [
        _write_png(os.path.join(video_frame_dir, f"frame_{idx:03d}.png"), color)
        for idx, color in enumerate([(64, 128, 240), (64, 200, 120), (220, 180, 64)])
    ]
    return {"image": image_path, "video": frame_paths}


def _base_row(split: str, source: str, index: int, question: str) -> dict:
    return {
        "data_source": f"llava_hound_dpo/{source}",
        "chosen": "The preferred answer correctly describes the dummy content.",
        "rejected": "The rejected answer gives an unrelated description.",
        "win_score": 1.0,
        "lose_score": 0.0,
        "ability": f"{source}_qa",
        "reward_model": {"style": "preference"},
        "extra_info": {
            "split": split,
            "index": index,
            "sample_id": f"{source}_{split}_{index}",
            "question": question,
            "source_video": "dummy_clip",
            "source_video_name": "dummy_clip",
        },
    }


def _image_row(split: str, index: int, image_path: str) -> dict:
    question = "What is shown in this dummy image?"
    row = _base_row(split, "image", index, question)
    row["prompt"] = [
        {"role": "system", "content": [_content_item("text", text=SYSTEM_PROMPT)]},
        {
            "role": "user",
            "content": [
                _content_item("image", image=image_path),
                _content_item("text", text=question),
            ],
        },
    ]
    row["extra_info"]["image_path"] = image_path
    return row


def _text_row(split: str, index: int) -> dict:
    question = "Answer this dummy text-only preference question."
    row = _base_row(split, "text", index, question)
    row["prompt"] = [
        {"role": "system", "content": [_content_item("text", text=SYSTEM_PROMPT)]},
        {"role": "user", "content": [_content_item("text", text=question)]},
    ]
    return row


def _video_row(split: str, index: int, frame_paths: list[str]) -> dict:
    question = "What changes across the dummy video frames?"
    row = _base_row(split, "video", index, question)
    row["prompt"] = [
        {"role": "system", "content": [_content_item("text", text=SYSTEM_PROMPT)]},
        {
            "role": "user",
            "content": [
                _content_item("video", video=os.path.dirname(frame_paths[0])),
                _content_item("text", text=question),
            ],
        },
    ]
    row["extra_info"]["video_frame_dir"] = os.path.dirname(frame_paths[0])
    row["extra_info"]["video_frame_count"] = len(frame_paths)
    return row


def _build_rows(split: str, size: int, assets: dict[str, str | list[str]]) -> dict[str, list[dict]]:
    rows = {"image": [], "text": [], "video": []}
    image_path = str(assets["image"])
    frame_paths = list(assets["video"])
    for idx in range(size):
        rows["image"].append(_image_row(split, idx, image_path))
        rows["text"].append(_text_row(split, idx))
        rows["video"].append(_video_row(split, idx, frame_paths))
    return rows


def _write_source_parquet(root: str, source: str, train_rows: list[dict], val_rows: list[dict]) -> None:
    source_dir = os.path.join(root, source)
    os.makedirs(source_dir, exist_ok=True)
    pd.DataFrame(train_rows).to_parquet(os.path.join(source_dir, "train.parquet"), index=False)
    pd.DataFrame(val_rows).to_parquet(os.path.join(source_dir, "test.parquet"), index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dummy LLaVA-Hound-DPO parquet data for e2e testing")
    parser.add_argument(
        "--local_save_dir",
        default=os.path.expanduser("~/data/dummy_llava_hound_dpo"),
        help="Directory to write source parquet folders and media files",
    )
    parser.add_argument("--train_size", type=int, default=4, help="Rows per source for train split")
    parser.add_argument("--val_size", type=int, default=2, help="Rows per source for val split")
    args = parser.parse_args()

    root = os.path.abspath(os.path.expanduser(args.local_save_dir))
    assets = _make_assets(root)
    train_rows = _build_rows("train", args.train_size, assets)
    val_rows = _build_rows("test", args.val_size, assets)

    for source in ("image", "text", "video"):
        _write_source_parquet(root, source, train_rows[source], val_rows[source])
        print(
            f"Wrote {len(train_rows[source])} train and {len(val_rows[source])} val rows for {source}",
            flush=True,
        )

    print(f"Dummy LLaVA-Hound-DPO data root: {root}", flush=True)


if __name__ == "__main__":
    main()
