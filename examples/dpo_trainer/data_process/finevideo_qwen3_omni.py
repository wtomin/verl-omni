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
"""Build FineVideo audio+image caption preference data for Qwen3-Omni.

The script streams FineVideo samples, extracts activity-level clips into one
16 kHz mono WAV plus N chronological JPG frames, and writes parquet rows whose
chat messages use independent ``audio`` and ``image`` content blocks. It does
not store or reference mp4 files in the final parquet.

Example smoke test:

    python3 examples/dpo_trainer/data_process/finevideo_qwen3_omni.py \
      --output_dir data/finevideo_qwen3_omni \
      --max_videos 50 \
      --max_segments 50
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import pandas as pd

MIN_PIXELS = 128 * 28 * 28
MAX_PIXELS = 768 * 28 * 28
CAPTION_QUESTION = (
    "The audio and images above are from the same video clip, and the images are shown in chronological order. "
    "Based on the audio and images, describe what is happening in this clip."
)
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


@dataclass(frozen=True)
class SegmentCandidate:
    video_id: str
    segment_id: str
    description: str
    start_s: float
    end_s: float
    start_timestamp: str
    end_timestamp: str
    scene_title: str
    category: str
    youtube_channel: str | None
    source_index: int
    activity_index: int


@dataclass(frozen=True)
class PreparedSegment:
    candidate: SegmentCandidate
    split: str
    audio_relpath: str
    image_relpaths: list[str]
    frame_names: list[str]


def _parse_json_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, bytes | bytearray):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return json.loads(value)
    raise TypeError(f"Unsupported FineVideo json metadata type: {type(value)}")


def _timestamp_to_seconds(value: Any) -> float:
    if isinstance(value, int | float):
        return float(value)
    if value is None:
        raise ValueError("Missing timestamp.")
    text = str(value).strip()
    if not text:
        raise ValueError("Empty timestamp.")
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return float(text)

    parts = text.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = "0"
        minutes, seconds = parts
    else:
        raise ValueError(f"Unsupported timestamp format: {text}")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _format_timestamp(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    total_seconds, ms = divmod(millis, 1000)
    minutes, sec = divmod(total_seconds, 60)
    hours, minute = divmod(minutes, 60)
    return f"{hours:02d}:{minute:02d}:{sec:02d}.{ms:03d}"


def _extract_timestamp_range(timestamp: Any) -> tuple[float, float, str, str]:
    if isinstance(timestamp, dict):
        start_raw = timestamp.get("start_timestamp") or timestamp.get("start") or timestamp.get("start_time")
        end_raw = timestamp.get("end_timestamp") or timestamp.get("end") or timestamp.get("end_time")
    elif isinstance(timestamp, list | tuple) and len(timestamp) == 2:
        start_raw, end_raw = timestamp
    elif isinstance(timestamp, str) and re.search(r"\s+-\s+|\s+to\s+", timestamp, flags=re.IGNORECASE):
        parts = re.split(r"\s+-\s+|\s+to\s+", timestamp, maxsplit=1, flags=re.IGNORECASE)
        start_raw, end_raw = parts[0], parts[1]
    else:
        raise ValueError(f"Unsupported timestamp range: {timestamp}")

    start_s = _timestamp_to_seconds(start_raw)
    end_s = _timestamp_to_seconds(end_raw)
    if end_s <= start_s:
        raise ValueError(f"Invalid timestamp range: {timestamp}")
    return start_s, end_s, _format_timestamp(start_s), _format_timestamp(end_s)


def _safe_path_component(value: str, fallback: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())[:120].strip("._-")
    return safe or fallback


def _video_id_from_metadata(meta: dict[str, Any], source_index: int) -> str:
    for key in ("video_id", "youtube_id", "id"):
        value = meta.get(key)
        if value:
            return _safe_path_component(str(value), f"video_{source_index:06d}")
    filename = meta.get("original_json_filename") or meta.get("original_mp4_filename") or meta.get("filename")
    if filename:
        return _safe_path_component(Path(str(filename)).stem, f"video_{source_index:06d}")
    return f"video_{source_index:06d}"


def _content_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    content = meta.get("content_metadata")
    return content if isinstance(content, dict) else meta


def _iter_activity_candidates(
    meta: dict[str, Any],
    *,
    source_index: int,
    min_duration: float,
    max_duration: float,
    max_segments_per_video: int,
    require_speech: bool,
) -> list[SegmentCandidate]:
    content = _content_metadata(meta)
    scenes = content.get("scenes") or []
    if not isinstance(scenes, list):
        return []

    video_id = _video_id_from_metadata(meta, source_index)
    category = str(meta.get("content_parent_category") or meta.get("category") or "")
    youtube_channel = meta.get("youtube_channel")
    candidates = []
    activity_index = 0
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        scene_title = str(scene.get("title") or scene.get("scene_title") or "")
        for activity in scene.get("activities") or []:
            if not isinstance(activity, dict):
                continue
            description = str(activity.get("description") or "").strip()
            if len(description) <= 10:
                continue
            try:
                start_s, end_s, start_ts, end_ts = _extract_timestamp_range(activity.get("timestamp"))
            except (TypeError, ValueError):
                continue
            duration = end_s - start_s
            if duration < min_duration or duration > max_duration:
                continue
            if require_speech and not _has_overlapping_speech(content, start_s, end_s):
                continue

            segment_id = f"act_{activity_index:03d}"
            candidates.append(
                SegmentCandidate(
                    video_id=video_id,
                    segment_id=segment_id,
                    description=description,
                    start_s=start_s,
                    end_s=end_s,
                    start_timestamp=start_ts,
                    end_timestamp=end_ts,
                    scene_title=scene_title,
                    category=category,
                    youtube_channel=str(youtube_channel) if youtube_channel is not None else None,
                    source_index=source_index,
                    activity_index=activity_index,
                )
            )
            activity_index += 1
            if 0 < max_segments_per_video <= len(candidates):
                return candidates
    return candidates


def _has_overlapping_speech(content: dict[str, Any], start_s: float, end_s: float) -> bool:
    speech_items = content.get("timecoded_text_to_speech") or content.get("timecodedTextToSpeech") or []
    if isinstance(speech_items, dict):
        speech_items = speech_items.get("segments") or speech_items.get("items") or []
    if not isinstance(speech_items, list):
        return False
    for item in speech_items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or item.get("description") or "").strip()
        if not text:
            continue
        timestamp = item.get("timestamp") or item
        try:
            speech_start, speech_end, _, _ = _extract_timestamp_range(timestamp)
        except (TypeError, ValueError):
            continue
        if speech_start < end_s and speech_end > start_s:
            return True
    return False


def _write_video_payload(video_payload: Any, dst: Path) -> None:
    if isinstance(video_payload, bytes | bytearray):
        dst.write_bytes(video_payload)
        return
    if isinstance(video_payload, dict):
        if video_payload.get("bytes") is not None:
            dst.write_bytes(video_payload["bytes"])
            return
        if video_payload.get("path"):
            shutil.copyfile(os.path.expanduser(str(video_payload["path"])), dst)
            return
    if isinstance(video_payload, str):
        shutil.copyfile(os.path.expanduser(video_payload), dst)
        return
    if hasattr(video_payload, "read"):
        dst.write_bytes(video_payload.read())
        return
    raise TypeError(f"Unsupported FineVideo mp4 payload type: {type(video_payload)}")


def _extract_audio(video_path: Path, output_path: Path, start_s: float, end_s: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_s:.3f}",
        "-to",
        f"{end_s:.3f}",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-y",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def _extract_frames(video_path: Path, output_dir: Path, start_s: float, end_s: float, num_frames: int) -> list[str]:
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_names = []
    timestamps = np.linspace(start_s, end_s, num_frames + 2, dtype=np.float64)[1:-1]
    try:
        for frame_idx, timestamp_s in enumerate(timestamps):
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_s * 1000.0))
            ok, frame_bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Could not read frame at {timestamp_s:.3f}s from {video_path}")
            frame_name = f"frame_{frame_idx:02d}.jpg"
            frame_path = output_dir / frame_name
            if not cv2.imwrite(str(frame_path), frame_bgr):
                raise RuntimeError(f"Could not write frame: {frame_path}")
            frame_names.append(frame_name)
    finally:
        capture.release()
    return frame_names


def _assign_split(video_id: str, test_ratio: float, seed: int) -> str:
    digest = hashlib.sha1(f"{seed}:{video_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "test" if bucket < test_ratio else "train"


def _relative_path(path: Path, base_dir: Path) -> str:
    return path.relative_to(base_dir).as_posix()


def _prepare_segment(
    video_path: Path,
    candidate: SegmentCandidate,
    *,
    output_dir: Path,
    split: str,
    num_frames: int,
    skip_existing: bool,
) -> PreparedSegment:
    segment_dir = output_dir / split / candidate.video_id / candidate.segment_id
    audio_path = segment_dir / "audio.wav"
    if not (skip_existing and audio_path.exists()):
        _extract_audio(video_path, audio_path, candidate.start_s, candidate.end_s)

    expected_frames = [f"frame_{idx:02d}.jpg" for idx in range(num_frames)]
    if skip_existing and all((segment_dir / name).exists() for name in expected_frames):
        frame_names = expected_frames
    else:
        frame_names = _extract_frames(video_path, segment_dir, candidate.start_s, candidate.end_s, num_frames)

    meta_path = segment_dir / "meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "video_id": candidate.video_id,
                "segment_id": candidate.segment_id,
                "start_timestamp": candidate.start_timestamp,
                "end_timestamp": candidate.end_timestamp,
                "description": candidate.description,
                "scene_title": candidate.scene_title,
                "category": candidate.category,
                "youtube_channel": candidate.youtube_channel,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return PreparedSegment(
        candidate=candidate,
        split=split,
        audio_relpath=_relative_path(audio_path, output_dir),
        image_relpaths=[_relative_path(segment_dir / name, output_dir) for name in frame_names],
        frame_names=frame_names,
    )


def _build_prompt(audio_relpath: str, image_relpaths: list[str], question: str) -> list[dict[str, Any]]:
    if CJK_RE.search(question):
        raise ValueError(f"Question must be English-only, got CJK characters: {question}")
    content = [{"type": "audio", "audio": audio_relpath}]
    content.extend(
        {
            "type": "image",
            "image": image_path,
            "min_pixels": MIN_PIXELS,
            "max_pixels": MAX_PIXELS,
        }
        for image_path in image_relpaths
    )
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


def _choose_negative(segment: PreparedSegment, all_segments: list[PreparedSegment]) -> str | None:
    same_video = [
        item.candidate.description
        for item in all_segments
        if item.candidate.video_id == segment.candidate.video_id
        and item.candidate.segment_id != segment.candidate.segment_id
        and item.candidate.description != segment.candidate.description
    ]
    if same_video:
        return same_video[0]

    same_category = [
        item.candidate.description
        for item in all_segments
        if item.candidate.category
        and item.candidate.category == segment.candidate.category
        and item.candidate.video_id != segment.candidate.video_id
        and item.candidate.description != segment.candidate.description
    ]
    if same_category:
        return same_category[0]

    for item in all_segments:
        if item.candidate.description != segment.candidate.description:
            return item.candidate.description
    return None


def _build_row(segment: PreparedSegment, negative_description: str, row_index: int) -> dict[str, Any]:
    candidate = segment.candidate
    return {
        "data_source": "finevideo/audio_caption",
        "ability": "audio_caption",
        "prompt": _build_prompt(segment.audio_relpath, segment.image_relpaths, CAPTION_QUESTION),
        "images": segment.image_relpaths,
        "audios": [segment.audio_relpath],
        "mm_processor_kwargs": {"use_audio_in_video": False},
        "answer_win": candidate.description,
        "answer_lose": negative_description,
        "win_score": 1.0,
        "lose_score": 0.0,
        "reward_model": {"style": "rule", "ground_truth": candidate.description},
        "extra_info": {
            "video_id": candidate.video_id,
            "segment_id": candidate.segment_id,
            "segment_type": "activity",
            "start_timestamp": candidate.start_timestamp,
            "end_timestamp": candidate.end_timestamp,
            "scene_title": candidate.scene_title,
            "description": candidate.description,
            "num_frames": len(segment.image_relpaths),
            "frame_order": segment.frame_names,
            "negative_descriptions": [negative_description],
            "split": segment.split,
            "index": row_index,
            "source_index": candidate.source_index,
            "activity_index": candidate.activity_index,
            "content_parent_category": candidate.category,
            "youtube_channel": candidate.youtube_channel,
        },
    }


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"Wrote {len(rows)} rows to {path}")


def _load_dataset(args: argparse.Namespace):
    kwargs = {"split": args.split, "streaming": args.streaming}
    if args.dataset_config:
        return datasets.load_dataset(args.dataset_name, args.dataset_config, **kwargs)
    return datasets.load_dataset(args.dataset_name, **kwargs)


def build_dataset(args: argparse.Namespace) -> None:
    output_dir = Path(os.path.expanduser(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = _load_dataset(args)

    prepared_segments: list[PreparedSegment] = []
    processed_videos = 0
    failed_segments = 0
    with tempfile.TemporaryDirectory(prefix="finevideo_qwen3_omni_") as tmpdir:
        tmp_video_path = Path(tmpdir) / "source.mp4"
        for source_index, sample in enumerate(dataset):
            if 0 <= args.max_videos <= processed_videos:
                break
            try:
                meta = _parse_json_metadata(sample[args.json_key])
                candidates = _iter_activity_candidates(
                    meta,
                    source_index=source_index,
                    min_duration=args.min_duration,
                    max_duration=args.max_duration,
                    max_segments_per_video=args.max_segments_per_video,
                    require_speech=args.require_speech,
                )
                if not candidates:
                    continue
                _write_video_payload(sample[args.video_key], tmp_video_path)
            except Exception as exc:
                print(f"Skipping source sample {source_index}: {exc}")
                continue

            processed_videos += 1
            for candidate in candidates:
                if 0 <= args.max_segments <= len(prepared_segments):
                    break
                split = _assign_split(candidate.video_id, args.test_ratio, args.seed)
                try:
                    prepared_segments.append(
                        _prepare_segment(
                            tmp_video_path,
                            candidate,
                            output_dir=output_dir,
                            split=split,
                            num_frames=args.num_frames,
                            skip_existing=args.skip_existing,
                        )
                    )
                except Exception as exc:
                    failed_segments += 1
                    print(f"Skipping {candidate.video_id}/{candidate.segment_id}: {exc}")
            tmp_video_path.unlink(missing_ok=True)
            if 0 <= args.max_segments <= len(prepared_segments):
                break

    train_rows = []
    test_rows = []
    skipped_without_negative = 0
    for row_index, segment in enumerate(prepared_segments):
        negative_description = _choose_negative(segment, prepared_segments)
        if negative_description is None:
            skipped_without_negative += 1
            continue
        row = _build_row(segment, negative_description, row_index)
        if segment.split == "test":
            test_rows.append(row)
        else:
            train_rows.append(row)

    _write_parquet(train_rows, output_dir / "train.parquet")
    _write_parquet(test_rows, output_dir / "test.parquet")
    print(
        "Finished FineVideo preprocessing: "
        f"{processed_videos} videos, {len(prepared_segments)} extracted segments, "
        f"{failed_segments} failed segments, {skipped_without_negative} skipped without negatives."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess FineVideo for Qwen3-Omni audio+image DPO/caption data.")
    parser.add_argument("--dataset_name", default="HuggingFaceFV/finevideo")
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video_key", default="mp4")
    parser.add_argument("--json_key", default="json")
    parser.add_argument("--output_dir", default="~/data/finevideo_qwen3_omni")
    parser.add_argument("--max_videos", type=int, default=50, help="Source videos to inspect; -1 means no limit.")
    parser.add_argument("--max_segments", type=int, default=50, help="Extracted segments to keep; -1 means no limit.")
    parser.add_argument("--max_segments_per_video", type=int, default=3)
    parser.add_argument("--num_frames", type=int, default=6)
    parser.add_argument("--min_duration", type=float, default=3.0)
    parser.add_argument("--max_duration", type=float, default=30.0)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require_speech", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    if args.num_frames < 1:
        raise ValueError("--num_frames must be positive.")
    if not 0.0 <= args.test_ratio < 1.0:
        raise ValueError("--test_ratio must be in [0, 1).")
    build_dataset(args)


if __name__ == "__main__":
    main()
