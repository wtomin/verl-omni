#!/usr/bin/env bash
# Download and unpack LLaVA-Hound-DPO annotations and video shards.
#
# Usage:
#   bash examples/dpo_trainer/data_process/download_llava_hound_dpo.sh
#
# Optional:
#   DATA_DIR=/data/llava_hound_dpo bash examples/dpo_trainer/data_process/download_llava_hound_dpo.sh

set -euo pipefail

DATA_DIR=${DATA_DIR:-"$HOME/data/llava_hound_dpo"}
DPO_DIR="$DATA_DIR/annotations/dpo"
VIDEO_ZIP_DIR="$DATA_DIR/video_zip"
VIDEO_DIR="$DATA_DIR/videos"

ANNOTATION_URL="https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction/resolve/main/video_instruction/train/dpo/sft_dpo_17k.jsonl?download=true"
SHARD_BASE_URL="https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction/resolve/main/train_300k"

download_file() {
  local url=$1
  local output=$2

  if command -v wget >/dev/null 2>&1; then
    wget -c -O "$output" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -L -C - -o "$output" "$url"
  else
    echo "ERROR: wget or curl is required." >&2
    return 1
  fi
}

wait_for_jobs() {
  local failed=0
  local pid

  for pid in "$@"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done

  if [[ "$failed" -ne 0 ]]; then
    echo "ERROR: one or more background jobs failed." >&2
    exit 1
  fi
}

mkdir -p "$DPO_DIR" "$VIDEO_ZIP_DIR" "$VIDEO_DIR"

echo "Downloading DPO annotations..."
download_file "$ANNOTATION_URL" "$DPO_DIR/sft_dpo_17k.jsonl"

echo "Downloading 16 video shards..."
download_pids=()
for i in $(seq 0 15); do
  download_file \
    "$SHARD_BASE_URL/chunk_${i}.tar.gz?download=true" \
    "$VIDEO_ZIP_DIR/chunk_${i}.tar.gz" &
  download_pids+=("$!")
done
wait_for_jobs "${download_pids[@]}"
echo "All shards downloaded."

echo "Unpacking video shards..."
extract_pids=()
for chunk in "$VIDEO_ZIP_DIR"/chunk_*.tar.gz; do
  tar -xzf "$chunk" -C "$VIDEO_DIR" &
  extract_pids+=("$!")
done
wait_for_jobs "${extract_pids[@]}"
echo "All shards unpacked."

echo "Done."
echo "Annotation: $DPO_DIR/sft_dpo_17k.jsonl"
echo "Videos:     $VIDEO_DIR"
