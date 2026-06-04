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

"""Incremental parquet writing with resume and crash-safe checkpoints."""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def parquet_tmp_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp")


def discard_incomplete_parquet_tmp(output_path: Path) -> None:
    tmp_path = parquet_tmp_path(output_path)
    if tmp_path.exists():
        print(f"Discarding incomplete temporary parquet: {tmp_path}")
        tmp_path.unlink()


def read_parquet_table(output_path: Path) -> pa.Table:
    """Read a parquet checkpoint, recovering from interrupted writes when possible."""
    discard_incomplete_parquet_tmp(output_path)
    try:
        return pq.read_table(output_path)
    except (pa.ArrowInvalid, OSError) as exc:
        corrupted_path = output_path.with_name(f"{output_path.name}.corrupted.{int(time.time())}")
        output_path.replace(corrupted_path)
        raise ValueError(
            f"{output_path} appears corrupted (likely interrupted while writing parquet). "
            f"Renamed to {corrupted_path}. Re-run with --resume to continue from the last "
            f"successfully completed checkpoint, or restore that file from backup before resuming."
        ) from exc


def row_prompt_index(extra_info: Any, row_idx: int) -> int:
    if isinstance(extra_info, dict) and "index" in extra_info:
        return int(extra_info["index"])
    return row_idx


class ChunkedParquetWriter:
    """Write parquet rows incrementally to bound peak memory usage."""

    def __init__(self, path: Path, flush_every: int, base_table: pa.Table | None = None):
        self.path = path
        self._tmp_path = parquet_tmp_path(path)
        self.flush_every = max(1, flush_every)
        self._chunk: list[dict[str, Any]] = []
        self._writer: pq.ParquetWriter | None = None
        self._base_table = base_table
        self.row_count = base_table.num_rows if base_table is not None else 0
        self._closed = False
        self._dirty = False
        discard_incomplete_parquet_tmp(path)

    def __enter__(self) -> ChunkedParquetWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def write(self, row: dict[str, Any]) -> None:
        self._chunk.append(row)
        self.row_count += 1
        self._dirty = True
        if len(self._chunk) >= self.flush_every:
            self._flush()

    def commit_checkpoint(self) -> None:
        """Flush buffered rows and atomically update the output parquet."""
        if self._closed:
            return
        self._finalize_to_disk(announce=True)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._dirty or self._writer is not None or self._chunk:
            self._finalize_to_disk(announce=False)
        elif self.row_count == 0 and not self.path.exists():
            pd.DataFrame().to_parquet(self._tmp_path)
            os.replace(self._tmp_path, self.path)

    def _finalize_to_disk(self, *, announce: bool) -> None:
        if self._chunk:
            self._flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            os.replace(self._tmp_path, self.path)
            self._base_table = pq.read_table(self.path)
            self._dirty = False
            if announce:
                print(f"Checkpoint saved: {self.row_count} rows in {self.path}", flush=True)
            return

        if self._base_table is not None and not self._dirty:
            return

        if self.row_count == 0 and not self.path.exists():
            pd.DataFrame().to_parquet(self._tmp_path)
            os.replace(self._tmp_path, self.path)
            self._dirty = False

    def _flush(self) -> None:
        if not self._chunk:
            return
        table = pa.Table.from_pandas(pd.DataFrame(self._chunk), preserve_index=False)
        if self._writer is None:
            if self._base_table is not None:
                self._writer = pq.ParquetWriter(self._tmp_path, self._base_table.schema)
                self._writer.write_table(self._base_table)
                self._base_table = None
            else:
                self._writer = pq.ParquetWriter(self._tmp_path, table.schema)
        self._writer.write_table(table)
        self._chunk.clear()


def _shutdown_signals() -> list[int]:
    signals = [signal.SIGINT]
    if hasattr(signal, "SIGTERM"):
        signals.append(signal.SIGTERM)
    return signals


class ParquetWriterShutdownGuard:
    """Close the parquet writer on SIGINT/SIGTERM so checkpoints stay readable."""

    def __init__(self, writer: ChunkedParquetWriter):
        self._writer = writer
        self._previous_handlers: dict[int, Any] = {}

    def __enter__(self) -> ParquetWriterShutdownGuard:
        for sig in _shutdown_signals():
            self._previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle_signal)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for sig, handler in self._previous_handlers.items():
            signal.signal(sig, handler)

    def _handle_signal(self, signum: int, _frame) -> None:
        print(f"\nReceived signal {signum}, saving parquet checkpoint before exit...", flush=True)
        try:
            self._writer.commit_checkpoint()
        except Exception as exc:
            print(f"Failed to save parquet checkpoint: {exc}", flush=True)
            raise SystemExit(128 + signum) from exc
        raise SystemExit(128 + signum)


def load_resume_checkpoint(output_path: Path) -> tuple[int, pa.Table | None]:
    """Return the next prompt index and any parquet rows to keep when resuming."""
    if not output_path.exists() or output_path.stat().st_size == 0:
        return 0, None

    try:
        table = read_parquet_table(output_path)
    except ValueError as exc:
        print(f"Warning: {exc}")
        return 0, None
    num_rows = table.num_rows
    if num_rows == 0:
        return 0, None

    try:
        extra_infos = table["extra_info"].to_pylist()
    except (KeyError, OSError, pa.ArrowInvalid):
        return num_rows, table

    has_prompt_index = any(isinstance(info, dict) and "index" in info for info in extra_infos)
    if not has_prompt_index:
        return num_rows, table

    row_indices = [row_prompt_index(info, row_idx) for row_idx, info in enumerate(extra_infos)]
    index_set = set(row_indices)
    start_idx = 0
    while start_idx in index_set:
        start_idx += 1

    keep_mask = [row_index < start_idx for row_index in row_indices]
    if all(keep_mask):
        return start_idx, table

    kept_row_indices = [row_idx for row_idx, keep in enumerate(keep_mask) if keep]
    removed = num_rows - len(kept_row_indices)
    base_table = table.take(kept_row_indices) if kept_row_indices else None
    print(f"Truncated {output_path}: removed {removed} stale row(s) with prompt index >= {start_idx} before resuming.")
    return start_idx, base_table
