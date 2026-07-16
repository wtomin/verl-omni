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

"""Compare nightly debug dumps against a baseline."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from tests.nightly.qwen_image_flowgrpo_single_sample.debug_dumper import parse_dump_steps


def _parse_step(path: Path) -> int | None:
    for part in path.parts:
        if part.startswith("step_"):
            try:
                return int(part.removeprefix("step_"))
            except ValueError:
                return None
    return None


def _safe_cosine_similarity(base: torch.Tensor, current: torch.Tensor) -> float:
    base_flat = base.float().flatten()
    current_flat = current.float().flatten()
    if base_flat.numel() == 0:
        return 1.0
    if float(base_flat.norm()) == 0.0 and float(current_flat.norm()) == 0.0:
        return 1.0
    return float(F.cosine_similarity(base_flat, current_flat, dim=0))


def compare_tensor(base: torch.Tensor, current: torch.Tensor, *, rtol: float, atol: float) -> dict[str, Any]:
    if tuple(base.shape) != tuple(current.shape):
        return {
            "status": "shape_mismatch",
            "pass": False,
            "baseline_shape": list(base.shape),
            "current_shape": list(current.shape),
        }

    base = base.float()
    current = current.float()
    diff = (base - current).abs()
    rel = diff / (base.abs() + atol)
    max_abs = float(diff.max()) if diff.numel() else 0.0
    mean_abs = float(diff.mean()) if diff.numel() else 0.0
    max_rel = float(rel.max()) if rel.numel() else 0.0
    mean_rel = float(rel.mean()) if rel.numel() else 0.0
    return {
        "status": "compared",
        "max_abs_err": max_abs,
        "mean_abs_err": mean_abs,
        "max_rel_err": max_rel,
        "mean_rel_err": mean_rel,
        "cos_sim": _safe_cosine_similarity(base, current),
        "pass": max_rel <= rtol or max_abs <= atol,
    }


def _compare_scalar(base: Any, current: Any, *, rtol: float, atol: float) -> dict[str, Any]:
    if isinstance(base, (int, float)) and isinstance(current, (int, float)):
        diff = abs(float(base) - float(current))
        rel = diff / (abs(float(base)) + atol)
        return {
            "baseline": base,
            "current": current,
            "abs_err": diff,
            "rel_err": rel,
            "pass": rel <= rtol or diff <= atol,
        }
    return {"baseline": base, "current": current, "pass": base == current}


def _compare_meta(
    base_meta: dict[str, Any],
    current_meta: dict[str, Any],
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    report: dict[str, Any] = {"status": "compared_meta", "fields": {}, "pass": True}
    keys = sorted(set(base_meta) | set(current_meta))
    for key in keys:
        if key not in current_meta:
            result = {"status": "missing_current", "pass": False}
        elif key not in base_meta:
            result = {"status": "missing_baseline", "pass": False}
        elif isinstance(base_meta[key], dict) and isinstance(current_meta[key], dict):
            result = _compare_meta(base_meta[key], current_meta[key], rtol=rtol, atol=atol)
        else:
            result = _compare_scalar(base_meta[key], current_meta[key], rtol=rtol, atol=atol)
        report["fields"][key] = result
        if not result.get("pass", False):
            report["pass"] = False
    return report


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _iter_compare_files(baseline_dir: Path, steps: set[int]) -> list[Path]:
    candidates = list(baseline_dir.rglob("tensors.pt"))
    if not candidates:
        candidates = list(baseline_dir.rglob("meta.json"))
    compare_files = []
    for path in candidates:
        step = _parse_step(path.relative_to(baseline_dir))
        if step is None or step in steps:
            compare_files.append(path)
    return compare_files


def compare_run(
    baseline_dir: Path,
    current_dir: Path,
    *,
    out_json: Path,
    steps: tuple[int, ...],
    rtol: float,
    atol: float,
) -> bool:
    report: dict[str, Any] = {
        "baseline_dir": str(baseline_dir),
        "current_dir": str(current_dir),
        "steps_compared": list(steps),
        "artifacts": {},
        "pass": True,
    }

    for baseline_path in _iter_compare_files(baseline_dir, set(steps)):
        rel = baseline_path.relative_to(baseline_dir)
        current_path = current_dir / rel
        artifact_report: dict[str, Any] = {"pass": True}
        if not current_path.exists():
            artifact_report = {"status": "missing_current_file", "pass": False}
        elif baseline_path.name == "tensors.pt":
            baseline_tensors = torch.load(baseline_path, map_location="cpu")
            current_tensors = torch.load(current_path, map_location="cpu")
            artifact_report["tensors"] = {}
            for key in sorted(set(baseline_tensors) | set(current_tensors)):
                if key not in current_tensors:
                    tensor_report = {"status": "missing_current_tensor", "pass": False}
                elif key not in baseline_tensors:
                    tensor_report = {"status": "missing_baseline_tensor", "pass": False}
                else:
                    tensor_report = compare_tensor(baseline_tensors[key], current_tensors[key], rtol=rtol, atol=atol)
                artifact_report["tensors"][key] = tensor_report
                artifact_report["pass"] = artifact_report["pass"] and tensor_report["pass"]
        else:
            artifact_report = _compare_meta(
                _load_json(baseline_path),
                _load_json(current_path),
                rtol=rtol,
                atol=atol,
            )
        report["artifacts"][str(rel)] = artifact_report
        report["pass"] = report["pass"] and artifact_report["pass"]

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True)
    return bool(report["pass"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=Path("outputs/debug_dumps/baseline"))
    parser.add_argument("--current-dir", type=Path, default=Path("outputs/debug_dumps/current"))
    parser.add_argument("--out-json", type=Path, default=Path("outputs/debug_dumps/current/compare_report.json"))
    parser.add_argument("--steps", default="1,2")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--bootstrap-if-missing", action="store_true")
    args = parser.parse_args()

    if args.bootstrap_if_missing and not args.baseline_dir.exists():
        if not args.current_dir.exists():
            raise FileNotFoundError(f"Current dump directory does not exist: {args.current_dir}")
        shutil.copytree(args.current_dir, args.baseline_dir)
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w", encoding="utf-8") as file:
            json.dump({"status": "bootstrapped", "pass": True, "baseline_dir": str(args.baseline_dir)}, file, indent=2)
        return

    if not args.baseline_dir.exists():
        raise FileNotFoundError(f"Baseline dump directory does not exist: {args.baseline_dir}")
    if not args.current_dir.exists():
        raise FileNotFoundError(f"Current dump directory does not exist: {args.current_dir}")

    passed = compare_run(
        args.baseline_dir,
        args.current_dir,
        out_json=args.out_json,
        steps=parse_dump_steps(args.steps),
        rtol=args.rtol,
        atol=args.atol,
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
