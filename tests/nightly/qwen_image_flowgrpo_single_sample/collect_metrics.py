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

"""Collect and compare nightly performance metrics from console and memory logs."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from tests.nightly.qwen_image_flowgrpo_single_sample.debug_dumper import parse_dump_steps

DEFAULT_METRIC_KEYS = (
    "timing_s/gen",
    "timing_s/step",
    "timing_per_image_ms/gen",
    "perf/throughput",
    "actor/loss",
    "actor/grad_norm",
)
LOWER_IS_BETTER = {"timing_s/gen", "timing_s/step", "timing_per_image_ms/gen", "memory/peak_gb"}
HIGHER_IS_BETTER = {"perf/throughput"}


def _parse_metric_dict_from_line(line: str) -> dict[str, Any] | None:
    stripped = line.strip()
    if "training/global_step" not in stripped:
        return None

    if stripped.startswith("{") and stripped.endswith("}"):
        candidates = [stripped]
    else:
        start = stripped.find("{")
        end = stripped.rfind("}")
        candidates = [stripped[start : end + 1]] if start >= 0 and end > start else []

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(candidate)
            except (SyntaxError, ValueError):
                continue
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_console_log(path: Path, metric_keys: tuple[str, ...] = DEFAULT_METRIC_KEYS) -> dict[int, dict[str, float]]:
    metrics_by_step: dict[int, dict[str, float]] = {}
    if not path.exists():
        return metrics_by_step

    key_pattern = "|".join(re.escape(key) for key in ("training/global_step", *metric_keys))
    fallback_pattern = re.compile(rf"({key_pattern})['\"]?\s*[:=]\s*([-+0-9.eE]+)")

    with path.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            metrics = _parse_metric_dict_from_line(line)
            if metrics is None:
                pairs = dict(fallback_pattern.findall(line))
                metrics = pairs if "training/global_step" in pairs else None
            if not metrics:
                continue

            try:
                step = int(float(metrics["training/global_step"]))
            except (KeyError, TypeError, ValueError):
                continue

            step_metrics = metrics_by_step.setdefault(step, {})
            for key in metric_keys:
                if key not in metrics:
                    continue
                try:
                    step_metrics[key] = float(metrics[key])
                except (TypeError, ValueError):
                    continue
    return metrics_by_step


def parse_memory_peak_gb(path: Path) -> float | None:
    if not path.exists():
        return None

    peak_mib = 0.0
    with path.open(encoding="utf-8", errors="replace", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) < 2 or "memory" in row[1].lower():
                continue
            match = re.search(r"([-+0-9.]+)", row[1])
            if match:
                peak_mib = max(peak_mib, float(match.group(1)))
    return peak_mib / 1024.0 if peak_mib else None


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))
    return ordered[index]


def _summarize(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": mean(values) if values else None,
        "p95": _p95(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def build_metrics_json(
    *,
    console_log: Path,
    memory_log: Path,
    precision_steps: tuple[int, ...],
    total_steps: int | None,
    n_gpus: int | None,
) -> dict[str, Any]:
    metrics_by_step = parse_console_log(console_log)
    perf_skip_steps = max(precision_steps) if precision_steps else 0
    sorted_steps = sorted(metrics_by_step)
    if total_steps is None and sorted_steps:
        total_steps = max(sorted_steps)

    all_steps = {key: [] for key in DEFAULT_METRIC_KEYS}
    compare_steps = {key: [] for key in DEFAULT_METRIC_KEYS}
    for step in sorted_steps:
        for key in DEFAULT_METRIC_KEYS:
            if key in metrics_by_step[step]:
                all_steps[key].append(metrics_by_step[step][key])
                if step > perf_skip_steps:
                    compare_steps[key].append(metrics_by_step[step][key])

    memory_peak_gb = parse_memory_peak_gb(memory_log)
    if memory_peak_gb is not None:
        all_steps["memory/peak_gb"] = [memory_peak_gb]
        compare_steps["memory/peak_gb"] = [memory_peak_gb]

    return {
        "run_id": datetime.now(timezone.utc).isoformat(),
        "model": "qwen_image_flowgrpo_single_sample",
        "n_gpus": n_gpus,
        "total_steps": total_steps,
        "perf_skip_steps": perf_skip_steps,
        "precision_steps": list(precision_steps),
        "steps_seen": sorted_steps,
        "metrics_all_steps": all_steps,
        "metrics_for_compare": compare_steps,
        "summaries_all_steps": {key: _summarize(values) for key, values in all_steps.items()},
        "summaries_for_compare": {key: _summarize(values) for key, values in compare_steps.items()},
    }


def _metric_mean(metrics: dict[str, Any], key: str) -> float | None:
    summary = metrics.get("summaries_for_compare", {}).get(key, {})
    value = summary.get("mean") if isinstance(summary, dict) else None
    return float(value) if value is not None else None


def compare_metrics(current: dict[str, Any], baseline: dict[str, Any], threshold: float) -> dict[str, Any]:
    report = {"threshold": threshold, "metrics": {}, "pass": True}
    for key in sorted(LOWER_IS_BETTER | HIGHER_IS_BETTER):
        current_mean = _metric_mean(current, key)
        baseline_mean = _metric_mean(baseline, key)
        if current_mean is None or baseline_mean is None:
            result = {
                "status": "missing",
                "current_mean": current_mean,
                "baseline_mean": baseline_mean,
                "pass": False,
            }
        else:
            ratio = current_mean / baseline_mean if baseline_mean != 0 else float("inf")
            if key in LOWER_IS_BETTER:
                passed = current_mean <= baseline_mean * (1.0 + threshold)
            else:
                passed = current_mean >= baseline_mean * (1.0 - threshold)
            result = {
                "status": "compared",
                "current_mean": current_mean,
                "baseline_mean": baseline_mean,
                "ratio": ratio,
                "pass": passed,
            }
        report["metrics"][key] = result
        report["pass"] = report["pass"] and result["pass"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--console-log", type=Path, required=True)
    parser.add_argument("--memory-log", type=Path, default=Path("outputs/debug_dumps/current/memory.log"))
    parser.add_argument("--out-json", type=Path, default=Path("outputs/debug_dumps/current/metrics.json"))
    parser.add_argument("--baseline-json", type=Path)
    parser.add_argument(
        "--compare-report-json",
        type=Path,
        default=Path("outputs/debug_dumps/current/perf_compare_report.json"),
    )
    parser.add_argument("--steps", default=os.environ.get("DEBUG_DUMP_STEPS", "1,2"))
    parser.add_argument("--total-steps", type=int, default=20)
    parser.add_argument("--n-gpus", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.05)
    args = parser.parse_args()

    metrics = build_metrics_json(
        console_log=args.console_log,
        memory_log=args.memory_log,
        precision_steps=parse_dump_steps(args.steps),
        total_steps=args.total_steps,
        n_gpus=args.n_gpus,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, sort_keys=True)

    if args.baseline_json:
        with args.baseline_json.open(encoding="utf-8") as file:
            baseline = json.load(file)
        report = compare_metrics(metrics, baseline, args.threshold)
        args.compare_report_json.parent.mkdir(parents=True, exist_ok=True)
        with args.compare_report_json.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, sort_keys=True)
        if not report["pass"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
