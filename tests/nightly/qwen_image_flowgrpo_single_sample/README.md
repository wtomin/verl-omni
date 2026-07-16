# Qwen-Image FlowGRPO Single-Sample Nightly

Last updated: 2026-07-16

This nightly runs 20 FlowGRPO training steps on `Qwen/Qwen-Image` with one fixed text-to-image sample.

- Steps in `DEBUG_DUMP_STEPS` (default `1,2`) dump driver tensors, actor forward outputs, and LoRA gradients.
- Performance comparison automatically skips steps `1..max(DEBUG_DUMP_STEPS)`.
- All debug logic is injected from this directory; production `verl_omni/` code is not modified.

## Run

```bash
NUM_GPUS=4 \
tests/nightly/qwen_image_flowgrpo_single_sample/run_qwen_image_flowgrpo_single_sample.sh
```

The wrapper creates `~/data/qwen_image_single/{train,test}.parquet`, writes dumps to
`outputs/debug_dumps/current`, and writes console/memory/metrics artifacts under the same directory.

## Baseline

To bootstrap a baseline from a known-good run:

```bash
BOOTSTRAP_BASELINE=1 \
tests/nightly/qwen_image_flowgrpo_single_sample/run_qwen_image_flowgrpo_single_sample.sh
```

Subsequent runs compare `outputs/debug_dumps/current` against `outputs/debug_dumps/baseline`.
Only update baselines from scheduled green runs on the same GPU topology.

## Standalone Tools

```bash
python tests/nightly/qwen_image_flowgrpo_single_sample/compare_dumps.py \
  --baseline-dir outputs/debug_dumps/baseline \
  --current-dir outputs/debug_dumps/current \
  --steps "${DEBUG_DUMP_STEPS:-1,2}"

python tests/nightly/qwen_image_flowgrpo_single_sample/collect_metrics.py \
  --console-log outputs/debug_dumps/current/console.log \
  --memory-log outputs/debug_dumps/current/memory.log \
  --baseline-json outputs/debug_dumps/baseline/metrics.json
```
