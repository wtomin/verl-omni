# Online DPO Training

This example runs direct-preference diffusion training with online rollout and
reward scoring. Unlike offline DPO, it does not consume pre-ranked win/lose
image pairs from parquet.

At each training step, online DPO:

- samples multiple candidate images for each prompt with vLLM-Omni rollout;
- scores the generated images through the configured reward function;
- forms one adjacent `[chosen, rejected]` pair per prompt by taking the
  highest- and lowest-scoring candidates;
- trains the existing diffusion DPO loss on those pairs.

Run Qwen-Image online DPO with:

```bash
bash examples/dpo_trainer/run_qwen_image_online_dpo_lora.sh \
  data.train_files=$WORKSPACE/data/ocr/qwen_image/train.parquet \
  data.val_files=$WORKSPACE/data/ocr/qwen_image/test.parquet
```

The initial implementation intentionally keeps pairing policy fixed to
top-vs-bottom. The only required sampling knob is
`actor_rollout_ref.rollout.n`, which must be at least `2` so each prompt has
enough candidates to form a preference pair.

The example disables True-CFG (`true_cfg_scale=1.0`) so the online dataset does
not need precomputed negative-prompt embeddings.
