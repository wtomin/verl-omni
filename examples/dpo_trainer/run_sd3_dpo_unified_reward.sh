# SD3 DPO training on PickScore prompts, vllm_omni rollout
set -x

# Set WORKSPACE to any writable directory; defaults to $HOME.
WORKSPACE=${WORKSPACE:-$HOME}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

pickscore_train_path=${PICKSCORE_TRAIN_PATH:-$SCRIPT_DIR/../../datasets/pickscore/train.txt}
pickscore_test_path=${PICKSCORE_TEST_PATH:-$SCRIPT_DIR/../../datasets/pickscore/test.txt}

model_name=stabilityai/stable-diffusion-3.5-medium
reward_model_name=CodeGoat24/UnifiedReward-2.0-qwen3vl-8b
reward_function_path=verl_omni/utils/reward_score/unified_reward.py

NUM_GPUS_ACTOR_ROLLOUT_REWARD=2
NUM_GPUS_ACTOR_ROLLOUT=1
ROLLOUT_TP=1
REWARD_TP=2

ENGINE=vllm_omni
REWARD_ENGINE=vllm

python3 -m verl_omni.trainer.diffusion.main_dpo \
    data=prompt_txt_data \
    data.train_files=$pickscore_train_path \
    data.val_files=$pickscore_test_path \
    trainer.resume_mode=disable \
    data.train_batch_size=4 \
    data.max_prompt_length=256 \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.rollout.pipeline.height=512 \
    actor_rollout_ref.rollout.pipeline.width=512 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=50 \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0','add_q_proj','add_k_proj','add_v_proj','to_add_out','img_mlp.net.0.proj','img_mlp.net.2','txt_mlp.net.0.proj','txt_mlp.net.2']" \
    actor_rollout_ref.actor.optim.lr=3e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=dpo \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.n=$NUM_GPUS_ACTOR_ROLLOUT \
    actor_rollout_ref.rollout.k_samples=4 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.pipeline.guidance_scale=4.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=256 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=50 \
    reward.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / REWARD_TP)) \
    reward.reward_model.enable=True \
    reward.reward_model.model_path=$reward_model_name \
    reward.reward_model.rollout.name=$REWARD_ENGINE \
    reward.reward_model.rollout.tensor_model_parallel_size=$REWARD_TP \
    reward.custom_reward_function.path=$reward_function_path \
    reward.custom_reward_function.name=compute_score_unified_reward \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=dpo \
    trainer.experiment_name=sd3_dpo_unified_reward \
    trainer.log_val_generations=8 \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT_REWARD \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=30 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=1000 "$@"
