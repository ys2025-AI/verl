#!/usr/bin/env bash
# =============================================================================
# Qwen3.6-35B-A3B 16-layer (reduced) + gsm8k | GRPO | Megatron | vLLM-Ascend NPU
# Router Replay (R3) 开/关吞吐测试 完整启动脚本（自包含，无需其他脚本）
#
# 模式（环境变量选择）:
#   ROUTING_REPLAY_MODE=R3      ENABLE_ROLLOUT_ROUTING_REPLAY=True    -> R3 开启
#   ROUTING_REPLAY_MODE=disabled ENABLE_ROLLOUT_ROUTING_REPLAY=False  -> R3 关闭
#
# 分支要求（BSHD R3 支持是 hybrid 模型的前置条件，见 runbook 13.2.2）:
#   优化前 = main + BSHD 修复        -> git checkout r3-npu-q36-before
#   优化后 = 3 个 perf 分支 + BSHD   -> git checkout perf/router-replay-npu-eval
#   （R3 off 模式对分支不敏感，任一分支均可）
#
# 用法示例:
#   # R3 ON（优化后分支）
#   git checkout perf/router-replay-npu-eval
#   ROUTING_REPLAY_MODE=R3 ENABLE_ROLLOUT_ROUTING_REPLAY=True TOTAL_TRAINING_STEPS=4 \
#       bash run_qwen36_16l_r3.sh 2>&1 | tee r3_on.log
#
#   # R3 OFF
#   ROUTING_REPLAY_MODE=disabled ENABLE_ROLLOUT_ROUTING_REPLAY=False TOTAL_TRAINING_STEPS=4 \
#       bash run_qwen36_16l_r3.sh 2>&1 | tee r3_off.log
# =============================================================================

set -xeuo pipefail

########################### 用户可调 ###########################
MODEL_PATH=${MODEL_PATH:-/vllm-workspace/reduced_models/Qwen3.6-35B-A3B-16layer}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-/vllm-workspace/data/gsm8k/train.parquet}
TEST_DATA_PATH=${TEST_DATA_PATH:-/vllm-workspace/data/gsm8k/test.parquet}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

ROUTING_REPLAY_MODE=${ROUTING_REPLAY_MODE:-R3}          # R3 | disabled
ENABLE_ROLLOUT_ROUTING_REPLAY=${ENABLE_ROLLOUT_ROUTING_REPLAY:-True}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
MICRO_BS=${MICRO_BS:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-4}
ROLLOUT_N=${ROLLOUT_N:-8}

ACTOR_TP=${ACTOR_TP:-1}
ACTOR_EP=${ACTOR_EP:-8}
ACTOR_ETP=${ACTOR_ETP:-1}
ROLLOUT_TP=${ROLLOUT_TP:-4}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.45}      # 必须 <= (HBM - actor峰值 - 余量)/HBM

ACTOR_LR=${ACTOR_LR:-1e-6}
PROJECT_NAME=${PROJECT_NAME:-verl_grpo_router_replay_qwen36_16l}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen36_16l_r3_${ROUTING_REPLAY_MODE}}
########################### end 用户可调 ###########################

########################### 前置检查 ###########################
# batch 可整除性: validate_config 要求 train_batch_size * rollout.n % DP == 0
DP=$(( NGPUS_PER_NODE / (ACTOR_TP * ACTOR_EP) ))
if [ $(( TRAIN_BATCH_SIZE * ROLLOUT_N % DP )) -ne 0 ]; then
    echo "ERROR: TRAIN_BATCH_SIZE(${TRAIN_BATCH_SIZE}) * ROLLOUT_N(${ROLLOUT_N}) % DP(${DP}) != 0" >&2
    exit 1
fi

for f in "$MODEL_PATH/config.json" "$TRAIN_DATA_PATH" "$TEST_DATA_PATH"; do
    [ -e "$f" ] || { echo "ERROR: missing $f" >&2; exit 1; }
done

if [ "${ROUTING_REPLAY_MODE}" != "disabled" ] && ! git branch --show-current | grep -q "npu-eval\|q36-before"; then
    echo "WARNING: 当前分支 $(git branch --show-current) 可能不含 BSHD R3 支持。" \
         "R3 + hybrid 模型需要 r3-npu-q36-before 或 perf/router-replay-npu-eval 分支（runbook 13.2.2）。" >&2
fi
########################### end 前置检查 ###########################

########################### Ascend NPU 环境 ###########################
export VERL_USE_EXTERNAL_MODULES=megatron_adaptor        # 4.2: nvidia_resiliency_ext __version__ 兼容
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DISABLE_L2_CACHE=1
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:1024
export CPU_AFFINITY_CONF=1
# vllm-ascend (practice recipe)
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
export VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_MQ_MAX_CHUNK_BYTES_MB=1
# v1/TQ + R3 必须
export TQ_ZERO_COPY_SERIALIZATION=False
export VLLM_USE_V2_MODEL_RUNNER=1                        # 12.2.1: R3 捕获要求 V2 runner
########################### end Ascend NPU 环境 ###########################

PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 2))}

########################### 参数数组 ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$TRAIN_DATA_PATH"
    data.val_files="$TEST_DATA_PATH"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.dataloader_num_workers=0
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=False     # hybrid 线性注意力不支持 THD（13.2.1，硬约束）
    actor_rollout_ref.model.use_fused_kernels=False
)

ACTOR=(
    model_engine=megatron                                # 4.3/4.4: 顶层字段，勿用 actor_rollout_ref.actor.model_engine
    actor_rollout_ref.actor.strategy=megatron
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.optim.use_precision_aware_optimizer=True
    actor_rollout_ref.actor.optim.main_grads_dtype=fp32
    actor_rollout_ref.actor.optim.exp_avg_dtype=fp32
    actor_rollout_ref.actor.optim.exp_avg_sq_dtype=fp32
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=0.40
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=False
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BS}
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${ACTOR_EP}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ACTOR_ETP}
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.optimizer_offload=True
    actor_rollout_ref.actor.megatron.grad_offload=True
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False
    actor_rollout_ref.actor.megatron.router_replay.mode=${ROUTING_REPLAY_MODE}
    # NPU 分支内核/显存配置（3.x 节）
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall
    +actor_rollout_ref.actor.megatron.override_ddp_config.grad_reduce_in_fp32=False
    # Qwen3.5/3.6 hybrid 配置（verl-ascend-recipe practice）
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_naive_l2norm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.01
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.001
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
    # R3 守卫要求 permute_fusion=False（启动守卫，见 int16 分支）
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
    actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BS}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.enable_rollout_routing_replay=${ENABLE_ROLLOUT_ROUTING_REPLAY}
    actor_rollout_ref.rollout.enforce_eager=True          # 10.5: torch 2.10.0+cpu 的 aot_compile 兼容问题
    actor_rollout_ref.rollout.skip_tokenizer_init=True
    # 12.2.2: RL 场景必须关 FRACTAL_NZ（rl_config 自动置 weight_nz_mode=0）
    +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.rl_config.enabled=true
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BS}
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${ACTOR_TP}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${ACTOR_EP}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ACTOR_ETP}
    actor_rollout_ref.ref.megatron.param_offload=True
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=False
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    trainer.val_before_train=False
    trainer.device=npu
    trainer.use_v1=True
    transfer_queue.enable=True
)
########################### end 参数数组 ###########################

# 可选：清理上一次异常退出残留（13.2.4: 先查 proc-mem，有残留 VLLMWorker 先 kill）
for c in 0 1 2 3 4 5 6 7; do
    for pid in $(npu-smi info -t proc-mem -i $c 2>/dev/null | grep -oE "Process id:[0-9]+" | cut -d: -f2 || true); do
        echo "WARNING: NPU card $c 仍有残留进程 $pid，尝试 kill -9" >&2
        kill -9 "$pid" 2>/dev/null || true
    done
done
ray stop --force >/dev/null 2>&1 || true
sleep 2

########################### 启动 ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "$@"
