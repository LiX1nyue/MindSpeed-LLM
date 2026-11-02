#!/bin/bash
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# Chakra 通算 DAG 导出示例（MCore 后端，Qwen3-8B，双卡 TP）。
#
# 这份脚本与 pretrain_qwen3_8b_4K_ptd_profile.sh 的差别只在 PROFILE_ARGS：
# 打开 profiling 的同时打开 --chakra-dag-export，训练结束后会自动产出
#   <profile_dir>/chakra_dag/{events_rank*, rank*.et, multi_rank.et,
#                             multi_rank_dag.json, manifest.json, export_report.json}
#
# 前置条件（在服务器上执行一次）：
#   pip install git+https://github.com/mlcommons/chakra.git
#   若只想看 JSON 图，可以跳过这步，用 --chakra-dag-format json。
#
# 注意：结构采集依赖 TorchDispatchMode，打开后有额外开销，只适合采集 1~2 个 step。

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=3600
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_ASD_ENABLE=0
export TASK_QUEUE_ENABLE=2

NPUS_PER_NODE=2
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# 按实际环境填写
CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"
PROFILE_DIR="./profile_dir"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# 采集配置要点：
#   --profile-ranks -1          必须采集所有 rank，否则跨卡 collective 无法合并
#   --profile-level level1      需要通信算子信息
#   --profile-with-cpu          需要 CPU 侧信息做 launch 归因
#   --profile-step-start/end    窗口尽量短（1 个 step），采集开销较大
PROFILE_ARGS="
    --profile \
    --profile-step-start 5 \
    --profile-step-end 6 \
    --profile-ranks -1 \
    --profile-level level1 \
    --profile-with-cpu \
    --profile-save-path $PROFILE_DIR \
    --chakra-dag-export \
    --chakra-dag-format both \
    --chakra-dag-fused-mode split-estimate
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 1 \
    --micro-batch-size 1 \
    --global-batch-size 2 \
    --seq-length 4096 \
    --train-iters 10 \
    --tokenizer-type NullTokenizer \
    --vocab-size 151936 \
    --num-layers 36 \
    --hidden-size 4096 \
    --ffn-hidden-size 12288 \
    --num-attention-heads 32 \
    --num-query-groups 8 \
    --group-query-attention \
    --normalization RMSNorm \
    --swiglu \
    --attention-softmax-in-fp32 \
    --no-gradient-accumulation-fusion \
    --norm-epsilon 1e-6 \
    --ckpt-format torch \
    --data-path $DATA_PATH \
    --tokenizer-path $TOKENIZER_PATH \
    --handler-name GeneralPretrainHandler \
    --workers 4 \
    --split 100,0,0 \
    --log-interval 1 \
    --save-interval 10 \
    --eval-interval 10 \
    --eval-iters 0 \
    --no-load-optim \
    --no-load-rng \
    --use-flash-attn \
    --transformer-impl local \
    --load $CKPT_LOAD_DIR \
    --save $CKPT_SAVE_DIR \
    --distributed-backend nccl \
    $PROFILE_ARGS \
    | tee logs/chakra_dag_demo.log

# 训练结束后，可用以下命令自检产出的 .et 是否可被 Chakra 读回：
#   python -m mindspeed_llm.tools.chakra_dag.cli --output $PROFILE_DIR/chakra_dag --check
#
# 用官方工具查看图：
#   chakra_jsonizer --input_filename $PROFILE_DIR/chakra_dag/multi_rank.et \
#                   --output_filename /tmp/multi_rank.json
#   chakra_visualizer --input_filename $PROFILE_DIR/chakra_dag/multi_rank.et \
#                     --output_filename /tmp/multi_rank.graphml
