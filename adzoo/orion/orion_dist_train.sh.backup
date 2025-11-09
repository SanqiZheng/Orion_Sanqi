#!/usr/bin/env bash

T=`date +%m%d%H%M`

# -------------------------------------------------- #
# Usually you only need to customize these variables #
CFG=$1                                               #
GPUS=$2                                              #
# -------------------------------------------------- #
# 使用if-else替代三元运算符，以兼容sh
define_gpus_per_node() {
    if [ $GPUS -lt 8 ]; then
        echo $GPUS
    else
        echo 8
    fi
}
GPUS_PER_NODE=$(define_gpus_per_node)
NNODES=`expr $GPUS / $GPUS_PER_NODE`

MASTER_PORT=${MASTER_PORT:-54621}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
RANK=${RANK:-0}

# 使用sh兼容的方式来获取WORK_DIR，避免${CFG%.*}的bash特定语法
WORK_DIR=$(echo $CFG | sed -e "s/\.[^.]*$//" -e "s/configs/work_dirs/g")/
# Intermediate files and logs will be saved to UniAD/projects/work_dirs/

if [ ! -d ${WORK_DIR}logs ]; then
    mkdir -p ${WORK_DIR}logs
fi

# Try running in non-distributed mode
# Set TORCH_DISTRIBUTED_DEBUG environment variable for detailed error info
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

# Run training script directly without distributed training
/home/xinzhang/anaconda3/envs/orion/bin/python3 $(dirname "$0")/train.py \
    $CFG \
    --work-dir=${WORK_DIR} \
    --seed=0 \
    --deterministic \
    --launcher=none \
    $3 $4 $5 $6 $7 $8 $9 \
    2>&1 | tee ${WORK_DIR}logs/train.$T