#!/usr/bin/env bash

# ====================================
# ORION 推理/评估脚本（增强版，带LLM）
# 基于 orion_dist_eval.sh，添加资源监控和保护
# 适用于带LLM和VQA的推理任务
# ====================================

set -e  # 遇到错误立即退出
set -u  # 使用未定义变量时报错
set -o pipefail  # 管道命令失败时退出

T=`date +%m%d%H%M`

# -------------------------------------------------- #
# 命令行参数
# -------------------------------------------------- #
CONFIG=${1:-}
CHECKPOINT=${2:-}
GPUS=${3:-1}
PORT=${PORT:-29503}

# 参数检查
if [ -z "$CONFIG" ] || [ -z "$CHECKPOINT" ]; then
    echo "用法: bash $0 <配置文件> <权重文件> [GPU数量=1]"
    echo ""
    echo "示例:"
    echo "  bash $0 configs/orion_stage3_infer_llm_vqa_light.py ckpts/stage3_model.pth 1"
    echo ""
    exit 1
fi

# 检查文件是否存在
if [ ! -f "$CONFIG" ]; then
    echo "❌ 错误: 配置文件不存在: $CONFIG"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ 错误: 权重文件不存在: $CHECKPOINT"
    exit 1
fi

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 工作目录
WORK_DIR=$(echo $CONFIG | sed -e "s/\.[^.]*$//" -e "s/configs/work_dirs/g")/
mkdir -p ${WORK_DIR}logs

# ====================================
# 清理函数
# ====================================
cleanup() {
    set +e
    local exit_code=${1:-$?}
    echo ""
    echo "================================"
    echo "🧹 清理资源..."
    
    # 终止推理进程
    if [ -n "${EVAL_PID:-}" ]; then
        echo "正在终止推理进程 (PID: $EVAL_PID)..."
        kill -TERM "$EVAL_PID" 2>/dev/null || true
        sleep 2
        if kill -0 "$EVAL_PID" 2>/dev/null; then
            kill -KILL "$EVAL_PID" 2>/dev/null || true
        fi
    fi
    
    # 清理监控进程
    if [ -n "${MONITOR_PID:-}" ]; then
        kill "$MONITOR_PID" 2>/dev/null || true
    fi
    
    # 清理GPU进程
    if command -v nvidia-smi &> /dev/null; then
        echo "清理GPU进程..."
        nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | \
            while read pid; do
                [ -n "$pid" ] && [ "$pid" != "$EVAL_PID" ] && kill -9 "$pid" 2>/dev/null || true
            done || true
    fi
    
    # 保存最终系统状态
    if [ -n "${WORK_DIR:-}" ]; then
        local snapshot_file="${WORK_DIR}logs/final_eval_state.log"
        {
            echo "================================"
            echo "最终系统状态快照"
            echo "时间: $(date)"
            echo "退出码: $exit_code"
            echo "================================"
            echo ""
            echo "内存状态:"
            free -h
            echo ""
            echo "GPU状态:"
            nvidia-smi 2>/dev/null || echo "无法获取GPU信息"
            echo ""
        } > "$snapshot_file" 2>&1
    fi
    
    sleep 1
    
    if [ $exit_code -ne 0 ]; then
        echo "❌ 推理异常退出 (退出码: $exit_code)"
    else
        echo "✅ 推理完成！"
    fi
    
    if [ -n "${WORK_DIR:-}" ]; then
        echo ""
        echo "📋 日志文件："
        echo "  - 推理日志: ${WORK_DIR}logs/eval.$T"
        echo "  - 资源监控: ${WORK_DIR}logs/eval_resource_monitor.log"
        echo "  - 警告信息: ${WORK_DIR}logs/eval_warnings.log"
        echo "  - 最终状态: ${WORK_DIR}logs/final_eval_state.log"
    fi
    echo "================================"
    exit $exit_code
}

# 注册清理函数
trap 'cleanup $?' EXIT
trap 'cleanup 130' INT
trap 'cleanup 143' TERM

# ====================================
# 系统资源检查
# ====================================
echo "================================"
echo "🚀 ORION 推理启动（带LLM和VQA）"
echo "================================"
echo "配置文件: $CONFIG"
echo "权重文件: $CHECKPOINT"
echo "GPU数量: $GPUS"
echo "工作目录: $WORK_DIR"
echo "================================"
echo ""

echo "📊 检查系统资源..."

# 检查内存
TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
AVAILABLE_MEM=$(free -g | awk '/^Mem:/{print $7}')
echo "系统总内存: ${TOTAL_MEM}GB"
echo "可用内存: ${AVAILABLE_MEM}GB"

if [ "$AVAILABLE_MEM" -lt 15 ]; then
    echo "⚠️  警告: 可用内存不足 (${AVAILABLE_MEM}GB < 15GB)"
    echo "推理LLM需要较多内存，建议："
    echo "  1. 关闭其他应用释放内存"
    echo "  2. 或运行不带LLM的轻量配置"
    echo ""
    echo "ℹ️  推理将自动继续（如需取消，请按 Ctrl+C）"
    sleep 3
fi

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU信息:"
    nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv
    echo ""
    
    GPU_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1)
    GPU_MEM_GB=$((GPU_MEM / 1024))
    echo "GPU-0 可用显存: ${GPU_MEM_GB}GB"
    
    if [ "$GPU_MEM_GB" -lt 12 ]; then
        echo "⚠️  警告: GPU显存可能不足 (${GPU_MEM_GB}GB < 12GB)"
        echo "推理LLM需要约12-14GB显存"
    fi
else
    echo "⚠️  警告: 未检测到nvidia-smi"
fi

echo ""
echo "================================"

# ====================================
# 环境变量设置
# ====================================
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONPATH="$PROJECT_ROOT":$PYTHONPATH

# GPU显存优化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

# CPU内存优化
export MALLOC_TRIM_THRESHOLD_=100000
export MALLOC_MMAP_THRESHOLD_=100000

# 多线程限制
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# 设置CUDA设备
if [ "$GPUS" -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

# 分布式环境变量
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${PORT}
export RANK=${RANK:-0}
export LOCAL_RANK=${LOCAL_RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

# ====================================
# 资源监控函数
# ====================================
monitor_resources() {
    local log_file="$1"
    local warning_file="$2"
    local kill_on_pressure="${3:-true}"
    
    while true; do
        local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        
        # 获取内存使用率
        local mem_info=$(free | grep Mem)
        local mem_total=$(echo $mem_info | awk '{print $2}')
        local mem_used=$(echo $mem_info | awk '{print $3}')
        local mem_percent=$((mem_used * 100 / mem_total))
        local mem_available=$(free -m | grep Mem | awk '{print $7}')
        
        # 详细日志
        {
            echo "=== $timestamp ==="
            echo "内存使用: ${mem_percent}% (可用: ${mem_available}MB)"
            free -h | grep -E "^Mem|^Swap"
            echo ""
            
            # GPU状态
            if command -v nvidia-smi &> /dev/null; then
                echo "GPU状态:"
                nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo "无法获取"
                echo ""
            fi
            
            # Python进程
            echo "Python进程内存:"
            ps aux | grep -E "[p]ython.*test.py" | awk '{printf "PID:%s MEM:%s%% RSS:%sMB CMD:%s\n", $2, $4, $6/1024, $11}' || echo "无"
            echo ""
            
        } >> "$log_file" 2>&1
        
        # 检查内存压力
        if [ "$mem_percent" -gt 90 ]; then
            echo "[$timestamp] 🚨 严重警告：内存使用率 ${mem_percent}% > 90%" | tee -a "$warning_file"
            
            if [ "$kill_on_pressure" = "true" ]; then
                echo "[$timestamp] 🛑 内存压力过大，为防止系统死机，主动终止推理" | tee -a "$warning_file"
                if [ -n "${EVAL_PID:-}" ]; then
                    kill -TERM "$EVAL_PID" 2>/dev/null || true
                fi
                break
            fi
        elif [ "$mem_percent" -gt 80 ]; then
            echo "[$timestamp] ⚠️  警告：内存使用率 ${mem_percent}% > 80%" | tee -a "$warning_file"
        fi
        
        # 检查可用内存
        if [ "$mem_available" -lt 3000 ]; then
            echo "[$timestamp] 🚨 严重警告：可用内存不足 ${mem_available}MB" | tee -a "$warning_file"
            
            if [ "$kill_on_pressure" = "true" ] && [ "$mem_available" -lt 2000 ]; then
                echo "[$timestamp] 🛑 可用内存过低，主动终止推理" | tee -a "$warning_file"
                if [ -n "${EVAL_PID:-}" ]; then
                    kill -TERM "$EVAL_PID" 2>/dev/null || true
                fi
                break
            fi
        fi
        
        sleep 5  # 每5秒检查一次
    done
}

# 启动资源监控
MONITOR_LOG="${WORK_DIR}logs/eval_resource_monitor.log"
WARNING_LOG="${WORK_DIR}logs/eval_warnings.log"
echo "资源监控启动于: $(date)" > "$MONITOR_LOG"
echo "警告日志启动于: $(date)" > "$WARNING_LOG"

monitor_resources "$MONITOR_LOG" "$WARNING_LOG" true &
MONITOR_PID=$!

echo "✅ 资源监控已启动 (PID: $MONITOR_PID)"
echo "   监控日志: $MONITOR_LOG"
echo "   警告日志: $WARNING_LOG"
echo ""

# ====================================
# 运行推理脚本
# ====================================
echo "🚀 启动推理进程..."
echo ""

# 构建推理命令
EVAL_LOG="${WORK_DIR}logs/eval.$T"

echo "推理日志: $EVAL_LOG"
echo ""

# 运行推理
/home/xinzhang/anaconda3/envs/orion/bin/python3 -u $SCRIPT_DIR/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher=pytorch \
    --eval bbox \
    ${@:4} \
    2>&1 | tee -a ${EVAL_LOG} &

EVAL_PID=$!

echo "推理进程已启动 (PID: $EVAL_PID)"
echo ""

# 等待推理完成
wait $EVAL_PID
EVAL_EXIT_CODE=$?

# 停止监控
kill $MONITOR_PID 2>/dev/null || true

# 检查退出状态
if [ $EVAL_EXIT_CODE -ne 0 ]; then
    echo "❌ 推理失败 (退出码: $EVAL_EXIT_CODE)"
    exit $EVAL_EXIT_CODE
fi

echo "✅ 推理成功完成！"

