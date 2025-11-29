#!/usr/bin/env bash

# ====================================
# ORION 快速推理脚本（带LLM）
# 提供常用推理场景的快捷命令
# ====================================

set -e

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}🚀 ORION 快速推理（带LLM和VQA）${NC}"
echo -e "${GREEN}================================${NC}"
echo ""

# ====================================
# 场景选择
# ====================================
echo "请选择推理场景："
echo "  1) 完整推理（带LLM和VQA）- 推荐用于完整评估"
echo "  2) 轻量推理（仅检测）- 不使用LLM，节省资源"
echo "  3) 调试模式（单样本）- 快速测试配置"
echo "  4) 自定义配置"
echo ""
read -p "请输入选项 [1-4]: " OPTION

case $OPTION in
    1)
        echo -e "${GREEN}选择：完整推理（带LLM和VQA）${NC}"
        echo "  - 使用配置: orion_stage3_infer_llm_vqa_light.py"
        echo "  - 资源需求: 12-14GB显存, 20-25GB内存"
        CONFIG="adzoo/orion/configs/orion_stage3_infer_llm_vqa_light.py"
        MODE="full"
        ;;
    2)
        echo -e "${YELLOW}选择：轻量推理（仅检测，不带LLM）${NC}"
        echo "  - 使用配置: orion_stage3_infer_light.py （推理配置）"
        echo "  - 资源需求: 7-10GB显存, 12-15GB内存"
        echo "  - 注意: 此配置不包含LLM和VQA功能"
        CONFIG="adzoo/orion/configs/orion_stage3_infer_light.py"
        MODE="light"
        ;;
    3)
        echo -e "${YELLOW}选择：调试模式${NC}"
        echo "  - 使用配置: orion_stage3_infer_llm_vqa_light.py"
        echo "  - 直接运行Python脚本（无后台监控）"
        CONFIG="adzoo/orion/configs/orion_stage3_infer_llm_vqa_light.py"
        MODE="debug"
        ;;
    4)
        echo -e "${YELLOW}选择：自定义配置${NC}"
        read -p "请输入配置文件路径: " CONFIG
        MODE="custom"
        ;;
    *)
        echo -e "${RED}❌ 无效选项${NC}"
        exit 1
        ;;
esac

# ====================================
# 权重文件选择
# ====================================
echo ""
echo "可用的权重文件："
echo "  1) eva02_petr_proj.pth（预训练权重）"
echo "  2) 自定义权重文件"
echo ""
read -p "请输入选项 [1-2]: " CKPT_OPTION

case $CKPT_OPTION in
    1)
        CHECKPOINT="ckpts/eva02_petr_proj.pth"
        ;;
    2)
        read -p "请输入权重文件路径: " CHECKPOINT
        ;;
    *)
        echo -e "${RED}❌ 无效选项${NC}"
        exit 1
        ;;
esac

# 检查文件是否存在
if [ ! -f "$CONFIG" ]; then
    echo -e "${RED}❌ 配置文件不存在: $CONFIG${NC}"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo -e "${RED}❌ 权重文件不存在: $CHECKPOINT${NC}"
    exit 1
fi

# ====================================
# 系统检查
# ====================================
echo ""
echo -e "${GREEN}📊 系统资源检查...${NC}"

TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
AVAILABLE_MEM=$(free -g | awk '/^Mem:/{print $7}')

echo "系统总内存: ${TOTAL_MEM}GB"
echo "可用内存: ${AVAILABLE_MEM}GB"

if command -v nvidia-smi &> /dev/null; then
    GPU_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1)
    GPU_MEM_GB=$((GPU_MEM / 1024))
    echo "GPU可用显存: ${GPU_MEM_GB}GB"
    
    # 根据模式检查资源
    if [ "$MODE" == "full" ]; then
        if [ "$AVAILABLE_MEM" -lt 15 ] || [ "$GPU_MEM_GB" -lt 12 ]; then
            echo -e "${YELLOW}⚠️  警告: 资源可能不足${NC}"
            echo "完整推理需要: 15GB+ 内存, 12GB+ 显存"
            read -p "是否继续? [y/N]: " CONTINUE
            if [ "$CONTINUE" != "y" ] && [ "$CONTINUE" != "Y" ]; then
                exit 0
            fi
        fi
    fi
fi

# ====================================
# 执行推理
# ====================================
echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}🚀 开始推理${NC}"
echo -e "${GREEN}================================${NC}"
echo "配置文件: $CONFIG"
echo "权重文件: $CHECKPOINT"
echo "推理模式: $MODE"
echo -e "${GREEN}================================${NC}"
echo ""

# 根据模式选择脚本
if [ "$MODE" == "full" ] || [ "$MODE" == "custom" ] || [ "$MODE" == "light" ]; then
    # 使用增强版推理脚本（带资源监控）
    bash "$SCRIPT_DIR/orion_dist_eval_llm.sh" "$CONFIG" "$CHECKPOINT" 1
elif [ "$MODE" == "debug" ]; then
    # 调试模式：直接运行Python脚本
    echo -e "${YELLOW}调试模式：直接运行推理（无后台监控）${NC}"
    python "$SCRIPT_DIR/test.py" \
        "$CONFIG" \
        "$CHECKPOINT" \
        --launcher=pytorch \
        --eval bbox \
        --debug
fi

echo ""
echo -e "${GREEN}✅ 推理完成！${NC}"

