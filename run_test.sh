#!/bin/bash
# 测试脚本启动器

echo "🔧 运行 Orion 模型测试..."
echo ""

# 直接使用 conda 环境的 python
cd /home/xinzhang/Documents/AI/Orion
/home/xinzhang/anaconda3/envs/orion/bin/python test_model_flow.py

echo ""
echo "✅ 测试完成！"

