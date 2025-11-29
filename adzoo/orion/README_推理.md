# ORION 推理快速指南

## ✅ 可用配置（RTX 4060 Ti 16GB）

### 推荐：轻量推理（不带LLM）- 已验证成功

```bash
cd /home/xinzhang/Documents/AI/Orion

bash adzoo/orion/orion_dist_eval_llm.sh \
    adzoo/orion/configs/orion_stage3_infer_light.py \
    ckpts/eva02_petr_proj.pth \
    1
```

**实测结果**：
- ✅ 推理成功：10个样本，~10秒
- ✅ 显存占用：8GB / 16GB
- ✅ 内存占用：12GB / 32GB
- ✅ 包含：3D检测、地图检测、运动预测

---

## ❌ 不可用配置

### 带LLM推理 - 显存不足

```bash
# ❌ 此配置无法在16GB显存上运行
bash adzoo/orion/orion_dist_eval_llm.sh \
    adzoo/orion/configs/orion_stage3_infer_llm_vqa_light.py \
    ckpts/eva02_petr_proj.pth \
    1
```

**原因**：
- LLM模型（FP16）：13.6GB
- 其他模型组件：3-5GB
- 总需求：18-20GB > 16GB ❌

**需要**：RTX 4090 24GB 或更大显存的GPU

---

## 📊 关键修复点

| 问题 | 解决方案 |
|------|---------|
| 训练配置用于推理 | 创建专门的推理配置 ✅ |
| 验证集数据损坏 | 使用训练集数据 ✅ |
| bbox_coder不匹配 | CustomNMSFreeCoder → NMSFreeCoder ✅ |
| 评估代码崩溃 | 添加安全访问 ✅ |
| LLM参数错误 | 删除enable_llm参数 ✅ |
| LLM显存不足 | 16GB卡无法运行 ⚠️ 硬件限制 |

---

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `orion_stage3_infer_light.py` | 推理配置（不带LLM）⭐推荐使用 |
| `orion_stage3_infer_llm_vqa_light.py` | 推理配置（带LLM）需24GB+显存 |
| `orion_dist_eval_llm.sh` | 推理脚本（带资源监控） |
| `quick_eval_llm.sh` | 快速启动（交互式） |

详细文档见其他 `.md` 文件。

---

**更新**: 2025-11-29  
**状态**: 轻量推理已验证可用

