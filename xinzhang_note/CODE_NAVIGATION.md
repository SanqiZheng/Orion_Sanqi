# ORION 代码导航

本文档帮助快速定位关键功能的代码实现位置。

## 核心文件结构

### 1. 主模型定义
- **`mmcv/models/detectors/orion.py`**: Orion主模型类
  - 第67-124行: 模型初始化
  - 第441-501行: `forward_train()` - 训练前向传播
  - 第689-959行: `forward_test()` / `simple_test()` - 推理
  - 第1388-1421行: `distribution_forward()` - VAE前向传播
  - 第1171-1188行: `future_states_predict()` - 未来状态预测

### 2. QT-Former (Query-based Temporal Transformer)
- **`mmcv/models/dense_heads/orion_head.py`**: 检测头，包含QT-Former逻辑
  - 第550-576行: `temporal_alignment()` - 时空对齐
  - 第507-548行: `post_update_memory()` - 更新memory bank
  - 第709-890行: `forward()` - 前向传播，包含query propagation
  - 第741-756行: Scene-level memory处理（如果启用use_memory）

- **`mmcv/models/utils/petr_transformers.py`**:
  - 第119-192行: `PETRTransformerDecoderLayer` - Transformer decoder层
  - 第160-192行: `forward()` - 包含temporal self-attention逻辑

### 3. LLM融合 (Scene Tokens + History + Instruction)
- **`mmcv/utils/llava_arch.py`**:
  - 第49-153行: `prepare_inputs_labels_for_multimodal()` - 融合图像和文本token
  - 关键逻辑：将IMAGE_TOKEN_INDEX替换为image_features

- **`mmcv/utils/llava_llama.py`**:
  - 第83-198行: `forward()` - LLM前向传播
  - 第141-154行: `return_ego_feature=True`时提取planning token
  - 第242-300行: `inference_ego()` - 推理时提取planning token

- **`mmcv/models/detectors/orion.py`**:
  - 第564-687行: `forward_pts_train()` - 训练时融合scene tokens和LLM输入
  - 第765-898行: `simple_test_pts()` - 推理时的融合逻辑

### 4. Planning-QA 模板
- **`mmcv/datasets/pipelines/transforms_3d.py`**:
  - 第870-1068行: `LoadAnnoatationVQA` - 数据预处理
  - 第1022-1028行: Planning-QA模板定义
  - 第1048-1049行: 添加`<waypoint_ego>`特殊token

- **`mmcv/models/detectors/orion.py`**:
  - 第197-199行: 添加特殊token到tokenizer和model
  - 第783-793行: 推理时使用planning-QA模板

### 5. VAE实现 (Generative Planner)
- **`mmcv/models/utils/distributions.py`**:
  - 第9-44行: `DistributionModule` - VAE编码器
  - 第78-94行: `DistributionEncoder1DV2` - 编码器网络
  - 第114-130行: `PredictModel` - VAE解码器（GRU-based）
  - 第132-153行: `ProbabilisticLoss` - KL散度损失

- **`mmcv/models/detectors/orion.py`**:
  - 第211-249行: VAE组件初始化
  - 第1388-1421行: `distribution_forward()` - VAE前向传播
  - 第1171-1188行: `future_states_predict()` - 从latent space预测未来状态
  - 第595-600行: 从future states解码轨迹

### 6. Reasoning Space → Action Space 映射
- **`mmcv/models/detectors/orion.py`**:
  - 第577-616行: `loss_planning()` - Planning损失计算
  - 第1388-1421行: `distribution_forward()` - 从reasoning space到latent space
  - 第1171-1188行: `future_states_predict()` - 从latent space到future states
  - 第595-600行: 从future states到action space（轨迹坐标）

## 关键数据流

### 训练时数据流
```
多视角图像 → extract_img_feat() 
  → img_feats (B, N, C, H, W)
  → pts_bbox_head.forward() 
    → temporal_alignment() [QT-Former压缩历史]
    → transformer() [当前query + 历史query]
    → det_query (B, num_query, 4096)
  → map_head.forward()
    → map_query (B, num_map_query, 4096)
  → vision_embeded = cat([det_query, map_query]) (B, 513, 4096)
  → lm_head.forward()
    → prepare_inputs_labels_for_multimodal() [融合scene tokens + text]
    → LLM处理
    → ego_feature (B, 4096) [planning token]
  → distribution_forward()
    → present_distribution(ego_feature) → (mu, log_sigma)
    → sample = mu + sigma * noise
  → future_states_predict()
    → predict_model(sample, hidden_state) → future_states
  → ego_fut_decoder()
    → ego_fut_preds (B, 6, 6, 2)
```

### 推理时数据流
```
输入: images, input_ids (包含planning-QA)
  → vision_embeded (scene tokens)
  → lm_head.inference_ego()
    → 生成planning-QA回答
    → 提取<waypoint_ego> token的embedding → ego_feature
  → distribution_forward()
    → present_distribution(ego_feature) → (mu, log_sigma)
    → sample = mu + sigma * noise [推理时从present分布采样]
  → future_states_predict()
    → predict_model(sample, hidden_state) → future_states
  → ego_fut_decoder()
    → ego_fut_preds (B, 6, 6, 2)
```

## 关键参数

### QT-Former相关
- `num_propagated`: 传播的历史query数量（默认256）
- `memory_len`: Memory bank最大长度（默认1024）
- `use_memory`: 是否启用scene-level memory（默认False）
- `num_memory`: Scene memory的query数量（默认16）

### LLM相关
- `use_gen_token`: 是否使用planning token（默认False）
- `use_critical_qa`: 是否使用critical QA（默认False）
- `lm_head`: LLM模型配置路径

### VAE相关
- `latent_dim`: 潜在空间维度（默认32）
- `present_distribution_in_channels`: Present分布输入维度（4096）
- `future_distribution_in_channels`: Future分布输入维度（4096+12）
- `ego_fut_mode`: 轨迹模式数量（默认6）
- `fut_ts`: 未来时间步数（默认6）

## 调试技巧

1. **查看memory bank状态**:
   - 在 `orion_head.py` 的 `post_update_memory()` 中打断点
   - 检查 `self.memory_embedding`, `self.memory_timestamp` 等

2. **查看planning token提取**:
   - 在 `llava_llama.py` 的 `inference_ego()` 第297行打断点
   - 检查 `loc_positions` 和 `selected_hidden_states`

3. **查看VAE采样过程**:
   - 在 `orion.py` 的 `distribution_forward()` 第1410行打断点
   - 检查 `sample`, `mu`, `sigma` 的值

4. **查看轨迹解码**:
   - 在 `orion.py` 的 `forward_pts_train()` 第595行打断点
   - 检查 `ego_fut_preds` 的形状和值

## 常见配置

### 训练配置
- Stage 1: `adzoo/orion/configs/orion_stage1_train.py` - VQA预训练
- Stage 2: `adzoo/orion/configs/orion_stage2_train.py` - Planning训练
- Stage 3: `adzoo/orion/configs/orion_stage3_train.py` - 端到端训练

### 推理配置
- `adzoo/orion/configs/orion_stage3_infer.py` - 标准推理
- `adzoo/orion/configs/orion_stage3_cot.py` - Chain-of-Thought推理
- `adzoo/orion/configs/orion_stage3_fp16.py` - FP16推理

