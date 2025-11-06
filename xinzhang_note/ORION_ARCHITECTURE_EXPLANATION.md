# ORION 架构详解

## 1. 整体架构

ORION 是一个端到端的自动驾驶框架，主要包含三个核心组件：

1. **QT-Former (Query-based Temporal Transformer)**: 聚合多视角和长期历史信息
2. **LLM (Large Language Model)**: 进行驾驶场景推理
3. **Generative Planner (生成式规划器)**: 精确轨迹预测

整体流程：
- 多视角图像 → 视觉编码器 → QT-Former压缩历史信息 → Scene Tokens
- Scene Tokens + History Tokens + User Instruction → LLM
- LLM输出Planning Token → VAE解码器 → 轨迹预测

## 2. QT-Former 的作用与设计思想

### 2.1 核心作用

QT-Former 用于**query-based压缩多视角 + 长期历史**信息，主要实现位置在 `mmcv/models/dense_heads/orion_head.py`。

### 2.2 关键实现：Temporal Alignment

在 `temporal_alignment()` 方法中（第550-576行）：

```python
def temporal_alignment(self, query_pos, tgt, reference_points):
    # 1. 从memory bank中提取历史query的reference points
    temp_reference_point = (self.memory_reference_point - self.pc_range[:3]) / (self.pc_range[3:6] - self.pc_range[0:3])
    
    # 2. 生成历史query的位置编码
    temp_pos = self.query_pos(nerf_positional_encoding(temp_reference_point.repeat(1, 1, self.n_control)))
    temp_memory = self.memory_embedding  # 历史query的特征
    
    # 3. 如果启用ego pose对齐，进行时空对齐
    if self.with_ego_pos:
        # 计算ego motion的编码
        rec_ego_motion = ...
        memory_ego_motion = ...
        temp_pos = self.ego_pose_pe(temp_pos, memory_ego_motion)
    
    # 4. 添加时间编码
    query_pos += self.time_embedding(pos2posemb1d(torch.zeros_like(reference_points[...,:1])))
    temp_pos += self.time_embedding(pos2posemb1d(self.memory_timestamp).float())
    
    # 5. 将历史query与当前query拼接（query propagation）
    if self.num_propagated > 0:
        tgt = torch.cat([tgt, temp_memory[:, :self.num_propagated]], dim=1)
        query_pos = torch.cat([query_pos, temp_pos[:, :self.num_propagated]], dim=1)
        reference_points = torch.cat([reference_points, temp_reference_point[:, :self.num_propagated]], dim=1)
```

### 2.3 Query-based 压缩机制

1. **Memory Bank**: 维护历史query的embedding (`memory_embedding`), reference points (`memory_reference_point`), 时间戳 (`memory_timestamp`), ego pose (`memory_egopose`)等

2. **Query Propagation**: 通过 `num_propagated` 参数控制有多少历史query被传播到当前帧

3. **Temporal Self-Attention**: 在Transformer Decoder中，当前query和历史query通过self-attention机制交互（见 `petr_transformers.py` 第173-180行）：
   ```python
   if temp_memory is not None:
       temp_key = temp_value = torch.cat([query, temp_memory], dim=1)
       temp_pos = torch.cat([query_pos, temp_pos], dim=1)
   ```

4. **Scene-level Memory (可选)**: 如果启用 `use_memory`，还会维护scene-level的memory（`memory_scene_query`），用于更高层次的场景压缩

### 2.4 多视角处理

多视角图像特征通过 `extract_img_feat()` 提取，然后reshape为 `(B, N, C, H, W)`，其中N是视角数量。在Transformer中，这些特征作为 `memory`（key/value）与query进行cross-attention。

## 3. LLM中如何融合 Scene Tokens + History Tokens + User Instruction

### 3.1 Token准备

在 `orion.py` 的 `forward_pts_train()` 方法中（第564-566行）：

```python
if self.with_lm_head:
    # 1. 获取检测query和地图query（scene tokens）
    vision_embeded_obj = det_query.clone()  # 检测query: (B, num_query, 4096)
    vision_embeded_map = map_query.clone()  # 地图query: (B, num_map_query, 4096)
    
    # 2. 拼接成完整的vision embedding
    vision_embeded = torch.cat([vision_embeded_obj, vision_embeded_map], dim=1)  # (B, 513, 4096)
    
    # 3. 与LLM输入融合
    vlm_loss, ego_feature = self.lm_head(
        input_ids=input_ids,  # User instruction (文本token)
        attention_mask=vlm_attn_mask,
        labels=vlm_labels,
        images=vision_embeded,  # Scene tokens
        use_cache=False,
        return_ego_feature=True
    )
```

### 3.2 LLM中的融合机制

在 `llava_arch.py` 的 `prepare_inputs_labels_for_multimodal()` 方法中（第49-153行）：

1. **图像Token替换**: 
   - 输入序列中的 `IMAGE_TOKEN_INDEX` 被替换为对应的 `image_features`（scene tokens）
   - 每个图像token对应一段视觉特征序列

2. **位置编码**: 
   - 图像特征和文本token共享相同的位置编码空间
   - 通过 `position_ids` 统一管理

3. **Attention机制**: 
   - LLM的self-attention机制自然地将scene tokens和text tokens融合
   - Scene tokens可以attend到user instruction，反之亦然

### 3.3 历史信息的融合

历史信息通过以下方式融入：
- **Scene Memory**: 如果启用 `use_memory`，历史scene query会被压缩并融入当前scene tokens
- **对话历史**: `input_ids` 可以包含多轮对话历史（见 `simple_test_pts()` 第766-782行）

## 4. Planning-QA 模板与 Planning Token

### 4.1 Planning-QA 模板定义

在 `mmcv/datasets/pipelines/transforms_3d.py` 的 `LoadAnnoatationVQA` 类中（第1022-1028行）：

```python
if self.use_gen_token:
    planning_qa = [
        [{"from": 'human',
        "value": "Based on the above information, please provide a safe, executable, and reasonable planning trajectory for the ego car."},
        {"from": 'gpt',
        "value": "Here is the planning trajectory <waypoint_ego>"}]
    ]
```

### 4.2 Special Token 添加

在 `orion.py` 初始化时（第197-199行）：

```python
if use_gen_token:
    add_special_token([EGO_WAYPOINT_TOKEN], tokenizer=self.tokenizer, model=self.lm_head)
    self.lm_head.config.waypoint_token_idx = self.tokenizer(EGO_WAYPOINT_TOKEN, add_special_tokens=False).input_ids[0]
```

### 4.3 LLM输出Planning Token

在推理时（`orion.py` 第783-793行）：

```python
ego_feature = self.lm_head.inference_ego(
    inputs=context_input_ids,  # 包含planning-QA的完整对话
    images=vision_embeded,
    do_sample=True,
    temperature=0.1,
    top_p=0.75,
    num_beams=1,
    max_new_tokens=320,
    use_cache=True,
    return_ego_feature=True  # 返回planning token的embedding
)
```

在 `llava_llama.py` 的 `inference_ego()` 方法中（第297-300行），提取planning token：

```python
if return_ego_feature:
    loc_positions = (new_input_ids == self.config.waypoint_token_idx)
    selected_hidden_states = hidden_states[loc_positions]  # 提取planning token的embedding
```

## 5. Generative Planner中VAE的具体实现

### 5.1 VAE架构

VAE实现位于 `mmcv/models/utils/distributions.py` 和 `orion.py`：

#### DistributionModule (编码器)

```python
class DistributionModule(nn.Module):
    def __init__(self, in_channels, latent_dim, min_log_sigma, max_log_sigma):
        # 1. 压缩层：将4096维压缩到2048维
        self.encoder = DistributionEncoder1DV2(in_channels, compress_dim)
        
        # 2. 输出层：输出mu和log_sigma
        self.last_conv = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(compress_dim, 2 * latent_dim, kernel_size=1)
        )
    
    def forward(self, s_t):
        encoding = self.encoder(s_t.permute(0, 2, 1))
        mu_log_sigma = self.last_conv(encoding).permute(0, 2, 1)
        mu = mu_log_sigma[:, :, :latent_dim]
        log_sigma = mu_log_sigma[:, :, latent_dim:]
        log_sigma = torch.clamp(log_sigma, min_log_sigma, max_log_sigma)
        return mu, log_sigma
```

#### PredictModel (解码器)

```python
class PredictModel(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers):
        self.gru = nn.GRU(input_size=in_channels, hidden_size=hidden_channels, num_layers=num_layers)
        self.linear1 = nn.Linear(hidden_channels, hidden_channels*2)
        self.linear2 = nn.Linear(hidden_channels*2, hidden_channels*4)
        self.linear3 = nn.Linear(hidden_channels*4, out_channels)
    
    def forward(self, x, h):
        x, h = self.gru(x, h)  # GRU处理时序信息
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x
```

### 5.2 VAE前向过程

在 `orion.py` 的 `distribution_forward()` 方法中（第1388-1421行）：

```python
def distribution_forward(self, present_features, future_distribution_inputs=None, noise=None):
    # 1. Present分布：从当前planning token编码
    present_mu, present_log_sigma = self.present_distribution(present_features)
    
    # 2. Future分布（仅训练时）：从planning token + GT轨迹编码
    if future_distribution_inputs is not None:
        future_features = torch.cat([present_features, future_distribution_inputs], dim=2)
        future_mu, future_log_sigma = self.future_distribution(future_features)
    
    # 3. 采样
    if noise is None:
        noise = torch.randn_like(present_mu)
    
    if self.training:
        mu = future_mu  # 训练时使用future分布
        sigma = torch.exp(future_log_sigma)
    else:
        mu = present_mu  # 推理时使用present分布
        sigma = torch.exp(present_log_sigma)
    
    sample = mu + sigma * noise  # 重参数化技巧
    
    return sample, output_distribution
```

### 5.3 未来状态预测

在 `future_states_predict()` 方法中（第1171-1188行）：

```python
def future_states_predict(self, batch_size, sample, hidden_states, current_states):
    # 1. 将latent sample扩展到6个时间步
    future_prediction_input = sample.unsqueeze(0).expand(self.fut_ts, -1, -1, -1)
    future_prediction_input = future_prediction_input.reshape(self.fut_ts, -1, self.latent_dim)
    
    # 2. 准备GRU的hidden state（从planning token的4层特征中提取）
    hidden_states = hidden_states.permute(1,0,2)  # (4, B, 4096) -> (B, 4, 4096)
    hidden_state = hidden_states.reshape(self.layer_dim, -1, int(4096/4))  # (4, B, 1024)
    
    # 3. 通过GRU预测未来状态
    future_states = self.predict_model(future_prediction_input, hidden_state)
    
    # 4. 拼接当前状态和未来状态
    current_states_hs = current_states.unsqueeze(0).repeat(6, 1, 1, 1)
    future_states_hs = future_states.reshape(self.fut_ts, batch_size, -1, future_states.shape[2])
    
    if self.with_cur:
        states_hs = torch.cat((current_states_hs, future_states_hs), dim=-1)
    
    return states_hs, future_states_hs
```

### 5.4 轨迹解码

在 `forward_pts_train()` 中（第595-600行）：

```python
# 从future states解码轨迹
ego_query_hs = states_hs[:, :, 0, :].unsqueeze(1).permute(0, 2, 1, 3)
ego_fut_trajs_list = []
for i in range(self.fut_ts):
    outputs_ego_trajs = self.ego_fut_decoder(ego_query_hs[i]).reshape(B, self.ego_fut_mode, 2)
    ego_fut_trajs_list.append(outputs_ego_trajs)

ego_fut_preds = torch.stack(ego_fut_trajs_list, dim=2)  # (B, ego_fut_mode, fut_ts, 2)
```

## 6. Reasoning Space 到 Action Space 的映射

### 6.1 映射流程

1. **Reasoning Space (LLM)**: 
   - Planning token的embedding: `ego_feature` (4096维)
   - 语义层面的推理结果

2. **Latent Space (VAE Encoder)**:
   - `present_distribution(ego_feature)` → `(mu, log_sigma)` (32维)
   - 将语义推理压缩到低维潜在空间

3. **Future States (VAE Decoder)**:
   - `predict_model(sample, hidden_state)` → `future_states` (4096维)
   - 从潜在空间解码到未来状态表示

4. **Action Space (Trajectory Decoder)**:
   - `ego_fut_decoder(future_states)` → `ego_fut_preds` (6个时间步 × 6个模式 × 2维坐标)
   - 最终输出轨迹点坐标

### 6.2 关键设计

1. **两阶段分布**:
   - **Present分布**: 推理时使用，从planning token编码
   - **Future分布**: 训练时使用，从planning token + GT轨迹编码，提供更强的监督

2. **KL散度损失**: 
   - 在 `ProbabilisticLoss` 中（`distributions.py` 第132-153行），通过KL散度约束present分布接近future分布，实现reasoning space和action space的对齐

3. **多模态输出**: 
   - `ego_fut_mode=6` 表示每个时间步输出6个候选轨迹模式
   - 通过分类头选择最佳模式

### 6.3 训练策略

- **混合训练**: 通过 `mix_qa_training` 参数，可以混合VQA任务和planning任务
- **端到端优化**: VLM loss和planning loss联合优化，实现reasoning space和action space的统一学习

## 总结

ORION的核心创新在于：
1. **QT-Former**: 通过query-based机制高效压缩多视角和长期历史
2. **Planning-QA模板**: 将规划任务转化为语言模型任务，输出planning token
3. **VAE-based生成器**: 将语义推理（reasoning space）映射到精确轨迹（action space）
4. **端到端对齐**: 通过联合训练实现reasoning和action的统一优化

这种设计使得模型能够利用LLM的强大推理能力，同时保证输出的轨迹在数值空间中的精确性。

