## Pedestrian Trajectory Prediction Module

轻量级行人轨迹预测模块示例，主要用于地下车库 / 低速场景的人车交互研究。
本目录下代码是**自洽的训练 + 推理 pipeline**，暂不与现有 C++ 工程做适配。

### 1. 功能概述

- 输入：行人在车体坐标系（或任意平面坐标系）下的历史轨迹  
  `past` 形状为 `N x T_obs x 2`（x, y）。
- 输出：未来多模态轨迹预测  
  `K` 条候选未来轨迹，形状 `N x K x T_pred x 2`，以及每条轨迹的概率 `N x K`。
- 模型：轻量级 GRU + MLP，多模态 Best-of-K 训练，适合在自动泊车 ECU 或仿真服务器上部署。

### 2. 目录结构

- `ped_pred/`
  - `models/trajectory_predictor.py`：核心预测模型（多模态轨迹预测）
  - `data/dataset.py`：基础数据集定义，使用 `.npz` 文件存储
  - `config.py`：超参数配置
- `train.py`：训练脚本
- `infer.py`：离线推理脚本
- `requirements.txt`：Python 依赖列表

### 3. 数据格式

训练 / 推理脚本默认使用 `.npz` 文件，包含至少以下字段：

- `past`: `float32`，形状 `[N, T_obs, 2]`
- `future`: `float32`，形状 `[N, T_pred, 2]`

可选字段（用于 mask 或 id）:

- `past_mask`: `bool`/`uint8`，形状 `[N, T_obs]`，1 表示有效点
- `future_mask`: `bool`/`uint8`，形状 `[N, T_pred]`
- 你可以在 `ped_pred/data/dataset.py` 中根据自己数据集扩展更多字段（如车道线、停车位拓扑等）。

### 4. 安装依赖

```bash
pip install -r pedestrian_prediction/requirements.txt
```

> 依赖默认包括：`torch`, `numpy`, `tqdm`, `pyyaml`。如需 GPU 训练，请根据环境单独安装合适版本的 PyTorch。

### 5. 训练示例

```bash
python pedestrian_prediction/train.py \
  --train_npz path/to/train.npz \
  --val_npz path/to/val.npz \
  --output_dir work_dirs/ped_pred_exp1 \
  --epochs 50 \
  --batch_size 256
```

训练完成后，会在 `output_dir` 下保存：

- `config.yaml`：运行时配置
- `best_model.pt`：验证集指标最佳的模型权重

### 6. 推理示例

```bash
python pedestrian_prediction/infer.py \
  --model_ckpt work_dirs/ped_pred_exp1/best_model.pt \
  --input_npz path/to/test.npz \
  --output_npz path/to/test_pred.npz
```

输出文件中将包含：

- `pred_trajs`: `[N, K, T_pred, 2]`
- `pred_probs`: `[N, K]`

你可以在现有 C++ 工程中按需要对接该接口，例如只取 `K=3` 中概率最大的那一条轨迹作为规划输入。

